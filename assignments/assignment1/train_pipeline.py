import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as nnf
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from torchaudio import functional as AF
from torchaudio.datasets import SPEECHCOMMANDS

from melbanks import LogMelFilterBanks


TARGET_SR = 16000
TARGET_LEN = 16000
LABEL2ID = {"no": 0, "yes": 1}


class YesNoCommands(SPEECHCOMMANDS):
    def __init__(self, root: str, subset: str):
        super().__init__(root=root, download=True, subset=subset)
        self._walker = [w for w in self._walker if Path(w).parent.name in LABEL2ID]


def validate_logmel_layer():
    signal = torch.randn(1, TARGET_LEN)
    melspec = torchaudio.transforms.MelSpectrogram(
        hop_length=160,
        n_mels=80,
    )(signal)
    logmelbanks = LogMelFilterBanks()(signal)

    assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
    assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def parse_int_list(raw: str):
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list of integers")
    return values


def prepare_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    wav = waveform.mean(dim=0)
    if sample_rate != TARGET_SR:
        wav = AF.resample(wav, sample_rate, TARGET_SR)

    if wav.numel() < TARGET_LEN:
        wav = nnf.pad(wav, (0, TARGET_LEN - wav.numel()))
    else:
        wav = wav[:TARGET_LEN]

    return wav


def collate_fn(batch):
    waves = []
    labels = []

    for waveform, sample_rate, label, *_ in batch:
        waves.append(prepare_waveform(waveform, sample_rate))
        labels.append(LABEL2ID[label])

    x = torch.stack(waves)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


class TinyKWS(nn.Module):
    def __init__(self, n_mels: int = 80, groups: int = 1):
        super().__init__()
        self.features = LogMelFilterBanks(samplerate=TARGET_SR, hop_length=160, n_mels=n_mels)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=groups),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.encoder(x)
        x = x.squeeze(-1)
        return self.classifier(x)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_conv_linear(model: nn.Module, device: torch.device) -> int:
    flops = 0
    hooks = []

    def hook_fn(module, _inp, out):
        nonlocal flops
        if isinstance(module, nn.Conv1d):
            batch, out_channels, out_len = out.shape
            kernel = module.kernel_size[0]
            in_per_group = module.in_channels // module.groups
            flops += batch * out_channels * out_len * in_per_group * kernel * 2
        elif isinstance(module, nn.Linear):
            batch = out.shape[0]
            flops += batch * module.in_features * module.out_features * 2

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, TARGET_LEN, device=device))

    for hook in hooks:
        hook.remove()

    return flops


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

    return correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    tag: str,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []
    epoch_times = []
    val_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()

        running_loss = 0.0
        samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = nnf.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            samples += y.size(0)

        epoch_time = time.perf_counter() - start
        epoch_loss = running_loss / max(samples, 1)
        val_acc = evaluate_accuracy(model, val_loader, device)

        epoch_losses.append(epoch_loss)
        epoch_times.append(epoch_time)
        val_accs.append(val_acc)

        print(
            f"{tag} | epoch={epoch:02d} | train_loss={epoch_loss:.4f} | "
            f"val_acc={val_acc:.4f} | time={epoch_time:.2f}s"
        )

    test_acc = evaluate_accuracy(model, test_loader, device)

    return {
        "params": count_trainable_params(model),
        "flops": estimate_flops_conv_linear(model, device),
        "avg_epoch_time_sec": sum(epoch_times) / len(epoch_times),
        "final_train_loss": epoch_losses[-1],
        "final_val_acc": val_accs[-1],
        "test_acc": test_acc,
        "epoch_train_losses": epoch_losses,
    }


def save_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_reference_check_and_plot(wav_path: Path, out_path: Path):
    import matplotlib.pyplot as plt

    signal, sr = torchaudio.load(str(wav_path))
    signal = prepare_waveform(signal, sr).unsqueeze(0)

    melspec = torchaudio.transforms.MelSpectrogram(hop_length=160, n_mels=80)(signal)
    logmelbanks = LogMelFilterBanks()(signal)

    ref = torch.log(melspec + 1e-6)
    assert ref.shape == logmelbanks.shape
    assert torch.allclose(ref, logmelbanks)

    ref_img = ref.squeeze(0).detach().cpu()
    own_img = logmelbanks.squeeze(0).detach().cpu()
    max_diff = (ref_img - own_img).abs().max().item()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(ref_img, origin="lower", aspect="auto")
    axes[0].set_title("torchaudio log-melspec")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("mel bin")

    axes[1].imshow(own_img, origin="lower", aspect="auto")
    axes[1].set_title("LogMelFilterBanks")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("mel bin")

    fig.suptitle(f"Max abs diff: {max_diff:.3e}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("Reference check passed")
    print(f"Saved: {out_path}")


def run_n_mels_experiment(
    n_mels_values,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_dir: Path,
):
    import matplotlib.pyplot as plt

    rows = []
    histories = {}

    for n_mels in n_mels_values:
        model = TinyKWS(n_mels=n_mels, groups=1).to(device)
        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            tag=f"n_mels={n_mels}",
        )
        histories[n_mels] = result["epoch_train_losses"]
        rows.append(
            {
                "n_mels": n_mels,
                "groups": 1,
                "params": result["params"],
                "flops": result["flops"],
                "avg_epoch_time_sec": result["avg_epoch_time_sec"],
                "final_train_loss": result["final_train_loss"],
                "final_val_acc": result["final_val_acc"],
                "test_acc": result["test_acc"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "n_mels_results.csv"
    save_csv(
        csv_path,
        [
            "n_mels",
            "groups",
            "params",
            "flops",
            "avg_epoch_time_sec",
            "final_train_loss",
            "final_val_acc",
            "test_acc",
        ],
        rows,
    )

    loss_plot = out_dir / "n_mels_train_loss.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    for n_mels in n_mels_values:
        ax.plot(range(1, epochs + 1), histories[n_mels], marker="o", label=f"n_mels={n_mels}")
    ax.set_title("Train loss for different n_mels")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_plot, dpi=150)
    plt.close(fig)

    testacc_plot = out_dir / "n_mels_test_accuracy.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["n_mels"] for r in rows], [r["test_acc"] for r in rows], marker="o")
    ax.set_title("Test accuracy vs n_mels")
    ax.set_xlabel("n_mels")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(testacc_plot, dpi=150)
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {loss_plot}")
    print(f"Saved: {testacc_plot}")


def run_groups_experiment(
    groups_values,
    baseline_n_mels: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_dir: Path,
):
    import matplotlib.pyplot as plt

    if any(64 % g != 0 for g in groups_values):
        raise ValueError("Each groups value must divide 64 for this model")

    rows = []

    for groups in groups_values:
        model = TinyKWS(n_mels=baseline_n_mels, groups=groups).to(device)
        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            tag=f"groups={groups}",
        )
        rows.append(
            {
                "n_mels": baseline_n_mels,
                "groups": groups,
                "params": result["params"],
                "flops": result["flops"],
                "avg_epoch_time_sec": result["avg_epoch_time_sec"],
                "final_train_loss": result["final_train_loss"],
                "final_val_acc": result["final_val_acc"],
                "test_acc": result["test_acc"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "groups_results.csv"
    save_csv(
        csv_path,
        [
            "n_mels",
            "groups",
            "params",
            "flops",
            "avg_epoch_time_sec",
            "final_train_loss",
            "final_val_acc",
            "test_acc",
        ],
        rows,
    )

    plot_path = out_dir / "groups_metrics.png"
    groups = [r["groups"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(groups, [r["avg_epoch_time_sec"] for r in rows], marker="o")
    axes[0, 0].set_title("Epoch time vs groups")
    axes[0, 0].set_xlabel("groups")
    axes[0, 0].set_ylabel("sec")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(groups, [r["params"] for r in rows], marker="o")
    axes[0, 1].set_title("Params vs groups")
    axes[0, 1].set_xlabel("groups")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(groups, [r["flops"] for r in rows], marker="o")
    axes[1, 0].set_title("FLOPs vs groups")
    axes[1, 0].set_xlabel("groups")
    axes[1, 0].set_ylabel("ops/sample")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(groups, [r["final_val_acc"] for r in rows], marker="o", label="val_acc")
    axes[1, 1].plot(groups, [r["test_acc"] for r in rows], marker="o", label="test_acc")
    axes[1, 1].set_title("Accuracy vs groups")
    axes[1, 1].set_xlabel("groups")
    axes[1, 1].set_ylabel("accuracy")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")


def run_all_experiments(
    data_root: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    n_mels_values,
    groups_values,
    baseline_n_mels: int,
    device: torch.device,
):
    data_root.mkdir(parents=True, exist_ok=True)

    train_ds = YesNoCommands(str(data_root), subset="training")
    val_ds = YesNoCommands(str(data_root), subset="validation")
    test_ds = YesNoCommands(str(data_root), subset="testing")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    run_n_mels_experiment(
        n_mels_values=n_mels_values,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        out_dir=output_dir,
    )

    run_groups_experiment(
        groups_values=groups_values,
        baseline_n_mels=baseline_n_mels,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        out_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Assignment 1 experiments")
    parser.add_argument("--mode", choices=["check", "experiments", "all"], default="all")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--wav-path", type=str, default="./file_for_test.wav")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-mels-values", type=str, default="20,40,80")
    parser.add_argument("--groups-values", type=str, default="2,4,8,16")
    parser.add_argument("--baseline-n-mels", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    set_seed(args.seed)
    validate_logmel_layer()
    print("LogMelFilterBanks validation passed")

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available. Use --device auto or --device cpu")
        device = torch.device("mps")
    else:
        device = torch.device(args.device)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    n_mels_values = parse_int_list(args.n_mels_values)
    groups_values = parse_int_list(args.groups_values)

    if args.mode in {"check", "all"}:
        run_reference_check_and_plot(Path(args.wav_path), output_dir / "logmel_reference_compare.png")

    if args.mode in {"experiments", "all"}:
        run_all_experiments(
            data_root=data_root,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_mels_values=n_mels_values,
            groups_values=groups_values,
            baseline_n_mels=args.baseline_n_mels,
            device=device,
        )


if __name__ == "__main__":
    main()
