from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F


class LogMelFilterBanks(nn.Module):
    def __init__(
        self,
        n_fft: int = 400,
        samplerate: int = 16000,
        hop_length: int = 160,
        n_mels: int = 80,
        pad_mode: str = "reflect",
        power: float = 2.0,
        normalize_stft: bool = False,
        onesided: bool = True,
        center: bool = True,
        return_complex: bool = True,
        f_min_hz: float = 0.0,
        f_max_hz: Optional[float] = None,
        norm_mel: Optional[str] = None,
        mel_scale: str = "htk",
    ):
        super(LogMelFilterBanks, self).__init__()
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.register_buffer("window", torch.hann_window(self.window_length), persistent=False)

        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power
        self.n_freqs = self.n_fft // 2 + 1 if self.onesided else self.n_fft

        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz if f_max_hz is not None else samplerate / 2
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale

        self.register_buffer("mel_fbanks", self._init_melscale_fbanks(), persistent=False)

    def _init_melscale_fbanks(self):
        return F.melscale_fbanks(
            n_freqs=self.n_freqs,
            f_min=self.f_min_hz,
            f_max=self.f_max_hz,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale,
        )

    def spectrogram(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window.to(device=x.device, dtype=x.dtype),
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalize_stft,
            onesided=self.onesided,
            return_complex=self.return_complex,
        )

    def forward(self, x):
        spec = self.spectrogram(x)
        if spec.is_complex():
            spec = spec.abs().pow(self.power)
        else:
            spec = spec.pow(2).sum(dim=-1).pow(self.power / 2.0)

        mel_fbanks = self.mel_fbanks.to(device=spec.device, dtype=spec.dtype)
        mel_spec = torch.matmul(mel_fbanks.transpose(0, 1), spec)

        return torch.log(mel_spec + 1e-6)
