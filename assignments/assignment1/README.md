## Assignment 1. Digital Signal Processing - [20 pts]

Implement a PyTorch layer (inherit a class from `torch.nn.Module`) for extraction of **logarithms of Mel-scale Filterbank energies** using basic `torch` math operations such as multiplication, power and so on. **[10 pts]**

1. Given a template [melbanks.py](melbanks.py), complete the implementation of class `LogMelFilterBanks`:
    - You're required to finish `LogMelFilterBanks.__init__(self, **kwargs)` and `.forward(self, x)` methods
        - Please note that there are two more methods to implement such as `.spectrogram(self, x)` and `._init_melscale_fbanks(self)`. See code comments with details
        - Your code should replace `<YOUR CODE GOES HERE>` placeholders with your code
    - Implementation notes:
        - In this exercise we use [**hann** window](https://pytorch.org/docs/stable/generated/torch.hann_window.html) only
        - Also, power spectrum of spectrogram has to be used
        - No `torchaudio` functions calls are allowed in this layer implementation other than already provided (i.e. `F.melscale_fbanks()` and `torchaudio.load()` are allowed)
    - Evaluation:
        - For an arbitrary audiofile of 16 kHz sampling frequency, plot the Log MelFilterbanks output versus native `torchaudio.transforms.MelSpectrogram` implementation and attach plots to the report
        - Don't forget that natural `torchaudio.transforms.MelSpectrogram` doesn't apply logarithm
        - Audio file can be loaded using the snippet below
        ```python3
        import torchaudio
        signal, sr = torchaudio.load(<wav_path>)
        ```
        - Your implementation will be checked via the following snippet:
        ```python3
        melspec = torchaudio.transforms.MelSpectrogram(
            hop_length=160,
            n_mels=80
        )(signal)
        logmelbanks = LogMelFilterBanks()(signal)

        assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
        assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)
        ```
    - Hints:
        - Be careful with `Optional` parameters initialization
        - Use [resources below](#resources) and lecture slides
        - Such simple operators as `torch.abs()`, `torch.log()` and so on should be sufficient to complete this exercise


---

Train a simple CNN model (no more than 100K parameters) with `LogMelFilterBanks` features for a binary classification problem with [**PyTorch**](https://pytorch.org/) on [Google Speech Commands](https://arxiv.org/abs/1804.03209) data. **[10 pts]**


2. Set-up training pipeline:
    - Use `from torchaudio.datasets import SPEECHCOMMANDS` dataset
    - Convert multi-classification problem into a **binary classification** problem by utilization of two `**"YES"**` and `**"NO"**` target classes only
    - Use provided by default training/validation/testing splits of data
    - Define a simple custom model architecture (based on [`torch.nn.Conv1d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)) **up to ~100K parameters**
    - You can use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) in this exercise
    - Track (log) **train loss**, **validation accuracy** and **epoch training time** (no matter what batch size is used)
    - Implement model testing on the testing subset with [accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) as a metric
    - Make sure you implement methods that can calculate **number of model parameters** and **FLOPs** (for [FLOPs](https://en.wikipedia.org/wiki/Floating_point_operations_per_second) calculation external library can be imported)


3. Train a model and experiment with a number of filterbanks in `LogMelFilterBanks` layer:
    - Try varying this number (e.g. `n_mels = X  # X from [20, 40, 80]`)
    - Dump to report comparisons and rest of your conclusions:
        - e.g. you can plot **train loss** of different model runs in one graph
        - you can plot **n_mels** vs **testing accuracy** and so on
    - Choose any of trained models as a baseline for the next stage task


4. Experiment with **`groups`** parameter of Conv1d layer:
    ![Group convolution](group_convolution.png)
    - Set this parameter to different values (e.g. one of `{2, 4, 8, 16}`) and train the model (while tracking all the metrics)
    - Attach a graph with dependency of **epoch training time** versus **groups** parameter, as well as **number of model parameters** and **FLOPs** versus **groups** and other
    - Dump to report all your graphs and conclusions


---

4. Submit assignment via Google Classroom as a link to a public GitHub repository with your code and a PDF report summarizing all your conclusions and plots

---

### Resources:
- [PyTorch audio features tutorial](https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html#mel-filter-bank)
- [Mel Spectrogram](https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html#melspectrogram)
- [Mel Scale](https://pytorch.org/audio/main/_modules/torchaudio/functional/functional.html#melscale_fbanks)
- [Group convolutions explained](https://www.youtube.com/watch?v=vVaRhZXovbw)
- [PyTroch Speechcommands tutorial](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html)
- [Image source](https://github.com/Daniil-Osokin/fully-learnable-group-convolution.pytorch)
