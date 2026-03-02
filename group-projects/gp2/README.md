
## Group Project 2. Neural Vocoder Training - [30 pts]

In this exercise you will be training neural vocoder to synthesize utterance in english language. In particular, you are given a pre-trained single-voice TTS system [FastPitch](https://arxiv.org/pdf/2006.06873) that outputs mel spectrogram given an input text, and you need to train GAN-based architecture to convert spectrogram to a waveform.

![Mel spec](output_spectrogram.png)

### Task Description

- Train a neural vocoder with two losses: [Adversarial](https://en.wikipedia.org/wiki/Generative_adversarial_network) loss and [L1](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)/[MSE](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) loss for waveforms in time domain

- You need to write your own simple GAN training loop and generate 5 audio samples for utterances in [test_sentences.txt](./test_sentences.txt)

- To develop model's architecture you can refer to existing methods such as: [MelGAN](https://arxiv.org/abs/1910.06711), [MultiBandMelGAN](https://arxiv.org/abs/2005.05106), [ParallelWaveGAN](https://arxiv.org/abs/1910.11480), [HiFiGAN](https://arxiv.org/abs/2010.05646)

- Extra: experiment with **spectrogram normalization**. Also consider using alternative loss functions such as [Feature Matching and STFT](https://github.com/coqui-ai/TTS/blob/dev/TTS/vocoder/layers/losses.py) losses


### Inputs

- You are given a single-speaker `tts_models/en/ljspeech/fast_pitch` model from [Coqui-TTS](https://github.com/coqui-ai/TTS) trained on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset and a corresponding `TextToSpecConverter` class with method `text2spec(text: str)` in [t2spec_converter.py](./t2spec_converter.py) that converts input text into a mel spectrogram:
```python
# Example usage
t2s = TextToSpecConverter()
mel_spec = t2s.text2spec("Hello, world!")
# output is a numpy.ndarray with shape [C, T], where:
#   C = 80 (num mels from config)
#   T = ?? (num frames, variable, depends on the string length)
```

- As a vocoder training data, please use [LJSpeech](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.datasets.LJSPEECH.html#torchaudio.datasets.LJSPEECH) dataset

- For implementation simplicity, make sure you match `tts_config` audio parameters (such as **sample_rate**) when training the vocoder, e.g. `TextToSpecConverter().config['audio']`:
```
BaseAudioConfig(fft_size=1024,
                win_length=1024,
                hop_length=256,
                sample_rate=22050,
                log_func='np.log',
                power=1.5,
                num_mels=80,
                mel_fmin=0.0,
                mel_fmax=8000.0,
                spec_gain=1,
                signal_norm=False,
                ...)
```


### Evaluation

* Generate audio samples for the utterances in the [test_sentences.txt](./test_sentences.txt) file

* Your work will be assessed via manual listening as well as using non-intrusive method of estimation audio [MOS](https://en.wikipedia.org/wiki/Mean_opinion_score) with [DistillMOS](https://github.com/microsoft/Distill-MOS)


### Hints

* When dealing with variable-length inputs during training, use padding to the maximum length in batch and masking

* You can modify [line 35](./t2spec_converter.py#35) of `t2spec_converter.py` in order to run whole pipeline on GPU device during training


### Deliverables

* Public GitHub repository containing your:
    - training pipeline code
    - model architecture and weights
    - inference script example
    - 5 generated test audio samples

* Google Classroom PDF report describing your work, experiments and results in free form


### Resources

- [Coqui-TTS](https://github.com/coqui-ai/TTS): examples of architectures, training objectives and pipelines
- DistillMOS: [source paper](https://arxiv.org/pdf/2502.05356v1)
- DLA course vocoders materials: [slides link](https://docs.google.com/presentation/d/1ZZp_tNfZAu5QQW4Rk_8Tqp_cnbrjcMEjrk8NzJsAxDo/edit?slide=id.p#slide=id.p)
