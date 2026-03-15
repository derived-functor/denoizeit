# denoizeit!

```
         _                      _             _  _    _
        | |                    (_)           (_)| |  | |
      __| |  ___  _ __    ___   _  ____  ___  _ | |_ | |
     / _` | / _ \| '_ \  / _ \ | ||_  / / _ \| || __|| |
    | (_| ||  __/| | | || (_) || | / / |  __/| || |_ |_|
     \__,_| \___||_| |_| \___/ |_|/___| \___||_| \__|(_)

```

Audio denoising tool using a UNet model with PyTorch.

## Usage

```bash
denoizeit process path/to/noisy.wav -o denoised.wav
```

## Options

- `wav_path` - Path to noisy audio file (required)
- `-o, --out` - Output file name (default: denoized.wav)
- `-c, --config` - Configuration file (default: config.yaml)

## Configuration

Configuration should be presented in yaml file of that structure:

```yaml
preprocessing:
  threshold: 10
  target_sr: 16000
  n_fft: 512
  hop_len: 160
common:
  device: cpu
  model_checkpoint: data/checkpoint.pth
```

> [!WARNING]
> Don't change preprocessing settings if you're not using custom model checkpoint.
> Default model was trained on that specific settings.
> The only setting is here to change is `threshold`. It defines duration of audio
> that will be processed at once. For example, if `threshold` is 10, then all audio
> files that shorter than 10 sec will be loaded to model at once, other files
> will use batching
