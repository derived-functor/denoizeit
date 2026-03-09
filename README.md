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
denoizeit path/to/noisy.wav -o denoised.wav
```

## Options

- `wav_path` - Path to noisy audio file (required)
- `-o, --out` - Output file name (default: denoized.wav)
- `-c, --config` - Configuration file (default: config.yaml)

## Configuration

Configuration should be presented in yaml file of that structure:

```yaml
preprocessing:
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
