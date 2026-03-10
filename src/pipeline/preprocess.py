"""Preprocessing functions"""

from pathlib import Path
import torch
import torchaudio


def load_wav(
    wav_path: str | Path,
    target_sr: int,
) -> torch.Tensor:
    """Loads wav from file and normalizes it"""
    wav, sr = torchaudio.load(wav_path)

    # Resample sample rate
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # Transform to mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Normalization to avoid clipping
    if (max_ := wav.abs().max()) > 1.0:
        # Adding 1e-8 to avoid zero division
        wav = wav / (max_ + 1e-8)

    return wav


def wav_to_spec(
    wav: torch.Tensor, n_fft: int, hop_len: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transforms wav to spectrogram"""

    # Adding batch dimension
    wav = wav.unsqueeze(0)

    win = torch.hann_window(n_fft).to(device)

    noisy_spec = torch.stft(
        wav.squeeze(1), n_fft=n_fft, hop_length=hop_len, window=win, return_complex=True
    )
    # Cutoff the 257-th bin
    noisy_in = noisy_spec.unsqueeze(1)[:, :, :256, :]

    return noisy_in, win


def preprocess(
    wav_path: str | Path,
    target_sr: int,
    n_fft: int,
    hop_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocessing function

    1. Resamples audio to target_sr
    2. Mixing the channels to 1 by meaning.
    3. Normalizes volume to 1.0 to avoid clipping.
    4. Performs STFT and cuts off 257-th bin of spectrogram.

    return noisy spectrogram and window
    """
    wav = load_wav(wav_path, target_sr).to(device)

    return wav_to_spec(wav, n_fft, hop_len, device)
