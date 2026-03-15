"""Preprocessing functions"""

from collections.abc import Generator

from pathlib import Path
from typing import Any
import torch
from torchcodec.decoders import AudioDecoder
import torchaudio


def transform_wav(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """Transforms wav to needed configuration"""
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


def load_wav(
    wav_path: str | Path,
    target_sr: int,
) -> torch.Tensor:
    """Loads wav from file and normalizes it"""
    wav, sr = torchaudio.load(wav_path)

    wav = transform_wav(wav, sr, target_sr)

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


def stream_audio_chunks(  # pylint: disable=R0914
    wav_path: str | Path,
    target_sr: int,
    chunk_size_frames: int,
    overlap_frames: int,
    hop_len: int,
) -> Generator[tuple[torch.Tensor, dict[str, Any]]]:
    """
    Streams chunks from audio file

    :yields: audio chunks and metadata that contains:
        - start_time - start time of chunk
        - start_sample - index of start sample of chunk
        - is_last
    """
    decoder = AudioDecoder(str(wav_path), sample_rate=target_sr, num_channels=1)

    frame_duration = hop_len / target_sr

    # in seconds
    chunk_duration = chunk_size_frames * frame_duration
    step_duration = (chunk_size_frames - overlap_frames) * frame_duration

    total_duration = decoder.metadata.duration_seconds
    current_time = 0.0

    while current_time < total_duration:
        stop_time = current_time + chunk_duration

        samples_obj = decoder.get_samples_played_in_range(
            start_seconds=current_time, stop_seconds=stop_time
        )

        chunk = samples_obj.data  # [1, Samples]

        # padding with zeros
        target_samples = chunk_size_frames * hop_len
        if chunk.shape[-1] < target_samples:
            pad_size = target_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))

        chunk = chunk[:, :target_samples]

        yield (
            chunk,
            {
                "start_time": current_time,
                "start_sample": int(round(current_time * target_sr)),
                "is_last": (current_time + step_duration) >= total_duration,
            },
        )

        current_time += step_duration
