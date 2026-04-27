"""Main pipelines"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from src.config import PreprocessingConfig
from src.model.unet import UNet

from .inference import inference
from .postprocess import postprocess
from .preprocess import preprocess, stream_audio_chunks, wav_to_spec


class Pipeline(ABC):
    """Abstract class for pipelines"""

    def __init__(self, model: UNet, config: PreprocessingConfig, device: str) -> None:
        self.model = model
        self.config = config
        self.device = device

    @abstractmethod
    def __call__(self, wav_path: str | Path) -> torch.Tensor:
        """Calling the pipeline"""


class DenoisingShortFilePipeline(Pipeline):
    """Pipeline for denoising short audio files"""

    def __call__(self, wav_path: str | Path) -> torch.Tensor:
        noisy_in, win = preprocess(
            wav_path,
            self.config.target_sr,
            self.config.n_fft,
            self.config.hop_len,
            self.device,
        )

        mask = inference(self.model, noisy_in)

        denoised = postprocess(mask, noisy_in, win, self.device)

        return denoised


class DenoisingLongFilePipeline(Pipeline):
    """Pipeline for denoising long audio files"""

    def __init__(self, model: UNet, config: PreprocessingConfig, device: str):
        super().__init__(model, config, device)

        self.hop_len = config.hop_len
        self.n_fft = config.n_fft

        self.chunk_size = 256
        self.overlap = 64
        self.L_samples = self.chunk_size * self.hop_len  # pylint: disable=C0103

        self.ola_window = self._create_ola_window()

    def _create_ola_window(self) -> torch.Tensor:
        """Creates overlap-add window"""
        window = torch.ones(self.L_samples, device=self.device)
        overlap_samples = self.overlap * self.hop_len
        if overlap_samples > 0:
            ramp = torch.linspace(0, 1, overlap_samples, device=self.device)
            window[:overlap_samples] = ramp
            window[-overlap_samples:] = ramp.flip(0)
        return window

    @torch.no_grad()
    def __call__(self, wav_path: Path | str) -> torch.Tensor:  # pylint: disable=R0914
        chunk_gen = stream_audio_chunks(
            wav_path,
            target_sr=self.config.target_sr,
            chunk_size_frames=self.chunk_size,
            overlap_frames=self.overlap,
            hop_len=self.hop_len,
        )

        # buffers for accumulation
        # reserves 10 minutes ahead
        est_size = self.config.target_sr * 600
        output_audio = torch.zeros(1, est_size, device=self.device)
        weight_sum = torch.zeros(1, est_size, device=self.device)

        last_sample_idx = 0

        for chunk, info in chunk_gen:
            chunk = chunk.to(self.device)
            start_s = info["start_sample"]
            end_s = start_s + self.L_samples

            # buffer extension
            if end_s > output_audio.shape[-1]:
                extension = torch.zeros(1, self.L_samples * 100, device=self.device)
                output_audio = torch.cat([output_audio, extension], dim=-1)
                weight_sum = torch.cat([weight_sum, extension], dim=-1)

            spec_in, win_stft = wav_to_spec(
                chunk, self.n_fft, self.hop_len, self.device
            )

            mask = inference(self.model, spec_in)

            denoised_wav = postprocess(mask, spec_in, win_stft, self.device)

            output_audio[:, start_s:end_s] += denoised_wav * self.ola_window
            weight_sum[:, start_s:end_s] += self.ola_window

            last_sample_idx = end_s

        final_wav = output_audio / (weight_sum + 1e-8)

        return final_wav[:, :last_sample_idx]
