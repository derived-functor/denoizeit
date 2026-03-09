"""Main pipelines"""

from pathlib import Path

import torch

from src.model.unet import UNet
from src.config import config
from .preprocess import preprocess
from .inference import inference
from .postprocess import postprocess


class DenoisingShortFilePipeline:
    """Pipeline for denoising short audio files"""

    def __init__(self, model: UNet) -> None:
        self.model = model

    def __call__(self, wav_path: str | Path) -> torch.Tensor:
        noisy_in, win = preprocess(
            wav_path,
            config.preprocessing.target_sr,
            config.preprocessing.n_fft,
            config.preprocessing.hop_len,
            config.device,
        )

        mask = inference(self.model, noisy_in)

        denoised = postprocess(mask, noisy_in, win, config.device)

        return denoised
