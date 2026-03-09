"""Main pipelines"""

from pathlib import Path

import torch

from src.config import Config
from src.model.unet import UNet
from .preprocess import preprocess
from .inference import inference
from .postprocess import postprocess


class DenoisingShortFilePipeline:
    """Pipeline for denoising short audio files"""

    def __init__(self, model: UNet, config: Config) -> None:
        self.model = model
        self.config = config

    def __call__(self, wav_path: str | Path) -> torch.Tensor:
        noisy_in, win = preprocess(
            wav_path,
            self.config.preprocessing.target_sr,
            self.config.preprocessing.n_fft,
            self.config.preprocessing.hop_len,
            self.config.common.device,
        )

        mask = inference(self.model, noisy_in)

        denoised = postprocess(mask, noisy_in, win, self.config.common.device)

        return denoised
