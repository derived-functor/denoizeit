"""Inference stage"""

import torch
from src.model.unet import UNet


@torch.no_grad()
def inference(model: UNet, noisy_spectrogram: torch.Tensor) -> torch.Tensor:
    """Inference U-Net model"""
    result = model(noisy_spectrogram)

    return result
