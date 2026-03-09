"""Inference stage"""

import torch
from src.model.unet import UNet
from src.pipeline.abc import Stage


@torch.no_grad()
def _inference(model: UNet, noisy_spectrogram: torch.Tensor) -> torch.Tensor:
    """Inference U-Net model"""
    result = model(noisy_spectrogram)

    return result


class InferenceStage(Stage[torch.Tensor, torch.Tensor]):
    """Stage of inferencing"""

    def __init__(self, model: UNet) -> None:
        self.model = model
        self.model.eval()

        super().__init__(self._run)

    def _run(self, noisy_spectrogram: torch.Tensor) -> torch.Tensor:
        return _inference(self.model, noisy_spectrogram)
