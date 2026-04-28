import pytest
import torch

from src.model.unet import UNet
from src.pipeline.inference import inference


class TestInferenceStage:
    def test_inference(self, noisy_in: torch.Tensor, model: UNet):
        mask = inference(model, noisy_in)

        assert mask.shape == noisy_in.shape
