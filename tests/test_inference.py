import pytest
import torch
from src.model.unet import UNet
from src.pipeline.inference import InferenceStage


class TestInferenceStage:
    @pytest.fixture
    def inference(self, model: UNet) -> InferenceStage:
        return InferenceStage(model)

    @pytest.fixture
    def noisy_in(self, device: str) -> torch.Tensor:
        return torch.load("tests/data/noisy_in.pth", map_location=device)

    def test_inference(self, inference: InferenceStage, noisy_in: torch.Tensor):
        mask = inference(noisy_in)

        assert mask.shape == noisy_in.shape
