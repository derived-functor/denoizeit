import pytest
import torch
from src.model.unet import UNet
from src.pipeline.inference import InferenceStage


class TestInferenceStage:
    @pytest.fixture
    def inference(self, model: UNet) -> InferenceStage:
        return InferenceStage(model)

    def test_inference(self, inference: InferenceStage, noisy_in: torch.Tensor):
        mask = inference(noisy_in)

        assert mask.shape == noisy_in.shape
