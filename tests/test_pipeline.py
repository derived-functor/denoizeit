import pytest
from src.model.unet import UNet
from src.pipeline.pipeline import DenoisingShortFilePipeline
from pathlib import Path


class TestDenoisingPipeline:
    @pytest.fixture
    def pipeline(self, model: UNet) -> DenoisingShortFilePipeline:
        return DenoisingShortFilePipeline(model)

    def test_pipeline(self, pipeline: DenoisingShortFilePipeline, noisy_path: Path):
        denoised = pipeline(noisy_path)

        assert denoised.dim() == 2
