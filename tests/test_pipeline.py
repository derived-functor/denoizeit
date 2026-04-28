from pathlib import Path

import pytest
import torch
import torchaudio

from src.config import Config
from src.model.unet import UNet
from src.pipeline.pipeline import DenoisingLongFilePipeline, DenoisingShortFilePipeline


class TestDenoisingPipeline:
    @pytest.fixture
    def pipeline(
        self, model: UNet, config: Config, device: str
    ) -> DenoisingShortFilePipeline:
        return DenoisingShortFilePipeline(model, config.preprocessing, device)

    def test_pipeline(self, pipeline: DenoisingShortFilePipeline, noisy_path: Path):
        denoised = pipeline(noisy_path)

        assert denoised.dim() == 2


class TestDenoisingLongFilePipeline:
    @pytest.fixture
    def pipeline(
        self, model: UNet, config: Config, device: str
    ) -> DenoisingLongFilePipeline:
        return DenoisingLongFilePipeline(model, config.preprocessing, device)

    @pytest.fixture
    def tmp_path(self) -> Path:
        return Path("./tmp")

    def test_pipeline_output_validity(
        self, pipeline: DenoisingLongFilePipeline, noisy_path: str
    ):
        denoised_wav = pipeline(noisy_path)

        assert denoised_wav.dim() == 2
        assert denoised_wav.shape[0] == 1

        assert not torch.isnan(denoised_wav).any(), "Pipeline produced NaN values!"

        assert not torch.isinf(denoised_wav).any(), "Pipeline produced Inf values!"

        assert denoised_wav.dtype == torch.float32

    def test_pipeline_reconstruction_limit(
        self, pipeline: DenoisingLongFilePipeline, tmp_path: Path
    ):
        """
        Test on short file
        """
        short_path = tmp_path / "short.wav"
        torchaudio.save(short_path, torch.randn(1, 1000), 16000)

        denoised = pipeline(short_path)

        assert denoised.shape[-1] >= 1000
