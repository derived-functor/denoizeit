import pytest
import torch

from src.pipeline.postprocess import postprocess


class TestPostprocessStage:
    @pytest.fixture
    def mask(self, device: str) -> torch.Tensor:
        return torch.load("tests/data/mask.pth", device)

    @pytest.fixture
    def win(self, device: str) -> torch.Tensor:
        return torch.load("tests/data/win.pth", device)

    def test_postprocess(
        self, mask: torch.Tensor, noisy_in: torch.Tensor, win: torch.Tensor, device: str
    ):
        denoised_wav = postprocess(mask, noisy_in, win, device).cpu()

        assert not denoised_wav.is_complex()
        assert denoised_wav.dtype == torch.float32

        assert denoised_wav.dim() == 2
