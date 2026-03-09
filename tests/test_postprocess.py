import pytest
import torchaudio
import torch

from src.pipeline.postprocess import PostprocessStage


class TestPostprocessStage:
    @pytest.fixture
    def mask(self, device: str) -> torch.Tensor:
        return torch.load("tests/data/mask.pth", device)

    @pytest.fixture
    def win(self, device: str) -> torch.Tensor:
        return torch.load("tests/data/win.pth", device)

    @pytest.fixture
    def postprocess(
        self, noisy_in: torch.Tensor, win: torch.Tensor, device: str
    ) -> PostprocessStage:
        return PostprocessStage(noisy_in, win, device)

    def test_postprocess(
        self, postprocess: PostprocessStage, mask: torch.Tensor, noisy_in: torch.Tensor
    ):
        denoised_wav = postprocess(mask).cpu()

        assert not denoised_wav.is_complex()
        assert denoised_wav.dtype == torch.float32

        assert denoised_wav.dim() == 2
