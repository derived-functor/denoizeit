from pathlib import Path
import torch
import pytest
from src.config import config
from src.pipeline.preprocess import PreprocessStage


class TestPreprocessStage:
    @pytest.fixture
    def preprocess(self) -> PreprocessStage:
        return PreprocessStage(
            config.preprocessing.target_sr,
            config.preprocessing.n_fft,
            config.preprocessing.hop_len,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def test_dimensions(self, noisy_path: Path, preprocess: PreprocessStage):
        noisy_in, win = preprocess(noisy_path)

        assert noisy_in.dim() == 4
        assert noisy_in.shape[0] == 1
        assert noisy_in.shape[1] == 1
        assert noisy_in.shape[2] == 256

        assert win.shape[0] == preprocess.n_fft
        assert noisy_in.is_complex()

        assert not torch.isnan(noisy_in).any()
