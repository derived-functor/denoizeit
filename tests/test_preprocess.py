from pathlib import Path
import torch
from src.config import Config
from src.pipeline.preprocess import preprocess


class TestPreprocessStage:
    def test_dimensions(self, noisy_path: Path, config: Config):
        noisy_in, win = preprocess(
            noisy_path,
            config.preprocessing.target_sr,
            config.preprocessing.n_fft,
            config.preprocessing.hop_len,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        assert noisy_in.dim() == 4
        assert noisy_in.shape[0] == 1
        assert noisy_in.shape[1] == 1
        assert noisy_in.shape[2] == 256

        assert win.shape[0] == config.preprocessing.n_fft
        assert noisy_in.is_complex()

        assert not torch.isnan(noisy_in).any()
