import pytest
from pathlib import Path
import torch
from src.config import Config
from src.pipeline.preprocess import preprocess, load_wav, wav_to_spec


class TestPreprocessStage:
    @pytest.fixture
    def wav(self, device: str) -> torch.Tensor:
        wav = torch.load("tests/data/wav.pth", device)

        return wav

    def test_load_wav(self, noisy_path: Path, config: Config):
        wav = load_wav(noisy_path, config.preprocessing.target_sr)

        assert wav.shape[0] == 1

    def test_wav_to_spec(self, wav: torch.Tensor, config: Config):
        noisy_in, win = wav_to_spec(
            wav,
            config.preprocessing.n_fft,
            config.preprocessing.hop_len,
            config.common.device,
        )

        assert noisy_in.dim() == 4
        assert noisy_in.shape[0] == 1
        assert noisy_in.shape[1] == 1
        assert noisy_in.shape[2] == 256

        assert win.shape[0] == config.preprocessing.n_fft
        assert noisy_in.is_complex()

        assert not torch.isnan(noisy_in).any()

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
