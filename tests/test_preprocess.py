import pytest
from pathlib import Path
import torch
from src.config import Config
from src.pipeline.preprocess import (
    preprocess,
    load_wav,
    stream_audio_chunks,
    wav_to_spec,
)


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

    def test_stream_audio_chunks_properties(self, noisy_path: Path):
        target_sr = 16000
        chunk_size = 256
        overlap = 64
        hop_len = 160

        expected_step = (chunk_size - overlap) * hop_len
        expected_chunk_samples = chunk_size * hop_len  # 40960

        gen = stream_audio_chunks(
            noisy_path,
            target_sr=target_sr,
            chunk_size_frames=chunk_size,
            overlap_frames=overlap,
            hop_len=hop_len,
        )

        chunks = list(gen)

        assert len(chunks) > 0

        first_chunk, info = chunks[0]
        assert first_chunk.shape == (1, expected_chunk_samples)
        assert info["start_sample"] == 0

        if len(chunks) > 1:
            _, info_next = chunks[1]
            assert info_next["start_sample"] == expected_step

        _, last_info = chunks[-1]
        assert last_info["is_last"] is True
