from pathlib import Path

import pytest
import torch

from src.config import Config, get_config
from src.model import UNet


@pytest.fixture
def config() -> Config:
    return get_config("config.yaml")


@pytest.fixture
def device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)
    return device


@pytest.fixture
def noisy_path() -> Path:
    return Path("tests/data/noisy_audio.wav")


@pytest.fixture
def noisy_in(device: str) -> torch.Tensor:
    return torch.load("tests/data/noisy_in.pth", map_location=device)


@pytest.fixture
def clean_path() -> Path:
    return Path("tests/data/clean_audio.wav")


@pytest.fixture
def model(device: str) -> UNet:
    model = UNet()
    model.load_state_dict(torch.load("data/checkpoint.pth"))
    return model.to(device)


@pytest.fixture
def mock_model_weights() -> dict[str, torch.Tensor]:
    model = UNet()
    return model.state_dict()


@pytest.fixture
def temp_pth_file(tmp_path: Path, mock_model_weights: dict[str, torch.Tensor]) -> Path:
    p = tmp_path / "model.pth"
    torch.save(mock_model_weights, p)
    return p


@pytest.fixture
def temp_safetensors_file(
    tmp_path: Path, mock_model_weights: dict[str, torch.Tensor]
) -> Path:
    from safetensors.torch import save_file

    p = tmp_path / "model.safetensors"
    save_file(mock_model_weights, str(p))
    return p
