import pytest
import torch
from src.model import UNet
from pathlib import Path


@pytest.fixture
def device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)
    return device


@pytest.fixture
def noisy_path() -> Path:
    return Path("tests/data/noisy_audio.wav")


@pytest.fixture
def clean_path() -> Path:
    return Path("tests/data/clean_audio.wav")


@pytest.fixture
def model(device: str) -> UNet:
    model = UNet()
    model.load_state_dict(torch.load("data/best_unet.pth"))
    return model.to(device)
