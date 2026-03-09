import pytest
from pathlib import Path


@pytest.fixture
def noisy_path() -> Path:
    return Path("tests/data/noisy_audio.wav")


@pytest.fixture
def clean_path() -> Path:
    return Path("tests/data/clean_audio.wav")
