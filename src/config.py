"""Configuration of project"""

from pathlib import Path

import yaml
from platformdirs import user_cache_dir, user_config_dir
from pydantic import BaseModel
from pydantic_settings import BaseSettings

DEFAULT_CONFIG_PATH = user_config_dir("denoizeit").rstrip("/")


class PreprocessingConfig(BaseModel):
    """Config for preprocessing"""

    threshold: float
    target_sr: int
    n_fft: int
    hop_len: int


class Common(BaseModel):
    """Common options"""

    device: str


class HuggingFaceCheckpoint(BaseModel):
    """Checkpoint config for huggingface load"""

    repo_id: str
    filename: str


class ModelCheckpoint(BaseModel):
    """Config for model checkpoint"""

    hf: HuggingFaceCheckpoint | None = None
    local: str | None


class Config(BaseSettings):
    """Main config"""

    appname: str = "denoizeit"
    cache_dir: str = user_cache_dir(appname)

    preprocessing: PreprocessingConfig
    common: Common
    model_checkpoint: ModelCheckpoint


def get_config(path: str | Path) -> Config:
    """Reads config file"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)
