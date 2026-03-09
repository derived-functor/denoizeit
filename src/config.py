"""Configuration of project"""

from pathlib import Path
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class PreprocessingConfig(BaseModel):
    """Config for preprocessing"""

    target_sr: int
    n_fft: int
    hop_len: int


class Common(BaseModel):
    """Common options"""

    device: str
    model_checkpoint: str


class Config(BaseSettings):
    """Main config"""

    preprocessing: PreprocessingConfig
    common: Common


def get_config(path: str | Path) -> Config:
    """Reads config file"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)
