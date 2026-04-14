"""This module provides local model factory"""

from pathlib import Path
from typing import Any

from src.model import UNet

from .abc import ModelFactory


class LocalModelFactory(ModelFactory):
    """Model factory for loading from local file"""

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> UNet:
        path = path if isinstance(path, Path) else Path(path)
        device = kwargs.get("device", "cpu")

        if not path.exists():
            raise ValueError(f"File {path} doen't exists")

        if cls._model:
            return cls._model

        return cls._load(path, device)
