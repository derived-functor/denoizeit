"""This model provides interface model factories"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from src.model.unet import UNet


class ModelFactory(ABC):
    """Interface for model factories"""

    _model: UNet | None = None

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, **kwargs: Any) -> UNet:
        """Main method for loading model"""

    @classmethod
    def _load_pth(cls, path: str | Path, device: str) -> UNet:
        """Loads model's weights from .pth file"""
        weights = torch.load(path, weights_only=True, map_location=device)

        model = UNet()
        model.load_state_dict(weights)

        cls._model = model

        return model

    @classmethod
    def _load_safetensors(cls, path: str | Path, device: str) -> UNet:
        """Loads model weights from safetensors type of file"""
        raise NotImplementedError("safetensors library is not supported yet")

    @classmethod
    def _load(cls, path: Path, device: str = "cpu") -> UNet:
        if "pth" in path.suffix:
            return cls._load_pth(path, device)

        if "safetensors" in path.suffix:  # pylint: disable=R1705
            return cls._load_safetensors(path, device)

        else:
            raise ValueError(
                f"Filetype of {path} not known\nKnown types: .pth, .safetensors"
            )
