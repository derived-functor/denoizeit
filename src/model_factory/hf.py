"""This module provides model factory for loading model weights from hugging face"""

from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from pydantic import BaseModel

from src.model import UNet

from .abc import ModelFactory


class HuggingFaceLoadParams(BaseModel):
    """Params for loading from huggingface hub"""

    repo_id: str
    filename: str
    cache_dir: str | Path


class HuggingFaceModelFactory(ModelFactory):
    """Model factory that loads weights from HF"""

    _reserved_kwargs: set[str] = {"filename", "cache_dir"}

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> UNet:
        if cls._model:
            return cls._model

        device = kwargs.get("device", "cpu")
        load_params = cls._parse_kwargs(kwargs)
        load_params.repo_id = path

        try:
            model_path: str = hf_hub_download(
                **load_params.model_dump(exclude_none=True)
            )
            model_path_ = Path(model_path)

            return cls._load(model_path_, device=device)

        except Exception as e:
            msg = "Error while loading weights"
            raise RuntimeError(msg) from e

    @classmethod
    def _parse_kwargs(cls, kwargs: dict[str, Any]) -> HuggingFaceLoadParams:
        data: dict[str, str | Path | bool] = {}
        for k, v in kwargs.items():
            if k in cls._reserved_kwargs and HuggingFaceLoadParams.model_fields:
                data[k] = v

        if len(data) < len(cls._reserved_kwargs):
            raise ValueError(
                f"Not found some kwargs. You should provide: {cls._reserved_kwargs}"
            )

        return HuggingFaceLoadParams(repo_id="", **data)
