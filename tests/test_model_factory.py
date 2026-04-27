from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.model.unet import UNet
from src.model_factory.abc import ModelFactory
from src.model_factory.hf import HuggingFaceModelFactory
from src.model_factory.local_fs import LocalModelFactory


class TestModelFactoryBase:
    def test_load_pth_success(
        self, temp_pth_file: Path, mock_model_weights: dict[str, torch.Tensor]
    ):
        model = ModelFactory._load_pth(temp_pth_file, "cpu")
        assert isinstance(model, UNet)
        assert model.state_dict().keys() == mock_model_weights.keys()

    def test_load_safetensors_success(
        self, temp_safetensors_file: Path, mock_model_weights: dict[str, torch.Tensor]
    ):
        model = ModelFactory._load_safetensors(temp_safetensors_file, "cpu")
        assert isinstance(model, UNet)
        assert model.state_dict().keys() == mock_model_weights.keys()

    def test_load_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Filetype of .* not known"):
            ModelFactory._load(Path("model.bin"), "cpu")

    def test_load_pth_invalid_file_raises(self, tmp_path: Path):
        invalid_file = tmp_path / "corrupted.pth"
        invalid_file.write_bytes(b"not a valid pth file")
        with pytest.raises(Exception):
            ModelFactory._load_pth(invalid_file, "cpu")


class TestLocalModelFactory:
    def test_load_existing_pth_file(self, temp_pth_file: Path):
        LocalModelFactory._model = None
        model = LocalModelFactory.load(temp_pth_file)
        assert isinstance(model, UNet)

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(ValueError, match="File .* doen't exists"):
            LocalModelFactory.load(Path("nonexistent/path/model.pth"))

    def test_load_caching(self, temp_pth_file: Path):
        LocalModelFactory._model = None
        model1 = LocalModelFactory.load(temp_pth_file)
        model2 = LocalModelFactory.load(temp_pth_file)
        assert model1 is model2

    def test_load_different_path_new_instance(
        self, temp_pth_file: Path, tmp_path: Path
    ):
        LocalModelFactory._model = None
        model1 = LocalModelFactory.load(temp_pth_file)
        LocalModelFactory._model = None
        weights2 = UNet().state_dict()
        temp_pth_file2 = tmp_path / "model2.pth"
        torch.save(weights2, temp_pth_file2)
        model2 = LocalModelFactory.load(temp_pth_file2)
        assert model1 is not model2

    def test_load_safetensors_routes_correctly(
        self, temp_safetensors_file: Path, mock_model_weights: dict[str, torch.Tensor]
    ):
        LocalModelFactory._model = None
        model = LocalModelFactory.load(temp_safetensors_file)
        assert isinstance(model, UNet)


class TestHuggingFaceModelFactory:
    @patch("src.model_factory.hf.hf_hub_download")
    def test_load_with_valid_kwargs(
        self, mock_download: MagicMock, temp_pth_file: Path
    ):
        mock_download.return_value = str(temp_pth_file)
        HuggingFaceModelFactory._model = None
        model = HuggingFaceModelFactory.load(
            "test/repo",
            filename="model.pth",
            cache_dir="/tmp/cache",
        )
        assert isinstance(model, UNet)
        mock_download.assert_called_once()

    @patch("src.model_factory.hf.hf_hub_download")
    def test_load_missing_filename_raises(self, mock_download: MagicMock):
        HuggingFaceModelFactory._model = None
        with pytest.raises(ValueError, match="Not found some kwargs"):
            HuggingFaceModelFactory.load("test/repo", cache_dir="/tmp/cache")

    @patch("src.model_factory.hf.hf_hub_download")
    def test_load_missing_cache_dir_raises(self, mock_download: MagicMock):
        HuggingFaceModelFactory._model = None
        with pytest.raises(ValueError, match="Not found some kwargs"):
            HuggingFaceModelFactory.load("test/repo", filename="model.pth")

    @patch("src.model_factory.hf.hf_hub_download")
    def test_load_calls_hf_hub_download_with_params(
        self, mock_download: MagicMock, temp_pth_file: Path
    ):
        mock_download.return_value = str(temp_pth_file)
        HuggingFaceModelFactory._model = None
        HuggingFaceModelFactory.load(
            "test/repo",
            filename="model.pth",
            cache_dir="/tmp/cache",
        )
        mock_download.assert_called_once_with(
            repo_id="test/repo",
            filename="model.pth",
            cache_dir="/tmp/cache",
        )

    @patch("src.model_factory.hf.hf_hub_download")
    def test_load_hub_error_wraps_exception(self, mock_download: MagicMock):
        mock_download.side_effect = Exception("Download failed")
        HuggingFaceModelFactory._model = None
        with pytest.raises(RuntimeError, match="Error while loading weights"):
            HuggingFaceModelFactory.load(
                "test/repo",
                filename="model.pth",
                cache_dir="/tmp/cache",
            )

    @patch("src.model_factory.hf.hf_hub_download")
    def test_load_caching_returns_same_instance(
        self, mock_download: MagicMock, temp_pth_file: Path
    ):
        mock_download.return_value = str(temp_pth_file)
        HuggingFaceModelFactory._model = None
        model1 = HuggingFaceModelFactory.load(
            "test/repo",
            filename="model.pth",
            cache_dir="/tmp/cache",
        )
        model2 = HuggingFaceModelFactory.load(
            "test/repo",
            filename="model.pth",
            cache_dir="/tmp/cache",
        )
        assert model1 is model2
        mock_download.assert_called_once()
