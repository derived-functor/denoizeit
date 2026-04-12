from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from typer.testing import CliRunner

from src.cli.cli import app
from src.config import (
    Common,
    Config,
    HuggingFaceCheckpoint,
    ModelCheckpoint,
    PreprocessingConfig,
)

runner = CliRunner()


@patch("src.cli.cli.get_config")
@patch("src.cli.cli.HuggingFaceModelFactory")
@patch("src.cli.cli.get_pipeline")
@patch("src.cli.cli.torchaudio.save")
class TestCliWithHuggingFace:
    def test_process_file(
        self,
        mock_save: MagicMock,
        mock_get_pipeline: MagicMock,
        mock_factory: MagicMock,
        mock_get_config: MagicMock,
        tmp_path: Path,
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: data")

        noisy_audio = tmp_path / "noisy.wav"
        noisy_audio.write_text("fake audio")

        real_config = Config(
            preprocessing=PreprocessingConfig(
                target_sr=16000,
                n_fft=512,
                hop_len=160,
                threshold=20.0,
            ),
            common=Common(device="cpu"),
            model_checkpoint=ModelCheckpoint(
                hf=HuggingFaceCheckpoint(
                    repo_id="derived-functor/denoizeit-unet", filename="model.pth"
                ),
                local=None,
            ),
            cache_dir="/tmp/hf_cache",
        )
        mock_get_config.return_value = real_config

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_factory.load.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        fake_tensor = torch.zeros(1, 16000)
        mock_pipeline_instance.return_value = fake_tensor

        mock_get_pipeline.return_value = (mock_pipeline_instance, "Mock log message")

        result = runner.invoke(
            app,
            [
                "process",
                str(noisy_audio),
                "--out",
                "test_out.wav",
                "--config",
                str(config_file),
            ],
        )

        assert result.exit_code == 0
        assert "Denoised audio saved to test_out.wav" in result.stdout

        mock_factory.load.assert_called_once_with(
            "derived-functor/denoizeit-unet",
            filename="model.pth",
            cache_dir="/tmp/hf_cache",
        )

        mock_get_pipeline.assert_called_once()

        mock_pipeline_instance.assert_called_once_with(Path(noisy_audio))

        mock_save.assert_called_once()


@patch("src.cli.cli.get_config")
@patch("src.cli.cli.LocalModelFactory")
@patch("src.cli.cli.get_pipeline")
@patch("src.cli.cli.torchaudio.save")
class TestCliWithLocalModel:
    def test_process_file(
        self,
        mock_save: MagicMock,
        mock_get_pipeline: MagicMock,
        mock_factory: MagicMock,
        mock_get_config: MagicMock,
        tmp_path: Path,
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: data")

        noisy_audio = tmp_path / "noisy.wav"
        noisy_audio.write_text("fake audio")

        real_config = Config(
            preprocessing=PreprocessingConfig(
                target_sr=16000,
                n_fft=512,
                hop_len=160,
                threshold=20.0,
            ),
            common=Common(device="cpu"),
            model_checkpoint=ModelCheckpoint(
                hf=None,
                local="data/checkpoint.pth",
            ),
        )
        mock_get_config.return_value = real_config

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_factory.load.return_value = mock_model

        mock_pipeline_instance = MagicMock()
        fake_tensor = torch.zeros(1, 16000)
        mock_pipeline_instance.return_value = fake_tensor

        mock_get_pipeline.return_value = (mock_pipeline_instance, "Mock log message")

        result = runner.invoke(
            app,
            [
                "process",
                str(noisy_audio),
                "--out",
                "test_out.wav",
                "--config",
                str(config_file),
            ],
        )

        assert result.exit_code == 0
        assert "Denoised audio saved to test_out.wav" in result.stdout

        mock_factory.load.assert_called_once_with("data/checkpoint.pth")

        mock_get_pipeline.assert_called_once()

        mock_pipeline_instance.assert_called_once_with(Path(noisy_audio))

        mock_save.assert_called_once()
