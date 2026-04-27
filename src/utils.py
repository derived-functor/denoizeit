"""Some useful utils"""

from pathlib import Path

from torchcodec.decoders import AudioDecoder

from src.config import Config
from src.model.unet import UNet
from src.pipeline.pipeline import (
    DenoisingLongFilePipeline,
    DenoisingShortFilePipeline,
    Pipeline,
)


def get_pipeline(
    wav_path: str | Path, model: UNet, config: Config
) -> tuple[Pipeline, str]:
    """Selecting pipeline depends on duration of audio

    :return: pipeline itself and log string
    """
    decoder = AudioDecoder(wav_path)
    meta = decoder.metadata
    duration = meta.duration_seconds or 0.0

    if duration <= 0.0:
        raise ValueError("Duration of audio is <= 0.0")

    if duration > config.preprocessing.threshold:
        return (
            DenoisingLongFilePipeline(
                model, config.preprocessing, config.common.device
            ),
            "Long file pipeline is used",
        )

    return (
        DenoisingShortFilePipeline(model, config.preprocessing, config.common.device),
        "Short file pipeline is used",
    )
