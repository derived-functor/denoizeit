"""Main CLI logic"""

from pathlib import Path
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
import torch
import torchaudio
import typer

from src.config import get_config
from src.pipeline import DenoisingShortFilePipeline
from src.model.unet import UNet

app = typer.Typer(name="denoizeit!", help="Denoize audio files")


@app.command()
def process_short_file(
    wav_path: Path = typer.Argument(..., help="Path to noisy file", exists=True),
    output_file: Path = typer.Option(
        "denoized.wav", "--out", "-o", help="Name of output file"
    ),
    config_path: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Configuration file"
    ),
) -> None:
    """Command for denoising audio files"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Denoising", total=100)
        config = get_config(config_path)
        progress.update(task, completed=25, description="Loaded config")

        model = UNet()
        model.load_state_dict(
            torch.load(config.common.model_checkpoint, weights_only=True),
        )
        model = model.to(config.common.device)
        progress.update(task, completed=50, description="Loaded model")

        pipeline = DenoisingShortFilePipeline(model, config)

        progress.update(task, completed=75, description="Running pipeline")
        denoised = pipeline(wav_path).cpu()

        torchaudio.save(output_file, denoised, config.preprocessing.target_sr)
        progress.update(task, completed=100, description="Saving audio")

    typer.echo(f"Denoised audio saved to {output_file}")


if __name__ == "__main__":
    app()
