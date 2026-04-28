"""Main CLI logic"""

from pathlib import Path

import torchaudio
import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from src.config import DEFAULT_CONFIG_PATH, get_config
from src.model_factory.hf import HuggingFaceModelFactory
from src.model_factory.local_fs import LocalModelFactory
from src.utils import get_pipeline

LOGO = r"""
         _                      _             _  _    _ 
        | |                    (_)           (_)| |  | |
      __| |  ___  _ __    ___   _  ____  ___  _ | |_ | |
     / _` | / _ \| '_ \  / _ \ | ||_  / / _ \| || __|| |
    | (_| ||  __/| | | || (_) || | / / |  __/| || |_ |_|
     \__,_| \___||_| |_| \___/ |_|/___| \___||_| \__|(_)

    """

app = typer.Typer(name="denoizeit!", help="Denoize audio files")


@app.callback()
def main():
    """Callback function before commands"""
    typer.secho(LOGO, fg=typer.colors.CYAN, bold=True)


@app.command()
def process(
    wav_path: Path = typer.Argument(..., help="Path to noisy file", exists=True),
    output_file: Path = typer.Option(
        "denoized.wav", "--out", "-o", help="Name of output file"
    ),
    config_path: Path = typer.Option(
        f"{DEFAULT_CONFIG_PATH}/config.yaml",
        "--config",
        "-c",
        help="Configuration file",
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

        if hf := config.model_checkpoint.hf:
            progress.update(task, completed=25, description="Loading config (HF)")
            model = HuggingFaceModelFactory.load(
                hf.repo_id,
                filename=hf.filename,
                cache_dir=config.cache_dir,
                device=config.common.device,
            )
        elif local := config.model_checkpoint.local:
            progress.update(task, completed=25, description="Loading config (Local)")
            model = LocalModelFactory.load(local, device=config.common.device)
        else:
            raise ValueError("No proper model checkpoint provided")

        model = model.to(config.common.device)
        progress.update(task, completed=50, description="Loaded model")

        pipeline, _ = get_pipeline(wav_path, model, config)

        progress.update(
            task,
            completed=75,
            description=f"Running pipeline ({pipeline.__class__.__name__})",
        )
        denoised = pipeline(wav_path).cpu()

        torchaudio.save(output_file, denoised, config.preprocessing.target_sr)
        progress.update(task, completed=100, description="Saving audio")

    typer.echo(f"Denoised audio saved to {output_file}")


if __name__ == "__main__":
    app()
