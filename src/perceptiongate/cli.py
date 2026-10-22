"""Command-line entry point: `pgate --help`."""

from __future__ import annotations

from pathlib import Path

import typer

from perceptiongate import __version__
from perceptiongate.config import load_config

app = typer.Typer(help="PerceptionGate pipeline commands.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("validate-config")
def validate_config(path: Path) -> None:
    """Validate an experiment config and print a short summary."""
    cfg = load_config(path)
    typer.echo(f"OK  {cfg.name}")
    typer.echo(f"    data:  {cfg.data.train_images} train / {cfg.data.val_images} val images")
    t = cfg.train
    typer.echo(f"    model: {t.weights}, {t.epochs} epochs, batch {t.batch}")


if __name__ == "__main__":
    app()
