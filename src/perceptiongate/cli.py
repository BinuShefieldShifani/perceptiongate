"""Command-line entry point: `pgate --help`."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from perceptiongate import __version__
from perceptiongate.config import load_config

app = typer.Typer(help="PerceptionGate pipeline commands.", no_args_is_help=True)
data_app = typer.Typer(help="Dataset commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("validate-config")
def validate_config(path: Path) -> None:
    """Validate an experiment config and print a short summary."""
    cfg = load_config(path)
    s = cfg.data.sampling
    t = cfg.train
    typer.echo(f"OK  {cfg.name}")
    typer.echo(f"    data:  {s.train_images} train / {s.val_images} val images")
    typer.echo(f"    model: {t.weights}, {t.epochs} epochs, batch {t.batch}")


@data_app.command("inspect")
def data_inspect(labels: Path) -> None:
    """Summarise a raw BDD100K label file (categories, weather, time of day)."""
    from perceptiongate.data.inspect import summarize
    from perceptiongate.data.schema import load_frames

    typer.echo(json.dumps(summarize(load_frames(labels)), indent=2))


@data_app.command("build-manifest")
def data_build_manifest(
    config: Path = typer.Option(..., help="Experiment config YAML."),
    train_labels: Path = typer.Option(..., help="BDD100K train label JSON."),
    val_labels: Path = typer.Option(..., help="BDD100K val label JSON."),
    out: Path | None = typer.Option(None, help="Output dir. Defaults to config data.manifest."),
) -> None:
    """Clean, validate and stratify-sample the labels into a versioned manifest."""
    from perceptiongate.data.build import DataValidationError, build_manifest

    cfg = load_config(config)
    try:
        summary = build_manifest(cfg, train_labels, val_labels, out)
    except DataValidationError as exc:
        typer.echo(f"FAILED data validation: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for split, info in summary["splits"].items():
        typer.echo(
            f"{split}: {info['images']} images, {info['boxes']} boxes, "
            f"{info['small_boxes']} small, {info['vru_boxes']} VRU"
        )
    typer.echo(f"Manifest written to {out or cfg.data.manifest}")


@data_app.command("verify-manifest")
def data_verify_manifest(manifest_dir: Path) -> None:
    """Check a committed manifest: hashes, row counts, no image in two splits."""
    from perceptiongate.data.manifest import verify_manifest

    problems = verify_manifest(manifest_dir)
    if problems:
        for p in problems:
            typer.echo(f"  - {p}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK  {manifest_dir}")


@data_app.command("materialize")
def data_materialize(
    config: Path = typer.Option(..., help="Experiment config YAML."),
    train_labels: Path = typer.Option(...),
    val_labels: Path = typer.Option(...),
    train_images: Path = typer.Option(..., help="Folder with BDD100K train images."),
    val_images: Path = typer.Option(..., help="Folder with BDD100K val images."),
    out: Path = typer.Option(Path("data/yolo"), help="Where to write the YOLO dataset."),
) -> None:
    """Turn the manifest into a YOLO dataset (symlinked images + label txt files)."""
    from perceptiongate.data.build import load_clean
    from perceptiongate.data.manifest import verify_manifest
    from perceptiongate.data.yolo import materialize

    cfg = load_config(config)
    problems = verify_manifest(cfg.data.manifest)
    if problems:
        typer.echo("Manifest is not intact: " + "; ".join(problems), err=True)
        raise typer.Exit(code=1)
    frames_by_split = {
        split: {f.name: f for f in load_clean(cfg, labels, split)[0]}
        for split, labels in (("train", train_labels), ("val", val_labels))
    }
    dataset_yaml = materialize(
        cfg.data.manifest,
        frames_by_split,
        {"train": train_images, "val": val_images},
        out,
        cfg.data.classes,
        cfg.data.image_width,
        cfg.data.image_height,
    )
    typer.echo(f"YOLO dataset ready: {dataset_yaml}")


if __name__ == "__main__":
    app()
