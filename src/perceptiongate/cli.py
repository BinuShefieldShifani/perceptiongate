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
    train_images: Path = typer.Option(..., help="Train image folder (subfolders are searched)."),
    val_images: Path = typer.Option(..., help="Val image folder (subfolders are searched)."),
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


@data_app.command("prepare")
def data_prepare(
    config: Path = typer.Option(..., help="Experiment config YAML."),
    train_labels: Path = typer.Option(...),
    val_labels: Path = typer.Option(...),
    train_images: Path = typer.Option(..., help="Train image folder (subfolders are searched)."),
    val_images: Path = typer.Option(..., help="Val image folder (subfolders are searched)."),
    out_dir: Path = typer.Option(Path("prepared"), help="Folder for the .tar file."),
    workers: int = typer.Option(32, help="Parallel image readers."),
    quality: int = typer.Option(90, help="JPEG quality of resized images."),
) -> None:
    """Resize the manifest's images once and pack them into one tar (fast to load later)."""
    from perceptiongate.data.build import load_clean
    from perceptiongate.data.manifest import verify_manifest
    from perceptiongate.data.prepare import prepare_dataset

    cfg = load_config(config)
    problems = verify_manifest(cfg.data.manifest)
    if problems:
        typer.echo("Manifest is not intact: " + "; ".join(problems), err=True)
        raise typer.Exit(code=1)
    typer.echo("Loading labels...")
    frames_by_split = {
        split: {f.name: f for f in load_clean(cfg, labels, split)[0]}
        for split, labels in (("train", train_labels), ("val", val_labels))
    }
    tar = out_dir / f"{cfg.data.manifest.name}-{cfg.data.image_size}.tar"
    info = prepare_dataset(
        cfg.data.manifest,
        frames_by_split,
        {"train": train_images, "val": val_images},
        tar,
        cfg.data.classes,
        cfg.data.image_width,
        cfg.data.image_height,
        long_side=cfg.data.image_size,
        quality=quality,
        workers=workers,
        log=typer.echo,
    )
    typer.echo(f"Prepared {tar} ({info['tar_bytes'] / 1e6:.0f} MB) in {info['prepare_seconds']}s")


@data_app.command("unpack")
def data_unpack(
    tar: Path = typer.Option(..., help="Prepared .tar from `pgate data prepare`."),
    out: Path = typer.Option(Path("data/yolo"), help="Local folder to unpack into."),
) -> None:
    """Unpack a prepared dataset to local disk and print its dataset.yaml path."""
    from perceptiongate.data.prepare import unpack_dataset

    typer.echo(f"YOLO dataset ready: {unpack_dataset(tar, out)}")


@data_app.command("make-synthetic")
def data_make_synthetic(
    config: Path = typer.Option(..., help="Experiment config (classes and image size)."),
    out: Path = typer.Option(Path("data/synthetic")),
    n_train: int = typer.Option(16),
    n_val: int = typer.Option(8),
) -> None:
    """Write a tiny synthetic YOLO dataset for smoke tests."""
    from perceptiongate.data.synthetic import make_synthetic

    cfg = load_config(config)
    path = make_synthetic(out, cfg.data.classes, n_train, n_val, cfg.data.image_size, cfg.data.seed)
    typer.echo(f"Synthetic dataset ready: {path}")


@app.command("train")
def train_cmd(
    config: Path = typer.Option(..., help="Experiment config YAML."),
    dataset_yaml: Path = typer.Option(..., help="dataset.yaml from `pgate data materialize`."),
    out: Path = typer.Option(..., help="Run output folder."),
    epochs: int | None = typer.Option(None, help="Override epochs (recorded in run_info)."),
    skip_manifest_check: bool = typer.Option(
        False, help="Only for synthetic smoke runs. Recorded as data_verified=false."
    ),
) -> None:
    """Train, validate and log one run."""
    from perceptiongate.training.train import PreflightError, train

    cfg = load_config(config)
    try:
        result = train(
            cfg, dataset_yaml, out, epochs=epochs, check_manifest=not skip_manifest_check
        )
    except PreflightError as exc:
        typer.echo(f"PREFLIGHT FAILED: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    o = result.metrics["overall"]
    typer.echo(
        f"Done in {result.metrics['train_seconds']}s  mAP50-95 {o['map50_95']:.4f}  "
        f"mAP50 {o['map50']:.4f}  P {o['precision']:.3f}  R {o['recall']:.3f}"
    )
    typer.echo(f"Outputs in {out}")


if __name__ == "__main__":
    app()
