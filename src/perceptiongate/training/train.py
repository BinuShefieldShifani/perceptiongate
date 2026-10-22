"""Train a YOLO detector from an experiment config.

Ultralytics does the training. This module wraps it with the things a team needs:
preflight checks so bad runs fail before using GPU time, fixed seeds, lineage, one
metrics file in a stable format, and tracking.
"""

from __future__ import annotations

import contextlib
import csv
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perceptiongate.config import ExperimentConfig
from perceptiongate.data.manifest import verify_manifest
from perceptiongate.data.yolo import SOURCE_MANIFEST, manifest_fingerprint
from perceptiongate.observability.run_info import collect_run_info
from perceptiongate.observability.tracking import Tracker, make_tracker

ModelFactory = Callable[[str], Any]


class PreflightError(RuntimeError):
    pass


@dataclass
class TrainResult:
    run_dir: Path
    weights: Path
    metrics: dict[str, Any]


def resolve_device(preference: str) -> str:
    if preference != "auto":
        return preference
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def preflight(cfg: ExperimentConfig, dataset_yaml: Path, check_manifest: bool) -> None:
    """Fail fast, before any training starts."""
    if not dataset_yaml.is_file():
        raise PreflightError(f"dataset.yaml not found: {dataset_yaml}")
    if not check_manifest:
        return
    problems = verify_manifest(cfg.data.manifest)
    if problems:
        raise PreflightError("manifest is not intact: " + "; ".join(problems))
    source = dataset_yaml.parent / SOURCE_MANIFEST
    if not source.is_file():
        raise PreflightError(f"{source} missing: dataset was not built by `pgate data materialize`")
    built_from = json.loads(source.read_text())["sha256"]
    if built_from != manifest_fingerprint(cfg.data.manifest):
        raise PreflightError(
            "dataset folder was built from a different manifest than the config points to; "
            "run `pgate data materialize` again"
        )


def read_history(results_csv: Path) -> list[dict[str, float]]:
    """Parse Ultralytics results.csv (one row per epoch) into clean dicts."""
    if not results_csv.is_file():
        return []
    rows: list[dict[str, float]] = []
    with results_csv.open() as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    row[key.strip()] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(row)
    return rows


def extract_metrics(val_results: Any, names: dict[int, str]) -> dict[str, Any]:
    box = val_results.box
    overall = {
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    per_class: dict[str, dict[str, float]] = {}
    for i, cls in enumerate(box.ap_class_index):
        per_class[names[int(cls)]] = {
            "map50_95": float(box.ap[i]),
            "map50": float(box.ap50[i]),
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
        }
    return {"overall": overall, "per_class": per_class}


def _default_factory(weights: str) -> Any:
    from ultralytics import YOLO, settings

    # We do our own tracking; stop Ultralytics from logging to MLflow/W&B on its own.
    for key in ("mlflow", "wandb", "clearml", "comet", "dvc", "neptune", "tensorboard"):
        with contextlib.suppress(KeyError, ValueError):
            settings.update({key: False})
    return YOLO(weights)


def train(
    cfg: ExperimentConfig,
    dataset_yaml: Path,
    out_dir: Path,
    *,
    epochs: int | None = None,
    check_manifest: bool = True,
    model_factory: ModelFactory | None = None,
    tracker: Tracker | None = None,
) -> TrainResult:
    preflight(cfg, dataset_yaml, check_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = tracker or make_tracker(out_dir)
    factory = model_factory or _default_factory
    t = cfg.train
    n_epochs = epochs or t.epochs
    device = resolve_device(t.device)

    info = collect_run_info(cfg, cfg.data.manifest if check_manifest else None)
    info.update({"epochs_run": n_epochs, "epochs_override": epochs is not None})
    info["data_verified"] = check_manifest
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")

    tracker.start(
        run_name=f"{cfg.name}-{info['git_commit'][:7]}",
        tags={
            "experiment": cfg.name,
            "git_commit": info["git_commit"],
            "config_hash": info["config_hash"],
            "data_verified": str(check_manifest).lower(),
            "gpu": info["gpu"],
        },
    )
    tracker.log_params({**cfg.model_dump(mode="json"), "epochs_run": n_epochs})
    status = "failed"
    try:
        start = time.monotonic()
        model = factory(t.weights)
        model.train(
            data=str(dataset_yaml),
            epochs=n_epochs,
            batch=t.batch,
            imgsz=cfg.data.image_size,
            lr0=t.lr0,
            seed=t.seed,
            deterministic=True,
            device=device,
            workers=t.workers,
            amp=t.amp,
            plots=t.plots,
            project=str(out_dir.resolve()),
            name="ultralytics",
            exist_ok=True,
            verbose=False,
        )
        train_dir = Path(model.trainer.save_dir)
        for epoch, row in enumerate(read_history(train_dir / "results.csv"), start=1):
            row.pop("epoch", None)
            tracker.log_metrics(row, step=epoch)

        val = model.val(
            data=str(dataset_yaml),
            imgsz=cfg.data.image_size,
            batch=t.batch,
            device=device,
            plots=False,
            project=str(out_dir.resolve()),
            name="val",
            exist_ok=True,
            verbose=False,
        )
        metrics = extract_metrics(val, dict(enumerate(cfg.data.classes)))
        metrics["train_seconds"] = round(time.monotonic() - start, 1)
        metrics["epochs"] = n_epochs
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        weights_dir = out_dir / "model"
        weights_dir.mkdir(exist_ok=True)
        best = weights_dir / "best.pt"
        shutil.copy2(train_dir / "weights" / "best.pt", best)

        tracker.log_metrics({f"final.{k}": v for k, v in metrics["overall"].items()})
        tracker.log_metrics(
            {f"class.{c}.map50": m["map50"] for c, m in metrics["per_class"].items()}
        )
        for path in (out_dir / "metrics.json", out_dir / "run_info.json", best):
            tracker.log_artifact(path)
        status = "finished"
        return TrainResult(run_dir=out_dir, weights=best, metrics=metrics)
    finally:
        tracker.end(status)
