"""Evaluate a model: predict (GPU) -> match -> score per slice (CPU) -> report files."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from perceptiongate.config import ExperimentConfig
from perceptiongate.data.manifest import read_rows
from perceptiongate.evaluation.matching import MatchedArrays, match_images
from perceptiongate.evaluation.predict import ModelFactory, predict_val
from perceptiongate.evaluation.report import build_report, to_markdown
from perceptiongate.training.train import resolve_device

ARRAYS = "eval_arrays.npz"
REPORT = "slice_report.json"
REPORT_MD = "slice_report.md"


def score(
    cfg: ExperimentConfig,
    arrays: MatchedArrays,
    out_dir: Path,
    extra: dict[str, Any] | None = None,
    n_boot_ap: int = 100,
    n_boot_count: int = 1000,
) -> dict[str, Any]:
    start = time.monotonic()
    rows = {r.name: r for r in read_rows(cfg.data.manifest, "val")}
    report = build_report(
        arrays,
        rows,
        cfg.data.classes,
        cfg.data.image_width,
        cfg.data.image_height,
        n_boot_ap=n_boot_ap,
        n_boot_count=n_boot_count,
        seed=cfg.data.seed,
    )
    report["experiment"] = cfg.name
    report["timing"] = {**(extra or {}), "score_seconds": round(time.monotonic() - start, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / REPORT_MD).write_text(to_markdown(report, f"Slice report: {cfg.name}"))
    return report


def evaluate(
    cfg: ExperimentConfig,
    weights: Path,
    dataset_dir: Path,
    out_dir: Path,
    *,
    op_conf: float = 0.25,
    n_boot_ap: int = 100,
    n_boot_count: int = 1000,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    rows = read_rows(cfg.data.manifest, "val")
    images, timing = predict_val(
        weights,
        dataset_dir,
        rows,
        cfg.data.image_size,
        resolve_device(cfg.train.device),
        model_factory=model_factory,
    )
    start = time.monotonic()
    arrays = match_images(images, op_conf=op_conf)
    timing["match_seconds"] = round(time.monotonic() - start, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays.save(str(out_dir / ARRAYS))
    return score(cfg, arrays, out_dir, timing, n_boot_ap, n_boot_count)
