"""Build a manifest from raw labels: load, clean, sample, validate, write."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from perceptiongate.config import ExperimentConfig
from perceptiongate.data.manifest import write_manifest
from perceptiongate.data.sampling import stratified_sample
from perceptiongate.data.schema import Frame, load_frames
from perceptiongate.data.validate import clean_frames


class DataValidationError(RuntimeError):
    pass


def load_clean(cfg: ExperimentConfig, labels: Path, split: str) -> tuple[list[Frame], Any]:
    d = cfg.data
    return clean_frames(
        load_frames(labels),
        classes=d.classes,
        width=d.image_width,
        height=d.image_height,
        max_invalid_fraction=d.max_invalid_box_fraction,
        split=split,
    )


def build_manifest(
    cfg: ExperimentConfig, train_labels: Path, val_labels: Path, out_dir: Path | None = None
) -> dict[str, Any]:
    d = cfg.data
    s = d.sampling
    out = out_dir or d.manifest
    targets = {"train": (train_labels, s.train_images), "val": (val_labels, s.val_images)}

    selected: dict[str, list[Frame]] = {}
    reports: dict[str, Any] = {}
    for split, (labels, n) in targets.items():
        frames, report = load_clean(cfg, labels, split)
        reports[split] = report.to_dict()
        if not report.ok:
            raise DataValidationError("; ".join(report.errors))
        selected[split] = stratified_sample(
            frames,
            n=n,
            keys=s.strata,
            min_per_stratum=s.min_per_stratum,
            seed=d.seed,
            require_objects=s.require_objects,
        )

    meta = {
        "source_labels": {"train": train_labels.name, "val": val_labels.name},
        "classes": d.classes,
        "sampling": s.model_dump(),
        "seed": d.seed,
    }
    summary = write_manifest(out, selected, meta)
    (out / "validation_report.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
