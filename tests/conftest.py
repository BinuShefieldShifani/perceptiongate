"""Shared fixtures: tiny fake BDD100K label files in both known layouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "experiments" / "baseline.yaml"

TIMES = ["daytime", "night", "dawn/dusk"]
WEATHERS = ["clear", "rainy", "foggy"]


def make_frame(i: int, legacy: bool = False) -> dict[str, Any]:
    ped = "person" if legacy else "pedestrian"
    return {
        "name": f"img_{i:05d}.jpg",
        "attributes": {
            "timeofday": TIMES[i % 3],
            "weather": WEATHERS[(i // 3) % 3],
            "scene": "highway" if i % 2 else "city street",
        },
        "labels": [
            {"category": "car", "box2d": {"x1": 100, "y1": 100, "x2": 300, "y2": 250}},
            {"category": ped, "box2d": {"x1": 600, "y1": 300, "x2": 620, "y2": 330}},
            {"category": "drivable area", "poly2d": []},
            {"category": "lane", "poly2d": []},
        ],
    }


def write_labels(path: Path, n: int, legacy: bool = False, start: int = 0) -> Path:
    path.write_text(json.dumps([make_frame(start + i, legacy) for i in range(n)]))
    return path


@pytest.fixture
def train_labels(tmp_path: Path) -> Path:
    return write_labels(tmp_path / "train.json", 90, legacy=True)


@pytest.fixture
def val_labels(tmp_path: Path) -> Path:
    return write_labels(tmp_path / "val.json", 45, start=1000)


@pytest.fixture
def small_config(tmp_path: Path) -> Path:
    text = BASELINE.read_text()
    text = text.replace("train_images: 10000", "train_images: 30")
    text = text.replace("val_images: 2000", "val_images: 18")
    text = text.replace("min_per_stratum: 50", "min_per_stratum: 2")
    text = text.replace("manifest: manifests/bdd100k-v1", f"manifest: {tmp_path / 'manifest'}")
    cfg = tmp_path / "small.yaml"
    cfg.write_text(text)
    return cfg
