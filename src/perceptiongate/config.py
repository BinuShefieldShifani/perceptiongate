"""Typed experiment configuration.

Every run is described by one YAML file. Loading it through these models means a typo
or a bad value fails immediately, before any GPU time is spent.

Machine-specific paths (where the raw dataset lives on Kaggle or your laptop) are NOT in
the config. They are passed on the command line, so the same config works everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

StratumKey = Literal["timeofday", "weather", "scene"]


class StrictModel(BaseModel):
    """Base model that rejects unknown keys, so misspelled settings are caught."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SamplingConfig(StrictModel):
    train_images: int = Field(gt=0)
    val_images: int = Field(gt=0)
    strata: list[StratumKey] = Field(default=["timeofday", "weather"], min_length=1)
    min_per_stratum: int = Field(default=50, ge=0)
    require_objects: bool = True


class DataConfig(StrictModel):
    manifest: Path = Path("manifests/bdd100k-v1")
    classes: list[str] = Field(min_length=1)
    image_width: int = Field(default=1280, gt=0)
    image_height: int = Field(default=720, gt=0)
    image_size: int = Field(default=640, ge=32)
    max_invalid_box_fraction: float = Field(default=0.005, ge=0, le=1)
    sampling: SamplingConfig
    seed: int = 42

    @field_validator("classes")
    @classmethod
    def classes_unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("classes must be unique")
        return v


class TrainConfig(StrictModel):
    weights: str = "yolo11n.pt"
    epochs: int = Field(gt=0)
    batch: int = Field(gt=0)
    lr0: float = Field(default=0.01, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 42


class ExperimentConfig(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    data: DataConfig
    train: TrainConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping at the top level: {path}")
    return ExperimentConfig.model_validate(raw)
