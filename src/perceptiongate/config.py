"""Typed experiment configuration.

Every run is described by one YAML file. Loading it through these models means a typo
or a bad value fails immediately, before any GPU time is spent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown keys, so misspelled settings are caught."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    root: Path = Path("data/bdd100k_subset")
    train_images: int = Field(gt=0)
    val_images: int = Field(gt=0)
    classes: list[str] = Field(min_length=1)
    image_size: int = Field(default=640, ge=32)
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
