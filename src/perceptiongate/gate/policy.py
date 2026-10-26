"""Gate policy file, validated like the experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator

from perceptiongate.config import StrictModel
from perceptiongate.data.slices import IMAGE_SLICES

COUNT_METRICS = {"recall", "precision", "vru_recall", "small_recall"}
MAP_METRICS = {"map50", "map50_95"}


class Rule(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    metric: str
    slices: list[str] = Field(min_length=1)
    max_drop: float = Field(ge=0, le=1)
    severity: Literal["fail", "review"]

    @field_validator("metric")
    @classmethod
    def known_metric(cls, v: str) -> str:
        if v in COUNT_METRICS | MAP_METRICS or v.startswith("recall:"):
            return v
        raise ValueError(f"unknown metric '{v}'")

    @field_validator("slices")
    @classmethod
    def known_slices(cls, v: list[str]) -> list[str]:
        unknown = [s for s in v if s not in IMAGE_SLICES]
        if unknown:
            raise ValueError(f"unknown slices {unknown}; known: {sorted(IMAGE_SLICES)}")
        return v


class LatencyRule(StrictModel):
    """Candidate vs baseline latency, measured in the same session on the same GPU."""

    target: Literal["gpu", "onnx_cpu"] = "gpu"
    max_ratio: float = Field(gt=1.0)
    severity: Literal["fail", "review"] = "review"


class Resamples(StrictModel):
    recall: int = Field(default=2000, gt=0)
    map: int = Field(default=100, gt=0)


class Policy(StrictModel):
    name: str
    data_config: Path
    baseline: Path
    min_images: int = Field(default=30, gt=0)
    # Family-wise false-alarm rate for the blocking ("fail") checks. It is split evenly
    # across them (Bonferroni), so more fail checks means stricter intervals for each.
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    resamples: Resamples = Resamples()
    rules: list[Rule] = Field(min_length=1)
    latency: LatencyRule | None = None

    @field_validator("rules")
    @classmethod
    def unique_ids(cls, v: list[Rule]) -> list[Rule]:
        ids = [r.id for r in v]
        if len(ids) != len(set(ids)):
            raise ValueError("rule ids must be unique")
        return v


def load_policy(path: str | Path) -> Policy:
    raw = yaml.safe_load(Path(path).read_text())
    return Policy.model_validate(raw)
