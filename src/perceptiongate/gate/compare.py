"""Paired comparison of two models evaluated on the same images.

Both models are scored on identical bootstrap resamples, and we look at the distribution
of the difference. Easy images and hard images move both models together, so the paired
interval is much narrower than comparing two separate intervals. That is what lets the
gate catch a real 1-point recall drop without flagging every random seed.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from perceptiongate.evaluation.matching import MatchedArrays
from perceptiongate.evaluation.metrics import Index, class_ap, map_values
from perceptiongate.evaluation.report import count_stats

Stat = Callable[[Any], float]


class GateInputError(ValueError):
    """The two evaluations can't be compared fairly."""


@dataclass
class Comparison:
    baseline: float
    candidate: float
    delta: float
    ci95: tuple[float, float]  # the interval used for the decision (see `level`)
    n_images: int
    level: float = 0.95

    @property
    def noise_floor(self) -> float:
        """Half-width of the interval: drops smaller than this can't be told from noise."""
        return (self.ci95[1] - self.ci95[0]) / 2


def check_comparable(base: MatchedArrays, cand: MatchedArrays) -> None:
    if base.names != cand.names:
        raise GateInputError(
            "baseline and candidate were evaluated on different images "
            f"({len(base.names)} vs {len(cand.names)}); the val set must be the same"
        )
    if base.op_conf != cand.op_conf:
        raise GateInputError(
            f"different operating points: baseline {base.op_conf}, candidate {cand.op_conf}"
        )


class MetricSource:
    """Builds statistic functions over image selections for one evaluation."""

    def __init__(self, arrays: MatchedArrays, classes: list[str], width: int, height: int):
        self.counts = count_stats(arrays, classes, width, height)
        self.index = Index(arrays, len(classes))

    def stat(self, metric: str) -> Stat:
        if metric in self.counts:
            return self.counts[metric]
        if metric == "map50":
            return lambda s: map_values(class_ap(self.index, s))[0]
        if metric == "map50_95":
            return lambda s: map_values(class_ap(self.index, s))[1]
        raise KeyError(f"unknown metric {metric}")


def paired_bootstrap(
    base: Stat, cand: Stat, images: Any, n_boot: int, seed: int, alpha: float = 0.05
) -> Comparison:
    b, c = base(images), cand(images)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        s = rng.choice(images, size=len(images), replace=True)
        d = cand(s) - base(s)
        if not np.isnan(d):
            deltas.append(d)
    if deltas:
        lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
        ci = (float(lo), float(hi))
    else:
        ci = (float("nan"), float("nan"))
    return Comparison(
        baseline=b, candidate=c, delta=c - b, ci95=ci, n_images=len(images), level=1 - alpha
    )


def seed_for(rule_id: str, slice_name: str) -> int:
    return zlib.crc32(f"{rule_id}/{slice_name}".encode())
