"""Metrics on matched arrays, for any multiset of images, plus bootstrap intervals.

A "selection" is an array of image indices, possibly with repeats (a bootstrap
resample picks some images twice and others not at all). Every metric here takes a
selection, so the same code computes the point estimate and every resample.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from perceptiongate.evaluation.matching import IOU_THRESHOLDS, MatchedArrays

RECALL_POINTS = np.linspace(0.0, 1.0, 101)  # COCO's 101-point interpolation
Selection = NDArray[np.int64]


class Index:
    """Precomputed per-image offsets so a selection maps to detection rows fast."""

    def __init__(self, arrays: MatchedArrays, n_classes: int) -> None:
        self.a = arrays
        self.n_images = len(arrays.names)
        self.n_classes = n_classes
        counts = np.bincount(arrays.det_img, minlength=self.n_images)
        self.det_count = counts
        self.det_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.gt_per_img_cls = np.zeros((self.n_images, n_classes), dtype=np.int64)
        np.add.at(self.gt_per_img_cls, (arrays.gt_img, arrays.gt_cls), 1)

    def det_rows(self, sel: Selection) -> NDArray[np.int64]:
        lengths = self.det_count[sel]
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        starts = self.det_start[sel]
        offsets = np.repeat(starts - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
        return offsets + np.arange(total)


def average_precision(
    scores: NDArray[np.float64], tp: NDArray[np.bool_], n_gt: int
) -> NDArray[np.float64]:
    """AP per IoU threshold for one class. Returns NaN when the class has no ground truth."""
    n_thr = tp.shape[1] if tp.ndim == 2 else len(IOU_THRESHOLDS)
    if n_gt == 0:
        return np.full(n_thr, np.nan)
    if len(scores) == 0:
        return np.zeros(n_thr)
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    ctp = np.cumsum(tp, axis=0)
    cfp = np.cumsum(~tp, axis=0)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1], axis=0)[::-1]
    ap = np.zeros(n_thr)
    for t in range(n_thr):
        idx = np.searchsorted(recall[:, t], RECALL_POINTS, side="left")
        valid = idx < len(scores)
        ap[t] = np.where(valid, precision[np.minimum(idx, len(scores) - 1), t], 0.0).mean()
    return ap


def class_ap(index: Index, sel: Selection, classes: list[int] | None = None) -> NDArray[np.float64]:
    """(n_classes, n_thresholds) AP matrix for a selection; NaN where a class has no GT."""
    rows = index.det_rows(sel)
    a = index.a
    n_gt = index.gt_per_img_cls[sel].sum(axis=0)
    out = np.full((index.n_classes, len(IOU_THRESHOLDS)), np.nan)
    for c in classes if classes is not None else range(index.n_classes):
        r = rows[a.det_cls[rows] == c]
        out[c] = average_precision(a.det_score[r], a.det_tp[r], int(n_gt[c]))
    return out


def map_values(ap: NDArray[np.float64], classes: list[int] | None = None) -> tuple[float, float]:
    """(mAP50, mAP50-95) averaged over classes that have ground truth."""
    m = ap if classes is None else ap[classes]
    present = ~np.isnan(m[:, 0])
    if not present.any():
        return float("nan"), float("nan")
    return float(m[present, 0].mean()), float(m[present].mean())


def bootstrap_many(
    stat: Callable[[Selection], Sequence[float]],
    images: Selection,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> list[tuple[float, float]]:
    """Percentile intervals for several statistics computed from the same resamples."""
    if len(images) == 0 or n_boot <= 0:
        return []
    rng = np.random.default_rng(seed)
    values = np.array(
        [stat(rng.choice(images, size=len(images), replace=True)) for _ in range(n_boot)],
        dtype=float,
    )
    out = []
    for col in values.T:
        col = col[~np.isnan(col)]
        if len(col) == 0:
            out.append((float("nan"), float("nan")))
        else:
            lo, hi = np.quantile(col, [alpha / 2, 1 - alpha / 2])
            out.append((float(lo), float(hi)))
    return out


def bootstrap(
    stat: Callable[[Selection], float],
    images: Selection,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval of `stat` over image-level resamples of `images`."""
    res = bootstrap_many(lambda s: [stat(s)], images, n_boot, seed, alpha)
    return res[0] if res else (float("nan"), float("nan"))


class CountStat:
    """Recall/precision style ratios from per-image counts: cheap to bootstrap."""

    def __init__(self, num: NDArray[np.int64], den: NDArray[np.int64]) -> None:
        self.num = num
        self.den = den

    def __call__(self, sel: Selection) -> float:
        d = float(self.den[sel].sum())
        return float(self.num[sel].sum()) / d if d else float("nan")


def per_image_counts(
    n_images: int, img: NDArray[np.int64], mask: NDArray[np.bool_]
) -> NDArray[np.int64]:
    return np.bincount(img[mask], minlength=n_images).astype(np.int64)
