"""Per-slice report: the numbers the gate and the agents work from.

For every image slice (all, night, rain, highway, ...) the report gives:
  * mAP50 and mAP50-95, with 95% bootstrap intervals
  * recall and precision at the operating point (confidence >= op_conf, IoU >= 0.5)
  * vulnerable-road-user recall and small-object recall, with intervals
  * per-class AP and recall
Intervals come from resampling images, so a slice with few images gets wide intervals,
which is the honest answer.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from perceptiongate.data.manifest import ManifestRow
from perceptiongate.data.schema import Frame
from perceptiongate.data.slices import IMAGE_SLICES, SMALL_AREA_PX, VULNERABLE_ROAD_USERS
from perceptiongate.evaluation.matching import MatchedArrays
from perceptiongate.evaluation.metrics import (
    CountStat,
    Index,
    bootstrap,
    bootstrap_many,
    class_ap,
    map_values,
    per_image_counts,
)

MIN_IMAGES_FOR_CI = 30


def _r(x: float) -> float | None:
    return None if np.isnan(x) else round(float(x), 4)


def _metric(value: float, ci: tuple[float, float]) -> dict[str, Any]:
    return {"value": _r(value), "ci95": [_r(ci[0]), _r(ci[1])]}


def count_stats(
    arrays: MatchedArrays, classes: list[str], image_width: int, image_height: int
) -> dict[str, CountStat]:
    """Recall-type statistics as per-image counts. Keys: recall, precision, vru_recall,
    small_recall, and recall:<class> for every class."""
    a = arrays
    n = len(a.names)
    vru = [i for i, c in enumerate(classes) if c in VULNERABLE_ROAD_USERS]
    small = a.gt_area * image_width * image_height < SMALL_AREA_PX
    is_vru = np.isin(a.gt_cls, vru)
    op = a.det_score >= a.op_conf
    gt_all = per_image_counts(n, a.gt_img, np.ones(len(a.gt_img), bool))
    stats = {
        "recall": CountStat(per_image_counts(n, a.gt_img, a.gt_hit), gt_all),
        "precision": CountStat(
            per_image_counts(n, a.det_img, op & a.det_tp[:, 0]),
            per_image_counts(n, a.det_img, op),
        ),
        "vru_recall": CountStat(
            per_image_counts(n, a.gt_img, a.gt_hit & is_vru),
            per_image_counts(n, a.gt_img, is_vru),
        ),
        "small_recall": CountStat(
            per_image_counts(n, a.gt_img, a.gt_hit & small),
            per_image_counts(n, a.gt_img, small),
        ),
    }
    for c, name in enumerate(classes):
        is_c = a.gt_cls == c
        stats[f"recall:{name}"] = CountStat(
            per_image_counts(n, a.gt_img, a.gt_hit & is_c), per_image_counts(n, a.gt_img, is_c)
        )
    return stats


def slice_selections(names: list[str], rows: dict[str, ManifestRow]) -> dict[str, Any]:
    """Image indices for every image slice that has at least one image."""
    frames = [
        Frame(name=nm, weather=rows[nm].weather, scene=rows[nm].scene, timeofday=rows[nm].timeofday)
        for nm in names
    ]
    out = {}
    for name, pred in IMAGE_SLICES.items():
        sel = np.array([i for i, f in enumerate(frames) if pred(f)], dtype=np.int64)
        if len(sel):
            out[name] = sel
    return out


def build_report(
    arrays: MatchedArrays,
    rows: dict[str, ManifestRow],
    classes: list[str],
    image_width: int,
    image_height: int,
    n_boot_ap: int = 100,
    n_boot_count: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    idx = Index(arrays, len(classes))
    a = arrays
    vru = [i for i, c in enumerate(classes) if c in VULNERABLE_ROAD_USERS]
    stats = count_stats(arrays, classes, image_width, image_height)
    gt_all = stats["recall"].den
    per_class_hits = [stats[f"recall:{c}"].num for c in classes]

    slices: dict[str, Any] = {}
    for name, sel in slice_selections(a.names, rows).items():
        s_seed = seed + zlib.crc32(name.encode())
        with_ci = len(sel) >= MIN_IMAGES_FOR_CI
        ap = class_ap(idx, sel)
        m50, m5095 = map_values(ap)

        def ci(
            stat: Any, boots: int, s: int = s_seed, sl: Any = sel, on: bool = with_ci
        ) -> tuple[float, float]:
            return bootstrap(stat, sl, boots, s) if on else (float("nan"), float("nan"))

        nan2 = [(float("nan"), float("nan"))] * 2
        map_ci = (
            bootstrap_many(lambda s: map_values(class_ap(idx, s)), sel, n_boot_ap, s_seed)
            if with_ci
            else nan2
        ) or nan2
        entry: dict[str, Any] = {
            "n_images": int(len(sel)),
            "n_gt": int(gt_all[sel].sum()),
            "map50": _metric(m50, map_ci[0]),
            "map50_95": _metric(m5095, map_ci[1]),
            "vru_map50": _r(map_values(ap, vru)[0]),
        }
        for key in ("recall", "precision", "vru_recall", "small_recall"):
            stat = stats[key]
            entry[key] = _metric(stat(sel), ci(stat, n_boot_count))
        entry["n_vru_gt"] = int(stats["vru_recall"].den[sel].sum())
        entry["n_small_gt"] = int(stats["small_recall"].den[sel].sum())
        entry["per_class"] = {
            cls: {
                "n_gt": int(idx.gt_per_img_cls[sel, c].sum()),
                "ap50": _r(ap[c, 0]),
                "ap50_95": _r(float(np.nanmean(ap[c])) if not np.isnan(ap[c, 0]) else float("nan")),
                "recall": _r(CountStat(per_class_hits[c], idx.gt_per_img_cls[:, c])(sel)),
            }
            for c, cls in enumerate(classes)
        }
        if not with_ci:
            entry["note"] = f"fewer than {MIN_IMAGES_FOR_CI} images: no intervals"
        slices[name] = entry

    return {
        "operating_point": {"conf": a.op_conf, "iou": 0.5},
        "bootstrap": {"ap_resamples": n_boot_ap, "count_resamples": n_boot_count, "seed": seed},
        "classes": classes,
        "slices": slices,
    }


def _fmt(m: dict[str, Any]) -> str:
    v, (lo, hi) = m["value"], m["ci95"]
    if v is None:
        return "–"
    return f"{v:.3f}" if lo is None else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def to_markdown(report: dict[str, Any], title: str = "Slice report") -> str:
    lines = [
        f"### {title}",
        "",
        f"Operating point: confidence ≥ {report['operating_point']['conf']}, IoU ≥ 0.5. "
        "Brackets are 95% bootstrap intervals.",
        "",
        "| Slice | Images | mAP50 | Recall | VRU recall | Small-object recall |",
        "|---|---:|---|---|---|---|",
    ]
    for name, s in report["slices"].items():
        lines.append(
            f"| {name} | {s['n_images']} | {_fmt(s['map50'])} | {_fmt(s['recall'])} | "
            f"{_fmt(s['vru_recall'])} | {_fmt(s['small_recall'])} |"
        )
    pc = report["slices"]["all"]["per_class"]
    lines += ["", "| Class | GT boxes | AP50 | Recall |", "|---|---:|---:|---:|"]
    for cls, m in pc.items():
        ap50 = "–" if m["ap50"] is None else f"{m['ap50']:.3f}"
        rec = "–" if m["recall"] is None else f"{m['recall']:.3f}"
        lines.append(f"| {cls} | {m['n_gt']} | {ap50} | {rec} |")
    return "\n".join(lines) + "\n"
