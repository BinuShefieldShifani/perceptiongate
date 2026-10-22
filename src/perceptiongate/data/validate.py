"""Label quality checks.

Real labels have mistakes: zero-size boxes, boxes outside the image, duplicate frames.
A few bad boxes are normal and are dropped. If too many are bad, the data is probably
broken (wrong file, wrong resolution) and the build fails instead of training on it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from perceptiongate.data.schema import Box, Frame

EDGE_TOLERANCE_PX = 2.0
TINY_SIDE_PX = 2.0


@dataclass
class ValidationReport:
    frames_in: int = 0
    boxes_in: int = 0
    boxes_kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def invalid_fraction(self) -> float:
        bad = sum(v for k, v in self.dropped.items() if k != "class_not_used")
        return bad / self.boxes_in if self.boxes_in else 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "frames_in": self.frames_in,
            "boxes_in": self.boxes_in,
            "boxes_kept": self.boxes_kept,
            "invalid_fraction": round(self.invalid_fraction, 6),
            "dropped": dict(self.dropped),
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _box_problem(box: Box, width: int, height: int) -> str | None:
    if box.x2 <= box.x1 or box.y2 <= box.y1:
        return "non_positive_size"
    t = EDGE_TOLERANCE_PX
    if box.x1 < -t or box.y1 < -t or box.x2 > width + t or box.y2 > height + t:
        return "outside_image"
    if box.width < TINY_SIDE_PX or box.height < TINY_SIDE_PX:
        return "tiny"
    return None


def clean_frames(
    frames: list[Frame],
    classes: list[str],
    width: int,
    height: int,
    max_invalid_fraction: float,
    split: str,
) -> tuple[list[Frame], ValidationReport]:
    """Drop bad boxes, and report whether the data is healthy enough to use."""
    report = ValidationReport(frames_in=len(frames))
    wanted = set(classes)
    names = Counter(f.name for f in frames)
    dupes = [n for n, c in names.items() if c > 1]
    if dupes:
        report.errors.append(f"{split}: {len(dupes)} duplicate frame names, e.g. {dupes[0]}")

    cleaned: list[Frame] = []
    per_class: Counter[str] = Counter()
    for frame in frames:
        keep: list[Box] = []
        for box in frame.boxes:
            report.boxes_in += 1
            if box.category not in wanted:
                report.dropped["class_not_used"] += 1
                continue
            problem = _box_problem(box, width, height)
            if problem:
                report.dropped[problem] += 1
                continue
            keep.append(box)
            per_class[box.category] += 1
        report.boxes_kept += len(keep)
        cleaned.append(replace(frame, boxes=tuple(keep)))

    if report.invalid_fraction > max_invalid_fraction:
        report.errors.append(
            f"{split}: {report.invalid_fraction:.2%} of boxes are invalid "
            f"(limit {max_invalid_fraction:.2%}). Check the label file and image size."
        )
    for cls in classes:
        if per_class[cls] == 0:
            report.warnings.append(f"{split}: class '{cls}' has no boxes")
    empty = sum(1 for f in cleaned if not f.boxes)
    if empty:
        report.warnings.append(f"{split}: {empty} frames have no usable boxes")
    return cleaned, report
