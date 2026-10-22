"""Quick summary of a label file, to confirm a dataset mirror has what we need."""

from __future__ import annotations

from collections import Counter
from typing import Any

from perceptiongate.data.schema import Frame


def summarize(frames: list[Frame]) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    for frame in frames:
        categories.update(box.category for box in frame.boxes)
    return {
        "frames": len(frames),
        "frames_with_boxes": sum(1 for f in frames if f.boxes),
        "boxes": sum(categories.values()),
        "categories": dict(categories.most_common()),
        "timeofday": dict(Counter(f.timeofday for f in frames).most_common()),
        "weather": dict(Counter(f.weather for f in frames).most_common()),
        "scene": dict(Counter(f.scene for f in frames).most_common()),
    }
