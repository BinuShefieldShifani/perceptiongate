"""Read BDD100K label files into one internal format.

BDD100K labels exist in two JSON layouts on Kaggle mirrors:
  * the 2018 release (bdd100k_labels_images_*.json): categories like "person", "bike"
  * the 2020 detection release (det_*.json): categories like "pedestrian", "bicycle"
Both are a list of frames with "name", "attributes" and "labels" (each label may have
"box2d"). We normalise both so nothing downstream cares which one you downloaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UNDEFINED = "undefined"

# Raw BDD category -> our class name. Anything not listed (lanes, drivable area,
# "other person", "trailer", ...) is ignored on purpose.
CATEGORY_ALIASES: dict[str, str] = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "traffic light": "traffic_light",
    "traffic sign": "traffic_sign",
}


@dataclass(frozen=True)
class Box:
    category: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass(frozen=True)
class Frame:
    name: str
    weather: str = UNDEFINED
    scene: str = UNDEFINED
    timeofday: str = UNDEFINED
    boxes: tuple[Box, ...] = field(default_factory=tuple)

    def attr(self, key: str) -> str:
        value = getattr(self, key)
        assert isinstance(value, str)
        return value


def parse_frame(raw: dict[str, Any]) -> Frame:
    attrs = raw.get("attributes") or {}
    boxes: list[Box] = []
    for label in raw.get("labels") or []:
        box2d = label.get("box2d")
        category = CATEGORY_ALIASES.get(str(label.get("category", "")).lower())
        if box2d is None or category is None:
            continue
        boxes.append(
            Box(
                category=category,
                x1=float(box2d["x1"]),
                y1=float(box2d["y1"]),
                x2=float(box2d["x2"]),
                y2=float(box2d["y2"]),
            )
        )
    return Frame(
        name=str(raw["name"]),
        weather=str(attrs.get("weather") or UNDEFINED),
        scene=str(attrs.get("scene") or UNDEFINED),
        timeofday=str(attrs.get("timeofday") or UNDEFINED),
        boxes=tuple(boxes),
    )


def load_frames(path: str | Path) -> list[Frame]:
    """Load a BDD100K label JSON file (either layout)."""
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "frames" in data:
        data = data["frames"]
    if not isinstance(data, list):
        raise ValueError(f"Unrecognised label file layout: {path}")
    return [parse_frame(item) for item in data]
