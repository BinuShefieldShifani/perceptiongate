"""Slice definitions shared by data building and evaluation.

An image slice groups images by condition (night, rain, highway). An object slice groups
boxes by what they are (small objects, vulnerable road users). Evaluation in a later
step reports metrics for every slice defined here, so they live in one place.
"""

from __future__ import annotations

from collections.abc import Callable

from perceptiongate.data.schema import Box, Frame

# COCO's definition of a small object: area below 32x32 pixels at original resolution.
SMALL_AREA_PX = 32 * 32

VULNERABLE_ROAD_USERS = frozenset({"pedestrian", "rider", "bicycle", "motorcycle"})

IMAGE_SLICES: dict[str, Callable[[Frame], bool]] = {
    "all": lambda f: True,
    "day": lambda f: f.timeofday == "daytime",
    "night": lambda f: f.timeofday == "night",
    "dawn_dusk": lambda f: f.timeofday == "dawn/dusk",
    "clear": lambda f: f.weather == "clear",
    "rain": lambda f: f.weather == "rainy",
    "snow": lambda f: f.weather == "snowy",
    "fog": lambda f: f.weather == "foggy",
    "highway": lambda f: f.scene == "highway",
    "city_street": lambda f: f.scene == "city street",
}


def is_small(box: Box) -> bool:
    return box.area < SMALL_AREA_PX


def is_vru(box: Box) -> bool:
    return box.category in VULNERABLE_ROAD_USERS


def image_slices_for(frame: Frame) -> list[str]:
    return [name for name, pred in IMAGE_SLICES.items() if pred(frame)]
