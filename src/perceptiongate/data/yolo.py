"""Write a manifest out as an Ultralytics YOLO dataset.

Images are symlinked, not copied, so on Kaggle this takes seconds and no extra disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from perceptiongate.data.manifest import SPLITS, read_rows
from perceptiongate.data.schema import Box, Frame


def to_yolo_line(box: Box, class_index: int, width: int, height: int) -> str | None:
    """Box in pixels -> 'cls cx cy w h' normalised to 0..1, clipped to the image."""
    x1, x2 = max(0.0, box.x1), min(float(width), box.x2)
    y1, y2 = max(0.0, box.y1), min(float(height), box.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def materialize(
    manifest_dir: Path,
    frames_by_split: dict[str, dict[str, Frame]],
    images_by_split: dict[str, Path],
    out_dir: Path,
    classes: list[str],
    width: int,
    height: int,
) -> Path:
    """Create images/, labels/ and dataset.yaml. Returns the dataset.yaml path."""
    index = {c: i for i, c in enumerate(classes)}
    for split in SPLITS:
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        frames = frames_by_split[split]
        for row in read_rows(manifest_dir, split):
            frame = frames.get(row.name)
            if frame is None:
                raise KeyError(f"{row.name} is in the manifest but not in the {split} labels")
            src = images_by_split[split] / row.name
            if not src.is_file():
                raise FileNotFoundError(f"Image not found: {src}")
            dst = img_out / row.name
            if not dst.exists():
                os.symlink(src.resolve(), dst)
            lines = [
                line
                for b in frame.boxes
                if b.category in index
                and (line := to_yolo_line(b, index[b.category], width, height))
            ]
            (lbl_out / Path(row.name).with_suffix(".txt").name).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
    dataset_yaml = out_dir / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": dict(enumerate(classes)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dataset_yaml
