"""Run a model over the frozen val set and collect ground truth + predictions."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from perceptiongate.data.manifest import ManifestRow
from perceptiongate.evaluation.matching import ImageData

ModelFactory = Callable[[str], Any]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _np(x: Any) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def read_yolo_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """YOLO txt (cls cx cy w h, normalised) -> (xyxy boxes, classes)."""
    if not path.is_file() or not path.read_text().strip():
        return np.zeros((0, 4)), np.zeros(0, dtype=np.int64)
    data = np.loadtxt(path, ndmin=2)
    cls = data[:, 0].astype(np.int64)
    cx, cy, w, h = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1), cls


def find_image(images_dir: Path, name: str) -> Path:
    stem = Path(name).stem
    for suffix in IMAGE_SUFFIXES:
        p = images_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image for {name} in {images_dir}")


def _default_factory(weights: str) -> Any:
    from ultralytics import YOLO

    return YOLO(weights)


def predict_val(
    weights: Path,
    dataset_dir: Path,
    rows: list[ManifestRow],
    image_size: int,
    device: str,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    chunk: int = 64,
    model_factory: ModelFactory | None = None,
) -> tuple[list[ImageData], dict[str, float]]:
    """Predictions for every val image in manifest order. Low conf on purpose: mAP needs the
    full precision-recall curve, the operating point is applied later."""
    model = (model_factory or _default_factory)(str(weights))
    images_dir, labels_dir = dataset_dir / "images" / "val", dataset_dir / "labels" / "val"
    paths = [find_image(images_dir, r.name) for r in rows]
    out: list[ImageData] = []
    start = time.monotonic()
    for i in range(0, len(paths), chunk):
        batch_rows, batch_paths = rows[i : i + chunk], paths[i : i + chunk]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            conf=conf,
            iou=iou,
            imgsz=image_size,
            max_det=max_det,
            device=device,
            verbose=False,
            stream=True,
        )
        for row, path, r in zip(batch_rows, batch_paths, results, strict=True):
            gt_boxes, gt_cls = read_yolo_labels(labels_dir / f"{path.stem}.txt")
            out.append(
                ImageData(
                    name=row.name,
                    gt_boxes=gt_boxes,
                    gt_cls=gt_cls,
                    det_boxes=_np(r.boxes.xyxyn).astype(np.float64).reshape(-1, 4),
                    det_cls=_np(r.boxes.cls).astype(np.int64).reshape(-1),
                    det_score=_np(r.boxes.conf).astype(np.float64).reshape(-1),
                )
            )
    seconds = time.monotonic() - start
    return out, {"predict_seconds": round(seconds, 1), "images": len(out)}
