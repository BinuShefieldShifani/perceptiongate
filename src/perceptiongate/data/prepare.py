"""Prepared dataset: the manifest's images, resized once, packed into one tar file.

Why: Kaggle serves attached datasets from a slow network mount. Reading 12,000 small
files from it took ~45 minutes per run, while training itself took ~1.5 minutes per epoch.
One large file streams fast from the same mount, so we pay the slow reads once:

    raw BDD100K (7 GB, 80k files)  --prepare-->  bdd100k-v1-640.tar  (one file)
    every training run:  unpack the tar to local disk in seconds, then train

Images are resized so the long side equals data.image_size. Ultralytics resizes to that
size when it loads an image anyway, so the model sees the same pixels as before. YOLO
labels are normalised to 0..1, so they don't change with the resize.

The tar carries source_manifest.json, so training still refuses data that doesn't match
the committed manifest.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from perceptiongate.data.manifest import SPLITS, read_rows
from perceptiongate.data.schema import Frame
from perceptiongate.data.yolo import (
    SOURCE_MANIFEST,
    index_images,
    manifest_fingerprint,
    to_yolo_line,
)

PREPARED_INFO = "prepared.json"


def _resize_one(src: Path, dst: Path, long_side: int, quality: int) -> tuple[int, int]:
    from PIL import Image

    with Image.open(src) as opened:
        rgb = opened.convert("RGB")
    scale = long_side / max(rgb.size)
    if scale < 1:
        size = (round(rgb.width * scale), round(rgb.height * scale))
        rgb = rgb.resize(size, Image.Resampling.BILINEAR)
    rgb.save(dst, "JPEG", quality=quality)
    return int(rgb.width), int(rgb.height)


def prepare_dataset(
    manifest_dir: Path,
    frames_by_split: dict[str, dict[str, Frame]],
    images_by_split: dict[str, Path],
    out_tar: Path,
    classes: list[str],
    width: int,
    height: int,
    long_side: int,
    quality: int = 90,
    workers: int = 32,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    start = time.monotonic()
    name = out_tar.name.removesuffix(".tar")
    index = {c: i for i, c in enumerate(classes)}
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    with tempfile.TemporaryDirectory(dir=out_tar.parent) as tmp:
        root = Path(tmp) / name
        for split in SPLITS:
            img_out = root / "images" / split
            lbl_out = root / "labels" / split
            img_out.mkdir(parents=True)
            lbl_out.mkdir(parents=True)
            found = index_images(images_by_split[split])
            rows = read_rows(manifest_dir, split)
            jobs: list[tuple[Path, Path]] = []
            for row in rows:
                frame = frames_by_split[split].get(row.name)
                src = found.get(row.name)
                if frame is None or src is None:
                    raise FileNotFoundError(f"{split}: {row.name} missing from labels or images")
                stem = Path(row.name).stem
                jobs.append((src, img_out / f"{stem}.jpg"))
                lines = [
                    line
                    for b in frame.boxes
                    if b.category in index
                    and (line := to_yolo_line(b, index[b.category], width, height))
                ]
                (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

            # The mount is latency bound, so many parallel readers help a lot.
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_resize_one, s, d, long_side, quality) for s, d in jobs]
                for future in futures:
                    future.result()
                    done += 1
                    if done % 1000 == 0 or done == len(jobs):
                        elapsed = time.monotonic() - start
                        log(f"  {split}: {done}/{len(jobs)} images ({elapsed:.0f}s)")
            counts[split] = len(jobs)

        (root / "dataset.yaml").write_text(
            # No "path" key: Ultralytics then resolves images relative to this file,
            # so the folder works wherever it is unpacked.
            yaml.safe_dump(
                {"train": "images/train", "val": "images/val", "names": dict(enumerate(classes))},
                sort_keys=False,
            )
        )
        fingerprint = manifest_fingerprint(manifest_dir)
        (root / SOURCE_MANIFEST).write_text(
            json.dumps({"manifest": str(manifest_dir), "sha256": fingerprint}, indent=2) + "\n"
        )
        info = {
            "name": name,
            "manifest_sha256": fingerprint,
            "long_side": long_side,
            "jpeg_quality": quality,
            "images": counts,
            "prepare_seconds": round(time.monotonic() - start, 1),
        }
        (root / PREPARED_INFO).write_text(json.dumps(info, indent=2) + "\n")
        with tarfile.open(out_tar, "w") as tar:
            tar.add(root, arcname=name)
    info["tar_bytes"] = out_tar.stat().st_size
    return info


def unpack_dataset(tar_path: Path, out_dir: Path) -> Path:
    """Extract a prepared tar to local disk. Returns the dataset.yaml path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        top = {m.name.split("/", 1)[0] for m in tar.getmembers()}
        if len(top) != 1:
            raise ValueError(f"{tar_path}: expected one top-level folder, found {sorted(top)}")
        target = out_dir / top.pop()
        if target.exists():
            shutil.rmtree(target)
        try:
            tar.extractall(out_dir, filter="data")
        except TypeError:  # Python < 3.10.12 / 3.11.4 has no filter argument
            tar.extractall(out_dir)  # noqa: S202 - our own artifact
    dataset_yaml = target / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"{dataset_yaml} missing in {tar_path}")
    return dataset_yaml
