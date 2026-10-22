"""Dataset manifests: the versioned definition of exactly which images we use.

The images are too big for git, but the list of image names is small. A manifest is a CSV
per split plus a manifest.json with counts and SHA-256 hashes. It is committed to git, so
every model can be traced back to the exact data it saw, and CI can check that nobody
edited a manifest by hand.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perceptiongate.data.schema import Frame
from perceptiongate.data.slices import is_small, is_vru

FIELDS = ["name", "split", "timeofday", "weather", "scene", "n_boxes", "n_small", "n_vru"]
SPLITS = ("train", "val")


@dataclass(frozen=True)
class ManifestRow:
    name: str
    split: str
    timeofday: str
    weather: str
    scene: str
    n_boxes: int
    n_small: int
    n_vru: int

    @classmethod
    def from_frame(cls, frame: Frame, split: str) -> ManifestRow:
        return cls(
            name=frame.name,
            split=split,
            timeofday=frame.timeofday,
            weather=frame.weather,
            scene=frame.scene,
            n_boxes=len(frame.boxes),
            n_small=sum(1 for b in frame.boxes if is_small(b)),
            n_vru=sum(1 for b in frame.boxes if is_vru(b)),
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(
    out_dir: Path, splits: dict[str, list[Frame]], meta: dict[str, Any]
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"meta": meta, "splits": {}}
    for split, frames in splits.items():
        rows = [ManifestRow.from_frame(f, split) for f in sorted(frames, key=lambda f: f.name)]
        path = out_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)
        summary["splits"][split] = {
            "file": path.name,
            "sha256": _sha256(path),
            "images": len(rows),
            "boxes": sum(r.n_boxes for r in rows),
            "small_boxes": sum(r.n_small for r in rows),
            "vru_boxes": sum(r.n_vru for r in rows),
            "timeofday": dict(sorted(Counter(r.timeofday for r in rows).items())),
            "weather": dict(sorted(Counter(r.weather for r in rows).items())),
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def read_rows(manifest_dir: Path, split: str) -> list[ManifestRow]:
    with (manifest_dir / f"{split}.csv").open(encoding="utf-8") as fh:
        return [
            ManifestRow(
                name=r["name"],
                split=r["split"],
                timeofday=r["timeofday"],
                weather=r["weather"],
                scene=r["scene"],
                n_boxes=int(r["n_boxes"]),
                n_small=int(r["n_small"]),
                n_vru=int(r["n_vru"]),
            )
            for r in csv.DictReader(fh)
        ]


def verify_manifest(manifest_dir: Path) -> list[str]:
    """Return a list of problems. Empty list means the manifest is intact."""
    problems: list[str] = []
    meta_path = manifest_dir / "manifest.json"
    if not meta_path.is_file():
        return [f"missing {meta_path}"]
    summary = json.loads(meta_path.read_text(encoding="utf-8"))
    seen: dict[str, str] = {}
    for split, info in summary["splits"].items():
        path = manifest_dir / info["file"]
        if not path.is_file():
            problems.append(f"missing {path}")
            continue
        if _sha256(path) != info["sha256"]:
            problems.append(f"{path.name}: hash mismatch (edited after build?)")
        rows = read_rows(manifest_dir, split)
        if len(rows) != info["images"]:
            problems.append(f"{path.name}: {len(rows)} rows, manifest says {info['images']}")
        for row in rows:
            if row.name in seen:
                problems.append(f"{row.name} is in both {seen[row.name]} and {split}")
            seen[row.name] = split
    return problems
