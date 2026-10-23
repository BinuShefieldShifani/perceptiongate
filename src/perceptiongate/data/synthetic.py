"""Tiny synthetic YOLO dataset for CI smoke tests.

Solid rectangles on a grey background, written as PNG with only the standard library,
so it works in any CI job. It exists to prove the training code runs, nothing more.
"""

from __future__ import annotations

import random
import struct
import zlib
from pathlib import Path

import yaml

Color = tuple[int, int, int]
CLASS_COLORS: list[Color] = [(220, 40, 40), (40, 90, 220), (40, 180, 60), (230, 200, 30)]


def _png(width: int, height: int, pixels: list[list[Color]]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def make_synthetic(
    out_dir: Path, classes: list[str], n_train: int, n_val: int, size: int, seed: int = 0
) -> Path:
    rng = random.Random(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            pixels: list[list[Color]] = [[(128, 128, 128)] * size for _ in range(size)]
            lines = []
            for _ in range(rng.randint(1, 3)):
                cls = rng.randrange(len(classes))
                w, h = rng.randint(size // 6, size // 2), rng.randint(size // 6, size // 2)
                x, y = rng.randint(0, size - w), rng.randint(0, size - h)
                color = CLASS_COLORS[cls % len(CLASS_COLORS)]
                for yy in range(y, y + h):
                    pixels[yy][x : x + w] = [color] * w
                lines.append(
                    f"{cls} {(x + w / 2) / size:.6f} {(y + h / 2) / size:.6f} "
                    f"{w / size:.6f} {h / size:.6f}"
                )
            (img_dir / f"{split}_{i:04d}.png").write_bytes(_png(size, size, pixels))
            (lbl_dir / f"{split}_{i:04d}.txt").write_text("\n".join(lines) + "\n")
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
        )
    )
    return dataset_yaml
