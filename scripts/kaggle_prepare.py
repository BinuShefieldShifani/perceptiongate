"""One-time job on Kaggle: build the prepared dataset tar from raw BDD100K.

Run from the repo root in a Kaggle notebook (CPU is enough, Internet on):
    !python scripts/kaggle_prepare.py

Writes /kaggle/working/prepared/<manifest>-<size>.tar plus dataset-metadata.json, so the
folder can be published as your own private Kaggle dataset with:
    !kaggle datasets create -p /kaggle/working/prepared
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/kaggle/input/datasets/solesensei/solesensei_bdd100k")
LABELS = DEFAULT_ROOT / "bdd100k_labels_release/bdd100k/labels"
IMAGES = DEFAULT_ROOT / "bdd100k/bdd100k/images/100k"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/baseline.yaml")
    p.add_argument("--labels-dir", type=Path, default=LABELS)
    p.add_argument("--images-dir", type=Path, default=IMAGES)
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/prepared"))
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    sys.path.insert(0, "src")
    from perceptiongate.config import load_config

    cfg = load_config(args.config)
    cmd = [
        sys.executable, "-m", "perceptiongate.cli", "data", "prepare",
        "--config", args.config,
        "--train-labels", str(args.labels_dir / "bdd100k_labels_images_train.json"),
        "--val-labels", str(args.labels_dir / "bdd100k_labels_images_val.json"),
        "--train-images", str(args.images_dir / "train"),
        "--val-images", str(args.images_dir / "val"),
        "--out-dir", str(args.out),
        "--workers", str(args.workers),
    ]  # fmt: skip
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    slug = f"perceptiongate-{cfg.data.manifest.name}-{cfg.data.image_size}"
    user = os.environ.get("KAGGLE_USERNAME", "YOUR_KAGGLE_USERNAME")
    meta = {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}
    (args.out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nReady to publish as {meta['id']}:")
    print(f"  kaggle datasets create -p {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
