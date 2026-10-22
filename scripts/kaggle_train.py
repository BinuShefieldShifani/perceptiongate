"""Run one training job on Kaggle: verify manifest -> build YOLO dataset -> train.

Usage inside a Kaggle notebook (GPU on), from the repo root:
    !python scripts/kaggle_train.py --config configs/experiments/baseline.yaml --epochs 3

Everything the run produces ends up in /kaggle/working/outputs/<experiment>, which is what
Kaggle keeps after the notebook finishes. Step 4 downloads that folder from GitHub Actions.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/kaggle/input/datasets/solesensei/solesensei_bdd100k")
LABELS = DEFAULT_ROOT / "bdd100k_labels_release/bdd100k/labels"
IMAGES = DEFAULT_ROOT / "bdd100k/bdd100k/images/100k"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/baseline.yaml")
    p.add_argument("--labels-dir", type=Path, default=LABELS)
    p.add_argument("--images-dir", type=Path, default=IMAGES)
    p.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    sys.path.insert(0, "src")
    from perceptiongate.config import load_config

    cfg = load_config(args.config)
    dataset = args.work / "yolo" / cfg.data.manifest.name
    out = args.work / "outputs" / cfg.name
    if out.exists():
        shutil.rmtree(out)

    pgate = [sys.executable, "-m", "perceptiongate.cli"]
    run([*pgate, "data", "verify-manifest", str(cfg.data.manifest)])
    run(
        [
            *pgate, "data", "materialize", "--config", args.config,
            "--train-labels", str(args.labels_dir / "bdd100k_labels_images_train.json"),
            "--val-labels", str(args.labels_dir / "bdd100k_labels_images_val.json"),
            "--train-images", str(args.images_dir / "train"),
            "--val-images", str(args.images_dir / "val"),
            "--out", str(dataset),
        ]
    )  # fmt: skip
    train_cmd = [*pgate, "train", "--config", args.config,
                 "--dataset-yaml", str(dataset / "dataset.yaml"), "--out", str(out)]  # fmt: skip
    if args.epochs:
        train_cmd += ["--epochs", str(args.epochs)]
    run(train_cmd)

    # Keep outputs small: drop Ultralytics' duplicate weights, keep our copy in model/.
    shutil.rmtree(out / "ultralytics" / "weights", ignore_errors=True)
    print(f"\nOutputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
