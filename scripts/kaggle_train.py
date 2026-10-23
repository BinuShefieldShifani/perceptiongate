"""Run one training job on Kaggle.

Fast path (normal): attach your prepared dataset and pass its tar file.
    !python scripts/kaggle_train.py --prepared /kaggle/input/.../bdd100k-v1-640.tar --epochs 3

Slow path (no prepared dataset yet): build the YOLO folder from raw BDD100K. Expect ~45
minutes of slow disk reads before training is fast.
    !python scripts/kaggle_train.py --epochs 3

Outputs go to /kaggle/working/outputs/<experiment>.
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


def find_tar(path: Path) -> Path:
    """Accept the tar itself or any folder containing exactly one .tar."""
    if path.is_file():
        return path
    tars = sorted(path.rglob("*.tar"))
    if len(tars) != 1:
        raise SystemExit(f"Expected one .tar under {path}, found {len(tars)}")
    return tars[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/baseline.yaml")
    p.add_argument("--prepared", type=Path, default=None, help="Prepared .tar or its folder.")
    p.add_argument("--labels-dir", type=Path, default=LABELS)
    p.add_argument("--images-dir", type=Path, default=IMAGES)
    p.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    sys.path.insert(0, "src")
    from perceptiongate.config import load_config

    cfg = load_config(args.config)
    out = args.work / "outputs" / cfg.name
    if out.exists():
        shutil.rmtree(out)

    pgate = [sys.executable, "-m", "perceptiongate.cli"]
    run([*pgate, "data", "verify-manifest", str(cfg.data.manifest)])
    if args.prepared:
        tar = find_tar(args.prepared)
        run([*pgate, "data", "unpack", "--tar", str(tar), "--out", str(args.work / "yolo")])
        dataset = args.work / "yolo" / tar.name.removesuffix(".tar")
    else:
        dataset = args.work / "yolo" / cfg.data.manifest.name
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

    # train's preflight checks the dataset's source_manifest.json against the committed
    # manifest, so a tar built from an older manifest is refused here.
    train_cmd = [*pgate, "train", "--config", args.config,
                 "--dataset-yaml", str(dataset / "dataset.yaml"), "--out", str(out)]  # fmt: skip
    if args.epochs:
        train_cmd += ["--epochs", str(args.epochs)]
    run(train_cmd)

    shutil.rmtree(out / "ultralytics" / "weights", ignore_errors=True)
    print(f"\nOutputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
