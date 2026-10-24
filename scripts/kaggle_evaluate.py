"""Evaluate an existing model on Kaggle (GPU), without training.

Attach two datasets: the prepared data (perceptiongate-bdd100k-v1-640) and the model
(e.g. perceptiongate-baseline-bdd100k-v1). Then from the repo root:
    !python scripts/kaggle_evaluate.py --name baseline

Writes /kaggle/working/outputs/eval-<name>/ with slice_report.json/.md and eval_arrays.npz.
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def find_one(pattern: str, what: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if len(hits) != 1:
        raise SystemExit(f"Expected one {what} ({pattern}), found {len(hits)}: {hits}")
    return Path(hits[0])


def find_weights(work: Path) -> Path:
    """best.pt, either directly in an attached dataset or inside its model.zip."""
    hits = sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True))
    # A published run folder also holds MLflow's artifact copy; ours is model/best.pt.
    ours = [h for h in hits if Path(h).parent.name == "model"]
    if len(ours) == 1:
        return Path(ours[0])
    if len(hits) == 1:
        return Path(hits[0])
    zip_path = find_one("/kaggle/input/**/model.zip", "model.zip or best.pt")
    with zipfile.ZipFile(zip_path) as z:
        member = next(m for m in z.namelist() if m.endswith("best.pt"))
        z.extract(member, work / "model")
    return work / "model" / member


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/baseline.yaml")
    p.add_argument("--name", default="baseline")
    p.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    args = p.parse_args()

    pgate = [sys.executable, "-m", "perceptiongate.cli"]
    data = find_one("/kaggle/input/**/*.bin", "prepared dataset .bin")
    tar = args.work / (data.stem + ".tar")
    shutil.copy(data, tar)
    subprocess.run(
        [*pgate, "data", "unpack", "--tar", str(tar), "--out", str(args.work / "yolo")], check=True
    )
    weights = find_weights(args.work)
    print(f"Weights: {weights}", flush=True)
    out = args.work / "outputs" / f"eval-{args.name}"
    subprocess.run(
        [*pgate, "evaluate", "--config", args.config, "--weights", str(weights),
         "--dataset-dir", str(args.work / "yolo" / data.stem), "--out", str(out)],
        check=True,
    )  # fmt: skip
    tar.unlink()
    print(f"\nOutputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
