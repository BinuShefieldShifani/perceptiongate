"""Run one training job on Kaggle.

Fast path (normal): attach your prepared dataset and pass its tar file.
    !python scripts/kaggle_train.py --prepared /kaggle/input/.../bdd100k-v1-640.tar --epochs 3

Slow path (no prepared dataset yet): build the YOLO folder from raw BDD100K. Expect ~45
minutes of slow disk reads before training is fast.
    !python scripts/kaggle_train.py --epochs 3

Full pipeline in one job (train, evaluate, gate, PR comment with a local Ollama model):
    !python scripts/kaggle_train.py --config configs/experiments/<run>.yaml \
        --prepared /kaggle/working/bdd100k-v1-640.tar --agent-model qwen2.5:14b

Outputs go to /kaggle/working/outputs/<experiment>; eval/ holds the slice report, the
gate result and pr_comment.md.
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
    p.add_argument("--no-eval", action="store_true", help="Skip the slice evaluation.")
    p.add_argument("--gate", action="store_true", help="Run the release gate after evaluation.")
    p.add_argument(
        "--baseline-weights",
        type=Path,
        default=None,
        help="Baseline best.pt: benchmark it next to the candidate (same session, same GPU).",
    )
    p.add_argument(
        "--agent-model",
        default=None,
        help="After the gate, write the PR comment with this local Ollama model "
        "(e.g. qwen2.5:14b). Implies --gate.",
    )
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
        # Use the folder the tar actually contains, not a guess from the tar's file name
        # (a copy named data.tar still unpacks to bdd100k-v1-640/).
        from perceptiongate.data.prepare import unpack_dataset

        print(f"+ unpack {tar}", flush=True)
        dataset = unpack_dataset(tar, args.work / "yolo").parent
        print(f"YOLO dataset ready: {dataset / 'dataset.yaml'}", flush=True)
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
    if not args.no_eval:
        run(
            [*pgate, "evaluate", "--config", args.config,
             "--weights", str(out / "model" / "best.pt"),
             "--dataset-dir", str(dataset), "--out", str(out / "eval")]
        )  # fmt: skip

    if args.baseline_weights and not args.no_eval:
        # Latency is a bonus: a benchmark failure must not stop the gate.
        subprocess.run(
            [*pgate, "benchmark", "--config", args.config,
             "--baseline-weights", str(args.baseline_weights),
             "--candidate-weights", str(out / "model" / "best.pt"),
             "--dataset-dir", str(dataset), "--out", str(out / "eval")],
            check=False,
        )  # fmt: skip
    if (args.gate or args.agent_model) and not args.no_eval:
        # Exit code 1 means FAIL, which is a result, not a crash: keep going.
        subprocess.run([*pgate, "gate", "--candidate", str(out / "eval"), "--name", cfg.name],
                       check=False)  # fmt: skip
    if args.agent_model and not args.no_eval:
        import os

        from perceptiongate.agents.ollama_server import ensure_ollama

        print(ensure_ollama([args.agent_model], args.work / "ollama.log"), flush=True)
        env = {**os.environ, "PGATE_LLM": "openai", "PGATE_LLM_MODEL": args.agent_model}
        subprocess.run(
            [*pgate, "explain", "--candidate", str(out / "eval"), "--name", cfg.name,
             "--candidate-config", args.config],
            check=True, env=env,
        )  # fmt: skip

    shutil.rmtree(out / "ultralytics" / "weights", ignore_errors=True)
    print(f"\nOutputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
