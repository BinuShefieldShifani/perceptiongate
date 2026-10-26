"""Run the gate and the analyst agent on Kaggle's GPU with local Ollama models.

Attach the output of the notebook that trained the candidate (Add Input -> Your Work ->
Notebooks), GPU on, Internet on. From the repo root:

    !python scripts/kaggle_agent.py --name short-25ep --models qwen2.5:7b,qwen2.5:14b

For each model it writes /kaggle/working/agent/<model>/pr_comment.md and agent_trace.json,
and a comparison table in /kaggle/working/agent/comparison.md. Everything is zipped to
/kaggle/working/agent-<name>.zip.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")


def find_eval(name: str, explicit: Path | None) -> Path:
    if explicit:
        return explicit
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}/eval/eval_arrays.npz", recursive=True))
    if len(hits) != 1:
        raise SystemExit(
            f"Expected one {name}/eval/eval_arrays.npz under /kaggle/input, found {len(hits)}. "
            "Attach the training notebook's output, or pass --eval-dir."
        )
    return Path(hits[0]).parent


def summarize(model: str, trace: dict) -> dict:  # type: ignore[type-arg]
    attempts = trace.get("attempts", [])
    first = attempts[0] if attempts else {}
    return {
        "model": model,
        "attempts": len(attempts),
        "first_try_grounded": bool(
            first and not first.get("error") and not first.get("unverified_numbers")
            and not first.get("contradictions")
        ),
        "used": "deterministic" if "fallback" in trace or not attempts else "llm",
        "seconds": [a.get("seconds") for a in attempts],
        "numbers_checked": [a.get("numbers_checked") for a in attempts],
        "unverified": [a.get("unverified_numbers") for a in attempts],
        "errors": [a.get("error") for a in attempts if a.get("error")],
    }  # fmt: skip


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Candidate experiment name, e.g. short-25ep.")
    p.add_argument("--eval-dir", type=Path, default=None, help="Candidate eval folder.")
    p.add_argument("--config", type=Path, default=None, help="Candidate experiment config.")
    p.add_argument("--models", default="qwen2.5:7b,qwen2.5:14b")
    p.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    args = p.parse_args()

    from perceptiongate.agents.ollama_server import ensure_ollama

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    eval_dir = find_eval(args.name, args.eval_dir)
    config = args.config or Path(f"configs/experiments/{args.name}.yaml")
    out = args.work / "agent"
    gate_dir = out / "gate"
    shutil.rmtree(out, ignore_errors=True)
    gate_dir.mkdir(parents=True)

    setup = ensure_ollama(models, args.work / "ollama.log")
    print(f"Ollama {setup['version']} ready: {setup}", flush=True)

    pgate = [sys.executable, "-m", "perceptiongate.cli"]
    # The input folder is read-only, so the gate writes next to a copy of the report.
    subprocess.run([*pgate, "gate", "--candidate", str(eval_dir), "--out", str(gate_dir),
                    "--name", args.name], check=False)  # fmt: skip
    shutil.copy(eval_dir / "slice_report.json", gate_dir / "slice_report.json")

    rows = []
    for model in models:
        env = {**os.environ, "PGATE_LLM": "openai", "PGATE_LLM_MODEL": model}
        dest = out / model.replace(":", "-")
        subprocess.run(
            [*pgate, "explain", "--candidate", str(gate_dir), "--name", args.name,
             "--candidate-config", str(config), "--out", str(dest)],
            check=True, env=env, stdout=subprocess.DEVNULL,
        )  # fmt: skip
        rows.append(summarize(model, json.loads((dest / "agent_trace.json").read_text())))

    lines = [
        f"### Analyst agent on `{args.name}`: model comparison",
        "",
        "| Model | Grounded first try | Attempts | Used | Seconds per attempt | Rejected numbers |",
        "|---|---|---:|---|---|---|",
    ]
    for r in rows:
        rejected = "; ".join(", ".join(u) for u in r["unverified"] if u) or "none"
        lines.append(
            f"| {r['model']} | {'yes' if r['first_try_grounded'] else 'no'} | {r['attempts']} | "
            f"{r['used']} | {', '.join(str(s) for s in r['seconds'])} | {rejected} |"
        )
    lines += ["", f"Ollama {setup['version']}, model pull times (s): {setup['pull_seconds']}"]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    (out / "comparison.json").write_text(json.dumps({"setup": setup, "runs": rows}, indent=2))
    print("\n".join(lines))

    archive = shutil.make_archive(str(args.work / f"agent-{args.name}"), "zip", out)
    print(f"\nSaved {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
