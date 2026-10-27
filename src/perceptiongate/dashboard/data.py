"""Collect run results for the dashboard. Plain Python, no Streamlit, so it is testable.

A run is any folder with a gate_result.json, for example an unzipped Actions artifact
(result/eval/) or a notebook output (outputs/<name>/eval/). The baseline comes from
baselines/<manifest>/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HEADLINE = [("overall-recall", "all", "recall"), ("vru-recall", "all", "VRU recall"),
            ("pedestrian-recall", "all", "pedestrian recall"),
            ("small-object-recall", "all", "small-object recall"),
            ("overall-map50", "all", "mAP50")]  # fmt: skip


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def _commit(eval_dir: Path) -> str | None:
    for d in (eval_dir, eval_dir.parent):
        info = _load(d / "run_info.json")
        if info:
            return str(info.get("git_commit", ""))[:7] or None
    return None


def load_run(eval_dir: Path, root: Path) -> dict[str, Any]:
    gate = _load(eval_dir / "gate_result.json")
    report = _load(eval_dir / "slice_report.json") or {}
    name = report.get("experiment") or eval_dir.parent.name
    row: dict[str, Any] = {
        "run": f"{name} · {eval_dir.relative_to(root).parts[0]}",
        "experiment": name,
        "folder": str(eval_dir),
        "commit": _commit(eval_dir),
        "decision": gate["decision"],
    }
    by_rule = {(r["rule"], r["slice"]): r for r in gate["results"]}
    for rule, sl, label in HEADLINE:
        r = by_rule.get((rule, sl), {})
        row[label] = r.get("candidate")
        row[f"Δ {label} (pt)"] = None if r.get("delta") is None else round(100 * r["delta"], 1)
    lats = gate.get("latency") or []
    if isinstance(lats, dict):  # older single-rule format
        lats = [lats]
    for lat in lats:
        if lat.get("status") != "skipped":
            row[f"latency {lat.get('id', lat['target'])} (x)"] = lat.get("ratio")
    row["vru_by_slice"] = {s: v["vru_recall"]["value"] for s, v in report.get("slices", {}).items()}
    row["gate_md"] = (
        (eval_dir / "gate_result.md").read_text() if (eval_dir / "gate_result.md").is_file() else ""
    )
    row["failure"] = _load(eval_dir / "failure_report.json")
    row["agent_eval"] = _load(eval_dir / "agent_eval" / "agent_eval.json")
    return row


def discover(root: Path) -> list[dict[str, Any]]:
    """Every run under root, newest folder first."""
    dirs = sorted({p.parent for p in root.rglob("gate_result.json")},
                  key=lambda p: p.stat().st_mtime, reverse=True)  # fmt: skip
    return [load_run(d, root) for d in dirs]


def baseline(baseline_dir: Path) -> dict[str, Any]:
    report = _load(baseline_dir / "slice_report.json") or {}
    s = report.get("slices", {}).get("all", {})
    return {
        "run": "baseline",
        "recall": s.get("recall", {}).get("value"),
        "VRU recall": s.get("vru_recall", {}).get("value"),
        "small-object recall": s.get("small_recall", {}).get("value"),
        "mAP50": s.get("map50", {}).get("value"),
        "vru_by_slice": {k: v["vru_recall"]["value"] for k, v in report.get("slices", {}).items()},
    }


def overview(
    runs: list[dict[str, Any]], base: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Flat rows for the main table (no nested fields)."""
    flat = [{k: v for k, v in r.items() if not isinstance(v, (dict, list)) and k not in
             ("gate_md", "folder")} for r in runs]  # fmt: skip
    if base:
        flat.append(
            {k: v for k, v in base.items() if not isinstance(v, dict)} | {"decision": "reference"}
        )
    return flat


def vru_heatmap(
    runs: list[dict[str, Any]], base: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """{run: {slice: VRU recall}} for a run-by-slice table."""
    table = {r["run"]: r["vru_by_slice"] for r in runs}
    if base:
        table = {"baseline": base["vru_by_slice"], **table}
    return table
