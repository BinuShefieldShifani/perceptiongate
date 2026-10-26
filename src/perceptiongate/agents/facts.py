"""The fact sheet: everything the analyst agent may say, computed by code.

The LLM never sees raw arrays or the full report. It gets this compact, already-correct
summary, and every number it writes is later checked against it. Deltas are given in
percentage points, the way people talk about recall changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from perceptiongate.gate.policy import Policy
from perceptiongate.observability.tracking import flatten

IGNORED_CONFIG_KEYS = {"name", "description"}
# Per-class recall on fewer boxes than this is too uncertain to read much into.
FEW_BOXES = 200


def _pts(x: float | None) -> float | None:
    return None if x is None else round(100 * x, 1)


def _check(r: dict[str, Any]) -> dict[str, Any]:
    lo, hi = r["delta_ci"]
    return {
        "check": r["rule"],
        "metric": r["metric"],
        "slice": r["slice"],
        "baseline": r["baseline"],
        "candidate": r["candidate"],
        "change_points": _pts(r["delta"]),
        "interval_points": [_pts(lo), _pts(hi)],
        "allowed_drop_points": _pts(r["max_drop"]),
    }


def config_changes(baseline_cfg: Path, candidate_cfg: Path | None) -> list[str]:
    if candidate_cfg is None:
        return []
    base = flatten(yaml.safe_load(baseline_cfg.read_text()))
    cand = flatten(yaml.safe_load(candidate_cfg.read_text()))
    keys = sorted((set(base) | set(cand)) - IGNORED_CONFIG_KEYS)
    return [f"{k}: {base.get(k)} -> {cand.get(k)}" for k in keys if base.get(k) != cand.get(k)]


def class_changes(
    base_report: dict[str, Any], cand_report: dict[str, Any], top: int = 3
) -> dict[str, list[dict[str, Any]]]:
    b = base_report["slices"]["all"]["per_class"]
    c = cand_report["slices"]["all"]["per_class"]
    rows = []
    for cls, m in b.items():
        if m["recall"] is None or c.get(cls, {}).get("recall") is None:
            continue
        rows.append(
            {
                "class": cls,
                "baseline_recall": m["recall"],
                "candidate_recall": c[cls]["recall"],
                "change_points": _pts(c[cls]["recall"] - m["recall"]),
                "boxes_in_val": m["n_gt"],
            }
        )
    rows.sort(key=lambda r: r["change_points"])
    return {
        "largest_drops": [r for r in rows[:top] if r["change_points"] < 0],
        "largest_gains": [r for r in reversed(rows[-top:]) if r["change_points"] > 0],
    }


def build_facts(
    gate: dict[str, Any],
    base_report: dict[str, Any],
    cand_report: dict[str, Any],
    policy: Policy,
    candidate_name: str,
    candidate_cfg: Path | None = None,
) -> dict[str, Any]:
    by_outcome: dict[str, list[dict[str, Any]]] = {"FAIL": [], "NEEDS_REVIEW": []}
    for r in gate["results"]:
        if r.get("outcome") in by_outcome:
            by_outcome[r["outcome"]].append(_check(r))
    return {
        "candidate": candidate_name,
        "decision": gate["decision"],
        "config_changes": config_changes(policy.data_config, candidate_cfg),
        "blocking_failures": by_outcome["FAIL"],
        "needs_review": by_outcome["NEEDS_REVIEW"],
        "real_drops_within_tolerance": [
            _check(r) for r in gate["results"] if r["status"] == "within-tolerance"
        ],
        "improvements": [_check(r) for r in gate["results"] if r["status"] == "improved"],
        "too_noisy_to_decide": gate.get("underpowered", []),
        "per_class_recall": class_changes(base_report, cand_report),
        "few_boxes_threshold": FEW_BOXES,
        "blocking_interval_level_percent": round(100 * gate["fail_ci_level"], 2),
        "review_interval_level_percent": round(100 * (1 - gate["alpha"]), 2),
    }
