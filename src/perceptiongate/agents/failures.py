"""Failure-mining agent: where does the candidate miss vulnerable road users, and what to
try next.

Code does the analysis from the saved predictions: every VRU box in the val set, missed
or found at the operating point, grouped by size, time of day, weather, scene and class,
next to the same numbers for the baseline. The LLM gets those facts, names the main
pattern (grounded like the analyst), and proposes ONE change from a fixed menu. Code
validates the proposal and writes it as a ready-to-run experiment config. A person
decides whether to open it as a PR; the agent never starts a run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from perceptiongate.agents.grounding import check_text
from perceptiongate.agents.llm import LLM
from perceptiongate.config import ExperimentConfig
from perceptiongate.data.manifest import ManifestRow
from perceptiongate.data.slices import VULNERABLE_ROAD_USERS
from perceptiongate.evaluation.matching import MatchedArrays

SIZE_BUCKETS = [(0, 16, "under 16 px"), (16, 32, "16-32 px"), (32, 64, "32-64 px"),
                (64, 128, "64-128 px"), (128, 10**9, "128 px and up")]  # fmt: skip
MIN_BOXES = 30

# What the agent may propose, and what each change needs. Anything else is rejected.
ALLOWED_CHANGES: dict[str, dict[str, Any]] = {
    "train.weights": {"choices": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]},
    "train.epochs": {"min": 10, "max": 150},
    "train.lr0": {"min": 0.001, "max": 0.05},
    "train.optimizer": {"choices": ["SGD", "AdamW"]},
    "data.image_size": {"choices": [640, 960, 1280],
                        "needs": "a new prepared dataset at that size (scripts/kaggle_prepare.py)"},
}  # fmt: skip


def _groups(
    arrays: MatchedArrays, rows: dict[str, ManifestRow], classes: list[str], w: int, h: int
) -> dict[str, Any]:
    img = arrays.gt_img
    size = np.sqrt(arrays.gt_area * w * h)  # geometric mean side, in original pixels
    meta = [rows[n] for n in arrays.names]
    return {
        "size": np.digitize(size, [b[1] for b in SIZE_BUCKETS[:-1]]),
        "timeofday": np.array([meta[i].timeofday for i in img]),
        "weather": np.array([meta[i].weather for i in img]),
        "scene": np.array([meta[i].scene for i in img]),
        "class": np.array([classes[c] for c in arrays.gt_cls]),
    }


def _table(keys: Any, mask: Any, hit: Any, label: Callable[[Any], str]) -> list[dict[str, Any]]:
    total_missed = int((mask & ~hit).sum())
    out = []
    for k in sorted(set(keys[mask].tolist()), key=str):
        m = mask & (keys == k)
        n, missed = int(m.sum()), int((m & ~hit).sum())
        out.append({
            "value": label(k),
            "boxes": n,
            "recall": round(float(hit[m].mean()), 3) if n else None,
            "share_of_misses": round(missed / total_missed, 3) if total_missed else 0.0,
        })  # fmt: skip
    return out


def mine(
    candidate: MatchedArrays,
    baseline: MatchedArrays,
    rows: dict[str, ManifestRow],
    classes: list[str],
    width: int,
    height: int,
) -> dict[str, Any]:
    """VRU miss statistics for candidate and baseline, by dimension. Pure code."""
    vru = np.array([c in VULNERABLE_ROAD_USERS for c in classes])
    report: dict[str, Any] = {"group": "vulnerable road users", "min_boxes": MIN_BOXES}
    for who, arr in (("candidate", candidate), ("baseline", baseline)):
        g = _groups(arr, rows, classes, width, height)
        mask = vru[arr.gt_cls]
        hit = arr.gt_hit
        report[who] = {
            "boxes": int(mask.sum()),
            "missed": int((mask & ~hit).sum()),
            "recall": round(float(hit[mask].mean()), 3),
            "by_size": _table(g["size"], mask, hit, lambda k: SIZE_BUCKETS[int(k)][2]),
            "by_timeofday": _table(g["timeofday"], mask, hit, str),
            "by_weather": _table(g["weather"], mask, hit, str),
            "by_scene": _table(g["scene"], mask, hit, str),
            "by_class": _table(g["class"], mask, hit, str),
        }
    report["headline"] = _headline(report["candidate"])
    return report


def _headline(c: dict[str, Any]) -> list[str]:
    """Largest concentrations of misses, among groups with enough boxes to mean something."""
    lines = []
    for dim in ("by_size", "by_scene", "by_timeofday", "by_class"):
        rows = [r for r in c[dim] if r["boxes"] >= MIN_BOXES]
        if not rows:
            continue
        top = max(rows, key=lambda r: r["share_of_misses"])
        lines.append(
            f"{dim[3:]} '{top['value']}': {top['share_of_misses']:.1%} of missed VRU boxes, "
            f"recall {top['recall']:.3f} on {top['boxes']} boxes"
        )
    return lines


SYSTEM = """You analyse where a driving-perception detector misses vulnerable road users \
(pedestrians, riders, cyclists, motorcyclists). FACTS holds miss statistics computed by \
code, for the candidate and the baseline, by object size, condition and class.

1. In two to four plain sentences, say where most misses are and how that compares with \
the baseline. Use only numbers from FACTS. Groups with fewer boxes than min_boxes are \
too small to draw conclusions from: don't build on them.
2. Propose exactly one next experiment from ALLOWED_CHANGES, as a JSON object in a \
```json block: {"key": "...", "value": ..., "reason": "one sentence"}. If a change lists \
"needs", mention that need in the reason."""


def _proposal_from(text: str) -> dict[str, Any] | None:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def validate_proposal(
    prop: dict[str, Any], cfg: ExperimentConfig
) -> tuple[ExperimentConfig | None, str]:
    key, value = prop.get("key"), prop.get("value")
    rule = ALLOWED_CHANGES.get(str(key))
    if rule is None:
        return None, f"key {key!r} is not in the allowed changes"
    if "choices" in rule and value not in rule["choices"]:
        return None, f"{key}={value!r} is not one of {rule['choices']}"
    if "min" in rule and not (
        isinstance(value, (int, float)) and rule["min"] <= value <= rule["max"]
    ):
        return None, f"{key}={value!r} is outside {rule['min']}..{rule['max']}"
    section, field = str(key).split(".")
    raw = cfg.model_dump(mode="json")
    if raw[section][field] == value:
        return None, f"{key} is already {value!r}"
    raw[section][field] = value
    raw["name"] = f"{cfg.name}-next"
    raw["description"] = f"Proposed by the failure-mining agent: {prop.get('reason', '')}".strip()
    try:
        return ExperimentConfig.model_validate(raw), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"config invalid: {exc}"


def rule_based_proposal(report: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    """Fallback when there is no LLM or its proposal is invalid."""
    small = sum(r["share_of_misses"] for r in report["candidate"]["by_size"]
                if r["value"] in ("under 16 px", "16-32 px"))  # fmt: skip
    if small >= 0.5 and cfg.data.image_size < 960:
        return {"key": "data.image_size", "value": 960,
                "reason": "most missed VRU boxes are small; a larger input keeps more pixels on "
                          "them (needs a new prepared dataset at 960)"}  # fmt: skip
    if cfg.train.weights == "yolo11n.pt":
        return {"key": "train.weights", "value": "yolo11s.pt", "reason": "a larger backbone"}
    return {"key": "train.epochs", "value": min(150, cfg.train.epochs + 25),
            "reason": "a longer schedule"}  # fmt: skip


def run_miner(report: dict[str, Any], cfg: ExperimentConfig, llm: LLM | None) -> dict[str, Any]:
    keep = ("boxes", "missed", "recall", "by_size")
    facts = {"min_boxes": report["min_boxes"], "candidate": report["candidate"],
             "baseline": {k: report["baseline"][k] for k in keep},
             "allowed_changes": ALLOWED_CHANGES}  # fmt: skip
    trace: dict[str, Any] = {"provider": getattr(llm, "name", "none"), "attempts": []}
    text, proposal, source = None, None, "rule-based"
    if llm is not None:
        try:
            reply = llm.complete(SYSTEM, "FACTS:\n" + json.dumps(facts, indent=1))
            prose = reply.text.split("```json")[0].strip()
            g = check_text(prose, {**facts, "decision": "PASS"}, require_decision=False)
            prop = _proposal_from(reply.text)
            valid, why = validate_proposal(prop, cfg) if prop else (None, "no JSON proposal")
            trace["attempts"].append({"model": reply.model, "seconds": round(reply.seconds, 2),
                                      "response": reply.text, "unverified_numbers": g.unverified,
                                      "proposal_problem": why or None})  # fmt: skip
            if g.ok:
                text = prose
            if valid is not None:
                proposal, source = prop, "llm"
        except Exception as exc:  # noqa: BLE001
            trace["attempts"].append({"error": f"{type(exc).__name__}: {exc}"})
    if proposal is None:
        proposal = rule_based_proposal(report, cfg)
    proposed_cfg, why = validate_proposal(proposal, cfg)
    return {"text": text, "proposal": proposal, "proposal_source": source,
            "proposed_config": proposed_cfg.model_dump(mode="json") if proposed_cfg else None,
            "proposal_problem": why or None, "trace": trace}  # fmt: skip


MARKER = "<!-- failure-mining -->"


def to_markdown(report: dict[str, Any], result: dict[str, Any]) -> str:
    c, b = report["candidate"], report["baseline"]
    lines = [MARKER, "### Where the VRU misses are (failure-mining agent)", "",
             f"Candidate misses {c['missed']} of {c['boxes']} VRU boxes (recall {c['recall']:.3f}; "
             f"baseline {b['recall']:.3f}).", ""]  # fmt: skip
    if result["text"]:
        lines += [result["text"], ""]
    lines += ["| Size (geometric mean side) | Boxes | Recall | Baseline | Share of misses |",
              "|---|---:|---:|---:|---:|"]  # fmt: skip
    base = {r["value"]: r for r in b["by_size"]}
    for r in c["by_size"]:
        br = base.get(r["value"], {})
        lines.append(f"| {r['value']} | {r['boxes']} | {r['recall']:.3f} | "
                     f"{br.get('recall', 0):.3f} | {r['share_of_misses']:.1%} |")  # fmt: skip
    lines += [
        "",
        "**Largest concentrations of misses:**",
        *[f"- {h}" for h in report["headline"]],
        "",
    ]
    p = result["proposal"]
    src = (
        "proposed by the agent, checked by code"
        if result["proposal_source"] == "llm"
        else "rule-based proposal (no valid agent proposal)"
    )
    lines += [
        f"**Suggested next experiment** ({src}): `{p['key']}: {p['value']}`. {p.get('reason', '')}"
    ]
    if result["proposed_config"]:
        cfg_yaml = yaml.safe_dump(result["proposed_config"], sort_keys=False).strip()
        lines += ["", "Ready to run: commit this as `configs/experiments/<name>.yaml`, open a PR, "
                  "add the `train` label.", "", "```yaml", cfg_yaml, "```"]  # fmt: skip
    return "\n".join(lines) + "\n"


def write_outputs(out: Path, report: dict[str, Any], result: dict[str, Any]) -> str:
    out.mkdir(parents=True, exist_ok=True)
    (out / "failure_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "failure_trace.json").write_text(json.dumps(result["trace"], indent=2) + "\n")
    if result["proposed_config"]:
        (out / "proposed_config.yaml").write_text(
            yaml.safe_dump(result["proposed_config"], sort_keys=False)
        )
    md = to_markdown(report, result)
    comment = out / "pr_comment.md"
    if comment.is_file():
        body = comment.read_text().split(MARKER)[0].rstrip()
        comment.write_text(body + "\n\n" + md)
    return md
