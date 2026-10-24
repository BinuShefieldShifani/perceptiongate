"""Apply a policy and produce the gate decision. Plain code, no model, no LLM."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from perceptiongate.config import load_config
from perceptiongate.data.manifest import read_rows
from perceptiongate.evaluation.matching import MatchedArrays
from perceptiongate.evaluation.report import slice_selections
from perceptiongate.gate.compare import (
    MetricSource,
    check_comparable,
    paired_bootstrap,
    seed_for,
)
from perceptiongate.gate.policy import MAP_METRICS, Policy

PASS, FAIL, REVIEW = "PASS", "FAIL", "NEEDS_REVIEW"


def _r(x: float) -> float | None:
    return None if math.isnan(x) else round(x, 4)


def classify(delta: float, ci: tuple[float, float], max_drop: float) -> str:
    """regression: dropped more than allowed AND the interval says the drop is real.
    possible-regression: dropped more than allowed, but it could still be noise."""
    if math.isnan(delta):
        return "no-data"
    if delta >= -max_drop:
        return "improved" if not math.isnan(ci[0]) and ci[0] > 0 else "ok"
    if not math.isnan(ci[1]) and ci[1] < 0:
        return "regression"
    return "possible-regression"


def rule_outcome(status: str, severity: str) -> str:
    """How one result affects the decision.

    fail rules (the few checks that matter most): a confirmed regression fails the gate;
    a possible one goes to a human. review rules (many slice checks, so many chances for
    noise): only a confirmed regression asks for review; a possible one is just shown.
    """
    if status == "regression":
        return FAIL if severity == "fail" else REVIEW
    if status == "possible-regression" and severity == "fail":
        return REVIEW
    return PASS


def run_gate(policy: Policy, baseline_arrays: Path, candidate_arrays: Path) -> dict[str, Any]:
    cfg = load_config(policy.data_config)
    base = MatchedArrays.load(str(baseline_arrays))
    cand = MatchedArrays.load(str(candidate_arrays))
    check_comparable(base, cand)

    classes, w, h = cfg.data.classes, cfg.data.image_width, cfg.data.image_height
    src_b, src_c = MetricSource(base, classes, w, h), MetricSource(cand, classes, w, h)
    rows = {r.name: r for r in read_rows(cfg.data.manifest, "val")}
    selections = slice_selections(base.names, rows)

    # Bonferroni across the blocking checks; review checks use the plain level.
    n_fail = sum(len(r.slices) for r in policy.rules if r.severity == "fail")
    fail_alpha = policy.alpha / max(n_fail, 1)

    results: list[dict[str, Any]] = []
    for rule in policy.rules:
        alpha = fail_alpha if rule.severity == "fail" else policy.alpha
        n_boot = policy.resamples.map if rule.metric in MAP_METRICS else policy.resamples.recall
        for slice_name in rule.slices:
            sel = selections.get(slice_name)
            entry: dict[str, Any] = {
                "rule": rule.id,
                "metric": rule.metric,
                "slice": slice_name,
                "severity": rule.severity,
                "max_drop": rule.max_drop,
            }
            if sel is None or len(sel) < policy.min_images:
                entry.update(status="skipped", n_images=0 if sel is None else len(sel))
                results.append(entry)
                continue
            comp = paired_bootstrap(
                src_b.stat(rule.metric),
                src_c.stat(rule.metric),
                sel,
                n_boot,
                seed_for(rule.id, slice_name),
                alpha,
            )
            entry.update(
                n_images=comp.n_images,
                baseline=_r(comp.baseline),
                candidate=_r(comp.candidate),
                delta=_r(comp.delta),
                delta_ci=[_r(comp.ci95[0]), _r(comp.ci95[1])],
                ci_level=round(comp.level, 4),
                noise_floor=_r(comp.noise_floor),
                status=classify(comp.delta, comp.ci95, rule.max_drop),
            )
            entry["outcome"] = rule_outcome(entry["status"], rule.severity)
            entry["underpowered"] = bool(comp.noise_floor > rule.max_drop)
            results.append(entry)

    failing = [r for r in results if r.get("outcome") == FAIL]
    review = [r for r in results if r.get("outcome") == REVIEW]
    decision = FAIL if failing else REVIEW if review else PASS
    return {
        "decision": decision,
        "policy": policy.name,
        "baseline": str(policy.baseline),
        "alpha": policy.alpha,
        "n_fail_checks": n_fail,
        "fail_ci_level": round(1 - fail_alpha, 4),
        "reasons": [_reason(r) for r in failing + review],
        "improvements": [_reason(r) for r in results if r["status"] == "improved"],
        "underpowered": [f"{r['rule']}/{r['slice']}" for r in results if r.get("underpowered")],
        "results": results,
    }


def _reason(r: dict[str, Any]) -> str:
    lo, hi = r["delta_ci"]
    return (
        f"{r['metric']} on {r['slice']}: {r['baseline']:.3f} -> {r['candidate']:.3f} "
        f"({r['delta']:+.3f}, {r['ci_level']:.2%} CI [{lo:+.3f}, {hi:+.3f}]) "
        f"[{r['rule']}, {r['status']}]"
    )


ICONS = {
    "ok": "✅ ok",
    "improved": "⬆️ improved",
    "regression": "❌ regression",
    "possible-regression": "⚠️ possible",
    "skipped": "– skipped",
    "no-data": "– no data",
}


def to_markdown(result: dict[str, Any], candidate_name: str = "candidate") -> str:
    head = {PASS: "✅ PASS", FAIL: "❌ FAIL", REVIEW: "⚠️ NEEDS REVIEW"}[result["decision"]]
    note = (
        f"Blocking checks use {result['fail_ci_level']:.2%} intervals "
        f"(Bonferroni over {result['n_fail_checks']} checks, family-wise α = "
        f"{result['alpha']}); review checks use {1 - result['alpha']:.0%}."
    )
    lines = [f"## Gate: {head}", "", f"`{candidate_name}` vs baseline `{result['baseline']}`, "
             f"policy `{result['policy']}`.", "", note, ""]  # fmt: skip
    if result["reasons"]:
        lines += ["**Why:**", *[f"- {r}" for r in result["reasons"]], ""]
    if result["improvements"]:
        lines += ["**Improved:**", *[f"- {r}" for r in result["improvements"]], ""]
    if result["underpowered"]:
        lines += [
            "**Too noisy to decide on:** "
            + ", ".join(f"`{u}`" for u in result["underpowered"])
            + ". On this val set these checks can't detect a drop as small as their "
            "threshold (noise floor > max_drop), so a pass there is weak evidence.",
            "",
        ]
    lines += [
        "| Rule | Metric | Slice | Baseline | Candidate | Δ [CI] | Noise floor | Result |",
        "|---|---|---|---:|---:|---|---:|---|",
    ]
    for r in result["results"]:
        if r["status"] == "skipped":
            lines.append(
                f"| {r['rule']} | {r['metric']} | {r['slice']} | | | "
                f"{r['n_images']} images | | {ICONS['skipped']} |"
            )
            continue
        lo, hi = r["delta_ci"]
        lines.append(
            f"| {r['rule']} | {r['metric']} | {r['slice']} | {r['baseline']:.3f} | "
            f"{r['candidate']:.3f} | {r['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
            f"±{r['noise_floor']:.3f} | {ICONS[r['status']]} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], out_dir: Path, candidate_name: str) -> None:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "gate_result.md").write_text(to_markdown(result, candidate_name))
