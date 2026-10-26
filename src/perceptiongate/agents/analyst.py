"""Analyst agent: turns a gate result into a PR comment people can act on.

Flow (one LLM call, at most one retry):
  facts (code) -> LLM writes the explanation -> grounding check
     ok          -> use it
     not ok      -> retry once, telling the model exactly which numbers were wrong
     still not   -> deterministic summary, and the comment says the LLM text was rejected
The gate decision in the comment header always comes from gate_result.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from perceptiongate.agents.grounding import GroundingResult, check_text
from perceptiongate.agents.llm import LLM

SYSTEM = """You are the analyst in an ML release pipeline for a driving-perception \
object detector. A deterministic gate has already decided whether the candidate model \
may replace the baseline. You explain that decision to engineers and reviewers. You do \
not change it.

Rules:
- Use only numbers that appear in FACTS. Do not compute new numbers, do not round \
differently, do not invent metrics. Changes are in percentage points ("points").
- The gate decision is already printed above your text. Do not contradict it.
- Say which config change most likely explains the result, if FACTS lists one.
- In "What happened", name the checks in blocking_failures: those decided the result.
  If real_drops_within_tolerance is not empty, add one sentence saying those checks
  passed, but the drops are real.
- Do not restate allowed drops or thresholds: the gate table below your text has them.
- real_drops_within_tolerance are confirmed drops: their interval is entirely below
  zero. They passed because they are small, NOT because they are noise. Never call them
  noise.
- If a check is listed as too noisy to decide, say a pass there is weak evidence.
- Per-class numbers with boxes_in_val under few_boxes_threshold are uncertain: say so.
- Use bullet points, never numbered lists. No headings other than the three below.
- 120 to 220 words, in plain, direct English.

Write exactly three sections:
**What happened**
**Why it matters**
**Suggested next step**"""


def user_prompt(facts: dict[str, Any], feedback: str | None = None) -> str:
    text = "FACTS:\n" + json.dumps(facts, indent=2)
    if feedback:
        text += f"\n\nYour previous answer was rejected: {feedback}. Write it again."
    return text


def _fmt_check(c: dict[str, Any]) -> str:
    lo, hi = c["interval_points"]
    return (
        f"{c['metric']} on {c['slice']}: {c['baseline']:.3f} → {c['candidate']:.3f} "
        f"({c['change_points']:+.1f} points, interval {lo:+.1f} to {hi:+.1f}; "
        f"allowed drop {c['allowed_drop_points']:.1f})"
    )


def _grouped(checks: list[dict[str, Any]]) -> list[str]:
    """One line per check run on a single slice; one line per check across many slices."""
    by_check: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in checks:
        by_check.setdefault((c["check"], c["metric"]), []).append(c)
    lines = []
    for (_, metric), group in by_check.items():
        if len(group) == 1:
            lines.append(_fmt_check(group[0]))
            continue
        changes = sorted(c["change_points"] for c in group)
        slices = ", ".join(c["slice"] for c in group)
        lines.append(
            f"{metric} dropped in {slices}, by {abs(changes[-1]):.1f} to "
            f"{abs(changes[0]):.1f} points (allowed drop {group[0]['allowed_drop_points']:.1f})"
        )
    return lines


def deterministic_summary(facts: dict[str, Any]) -> str:
    """Plain summary written by code. Used without an LLM, or when its text is rejected."""
    label = facts["decision"].replace("_", " ")
    lines = ["**What happened**", f"- Gate decision: **{label}**."]
    if facts["config_changes"]:
        lines.append("- Config changes vs baseline: " + "; ".join(facts["config_changes"]) + ".")
    lines += [f"- Blocking: {_fmt_check(c)}" for c in facts["blocking_failures"]]
    lines += [f"- Needs review: {line}" for line in _grouped(facts["needs_review"])]
    lines += ["", "**Why it matters**"]
    drops = facts["real_drops_within_tolerance"]
    lines += [f"- Real drop within tolerance: {_fmt_check(c)}" for c in drops]
    lines += [
        f"- Class recall {r['class']}: {r['change_points']:+.1f} points "
        f"({r['boxes_in_val']} boxes in val)"
        for r in facts["per_class_recall"]["largest_drops"]
    ]
    if facts["too_noisy_to_decide"]:
        lines.append("- Too noisy to decide: " + ", ".join(facts["too_noisy_to_decide"]) + ".")
    if len(lines) == lines.index("**Why it matters**") + 1:
        lines.append("- No confirmed drops beyond the gate's checks.")
    lines += ["", "**Suggested next step**"]
    if facts["decision"] == "FAIL":
        lines.append("- Do not merge. Revert or rework the change listed above and re-run.")
    elif facts["decision"] == "NEEDS_REVIEW":
        lines.append("- A person should look at the flagged checks before merging.")
    else:
        lines.append("- No blocking issues. Review the drops within tolerance, then merge.")
    return "\n".join(lines)


def decision_line(facts: dict[str, Any]) -> str:
    return f"**Gate decision: {facts['decision'].replace('_', ' ')}.** (written by code)"


@dataclass
class Explanation:
    text: str
    source: str  # "llm" or "deterministic"
    model: str | None
    grounding: GroundingResult | None
    trace: dict[str, Any] = field(default_factory=dict)


def explain(facts: dict[str, Any], llm: LLM | None, max_attempts: int = 2) -> Explanation:
    trace: dict[str, Any] = {"provider": getattr(llm, "name", "none"), "attempts": []}
    if llm is None:
        return Explanation(deterministic_summary(facts), "deterministic", None, None, trace)
    feedback = None
    for _ in range(max_attempts):
        attempt: dict[str, Any] = {"feedback_given": feedback}
        try:
            reply = llm.complete(SYSTEM, user_prompt(facts, feedback))
        except Exception as exc:  # noqa: BLE001 - an LLM outage must not break the pipeline
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            trace["attempts"].append(attempt)
            break
        grounding = check_text(reply.text, facts)
        attempt.update(
            model=reply.model,
            seconds=round(reply.seconds, 2),
            usage=reply.usage,
            response=reply.text,
            numbers_checked=grounding.checked,
            unverified_numbers=grounding.unverified,
            contradictions=grounding.contradictions,
        )
        trace["attempts"].append(attempt)
        if grounding.ok:
            text = f"{decision_line(facts)}\n\n{reply.text.strip()}"
            return Explanation(text, "llm", reply.model, grounding, trace)
        problems = []
        if grounding.unverified:
            problems.append("these numbers are not in FACTS: " + ", ".join(grounding.unverified))
        problems += grounding.contradictions
        feedback = "; ".join(problems)
    trace["fallback"] = "deterministic summary (LLM text failed the grounding check)"
    return Explanation(deterministic_summary(facts), "deterministic", None, None, trace)


HEADERS = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "NEEDS_REVIEW": "⚠️ NEEDS REVIEW"}


def pr_comment(facts: dict[str, Any], exp: Explanation, gate_markdown: str) -> str:
    head = f"## PerceptionGate: {HEADERS[facts['decision']]} · `{facts['candidate']}`"
    if exp.source == "llm" and exp.grounding is not None:
        footer = (
            f"Explanation written by `{exp.model}`. All {exp.grounding.checked} numbers in it "
            "were checked against `gate_result.json`. The decision comes from "
            "`configs/gate/policy.yaml`, not from the model."
        )
    elif exp.trace.get("fallback"):
        footer = (
            "The LLM explanation was rejected by the grounding check, so this summary was "
            "written by code. See `agent_trace.json`. The decision comes from "
            "`configs/gate/policy.yaml`."
        )
    else:
        footer = (
            "Summary written by code (no LLM configured). Decision from `configs/gate/policy.yaml`."
        )
    return (
        f"{head}\n\n{exp.text}\n\n<details><summary>Full gate result</summary>\n\n"
        f"{gate_markdown}\n</details>\n\n<sub>{footer}</sub>\n"
    )
