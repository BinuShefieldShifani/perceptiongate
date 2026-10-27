"""Evaluate the analyst agent on fixed cases taken from real runs.

For each case: run the agent exactly as the pipeline does, then score what came back.
  grounded_first_try   the first answer passed every check
  used_llm_text        the comment ended up with the LLM's text (possibly after a retry)
  cause_named          the LLM's text names the cause (one of the case's keywords)
A run that falls back to the code summary scores no cause: the point is the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from perceptiongate.agents.analyst import explain
from perceptiongate.agents.llm import LLM


def load_cases(directory: Path) -> dict[str, dict[str, Any]]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))}


def evaluate_agent(cases: dict[str, dict[str, Any]], llm: LLM | None) -> dict[str, Any]:
    rows = []
    models: set[str] = set()
    for name, case in cases.items():
        exp = explain(case["facts"], llm)
        attempts = exp.trace.get("attempts", [])
        models |= {a["model"] for a in attempts if a.get("model")}
        first = attempts[0] if attempts else {}
        first_ok = bool(first) and not first.get("error") and not first.get(
            "unverified_numbers") and not first.get("contradictions")  # fmt: skip
        used_llm = exp.source == "llm"
        text = exp.text.lower() if used_llm else ""
        cause = any(k.lower() in text for k in case["expect"]["cause_keywords"])
        rows.append({
            "case": name,
            "expected_decision": case["expect"]["decision"],
            "attempts": len(attempts),
            "grounded_first_try": first_ok,
            "used_llm_text": used_llm,
            "cause_named": cause,
            "seconds": round(sum(a.get("seconds") or 0 for a in attempts), 1),
            "first_try_problems": (first.get("unverified_numbers") or []) +
                                  (first.get("contradictions") or []) +
                                  ([first["error"]] if first.get("error") else []),
        })  # fmt: skip
    n = len(rows) or 1
    return {
        "model": ", ".join(sorted(models)) or getattr(llm, "name", "none"),
        "cases": len(rows),
        "grounded_first_try": round(sum(r["grounded_first_try"] for r in rows) / n, 2),
        "used_llm_text": round(sum(r["used_llm_text"] for r in rows) / n, 2),
        "cause_named": round(sum(r["cause_named"] for r in rows) / n, 2),
        "rows": rows,
    }


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"### Agent evaluation: `{result['model']}` on {result['cases']} cases from real runs",
        "",
        f"Grounded on first try: {result['grounded_first_try']:.0%} · LLM text used: "
        f"{result['used_llm_text']:.0%} · cause named: {result['cause_named']:.0%}",
        "",
        "| Case | Attempts | First try | LLM text | Cause named | Seconds | First-try problems |",
        "|---|---:|---|---|---|---:|---|",
    ]
    yes = {True: "yes", False: "no"}
    for r in result["rows"]:
        problems = "; ".join(r["first_try_problems"])[:120] or "none"
        lines.append(
            f"| {r['case']} | {r['attempts']} | {yes[r['grounded_first_try']]} | "
            f"{yes[r['used_llm_text']]} | {yes[r['cause_named']]} | {r['seconds']} | {problems} |"
        )
    return "\n".join(lines) + "\n"
