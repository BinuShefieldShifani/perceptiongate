"""Grounding check: every number the agent writes must come from the fact sheet.

A number in the text is accepted if some fact value, shown at the text's precision,
equals it. So "2.1 points" is accepted for a fact of -2.1 or of -0.021 (as points), and
"56.3%" for a recall of 0.563. Anything else is reported back as unverified.
Digits inside names (mAP50, yolo11n, short-25ep) are not numbers and are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

NUMBER = re.compile(r"(?<![A-Za-z_\d.\-])[-+−]?\d+(?:\.\d+)?(?![A-Za-z_\d])")

FORBIDDEN_WHEN_FAIL = ("safe to merge", "ready to merge", "can be merged", "ok to merge")

# How a text can claim each decision. Matched case-insensitively on word boundaries.
DECISION_CLAIMS = {
    "PASS": (r"\bPASS(ED)?\b", r"\bpasse[sd]\b", r"\bpassing\b"),
    "FAIL": (r"\bFAIL(ED|S)?\b", r"\bfailing\b"),
    "NEEDS_REVIEW": (r"\bNEEDS[ _]REVIEW\b",),
}
# Phrases that dismiss a drop as noise. Wrong when the facts list confirmed drops.
NOISE_CLAIMS = re.compile(
    r"(due to|just|only|likely|probably|mere(ly)?|random)\s+(\w+\s+){0,2}noise", re.IGNORECASE
)


def fact_values(obj: Any) -> set[float]:
    """All numbers in the facts, plus the same value as points/percent."""
    out: set[float] = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        v = abs(float(obj))
        out.add(v)
        if v <= 1:
            out.add(v * 100)
        return out
    if isinstance(obj, str):
        return {abs(float(m.group().replace("−", "-"))) for m in NUMBER.finditer(obj)}
    if isinstance(obj, dict):
        items: Any = obj.values()
    elif isinstance(obj, (list, tuple)):
        items = obj
    else:
        return out
    for item in items:
        out |= fact_values(item)
    return out


def _matches(text_number: str, allowed: set[float]) -> bool:
    raw = text_number.replace("−", "-").lstrip("+-")
    value = float(raw)
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    half_step = 0.5 * 10**-decimals + 1e-9
    return any(abs(value - a) <= half_step for a in allowed)


@dataclass
class GroundingResult:
    checked: int = 0
    unverified: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unverified and not self.contradictions


def check_text(text: str, facts: dict[str, Any]) -> GroundingResult:
    allowed = fact_values(facts)
    result = GroundingResult()
    for m in NUMBER.finditer(text):
        result.checked += 1
        if not _matches(m.group(), allowed):
            result.unverified.append(m.group())
    decision = facts["decision"]
    # The decision line is written by code; the text must not claim a different one.
    # "passed" is fine on a FAIL when it describes a single check, so only claims about
    # the gate, the candidate or the model count.
    for other, patterns in DECISION_CLAIMS.items():
        if other == decision:
            continue
        for pat in patterns:
            gate_claim = (
                rf"\b(gate|candidate|model)\W+(\w+\W+){{0,4}}?{pat}"
                rf"|{pat}\W+(\w+\W+){{0,3}}?gate\b"
            )
            if re.search(gate_claim, text, re.IGNORECASE):
                result.contradictions.append(
                    f"claims {other.replace('_', ' ')}, gate says {decision.replace('_', ' ')}"
                )
                break
    if decision == "FAIL":
        lowered = text.lower()
        result.contradictions += [
            f"says '{p}' on a FAIL" for p in FORBIDDEN_WHEN_FAIL if p in lowered
        ]
    noise = NOISE_CLAIMS.search(text) if facts.get("real_drops_within_tolerance") else None
    if noise:
        result.contradictions.append(
            f"calls a drop noise ('{noise.group()}'), but the facts list confirmed real drops"
        )
    return result
