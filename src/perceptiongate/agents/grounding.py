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
    label = decision.replace("_", " ")
    if label not in text and decision not in text:
        result.contradictions.append(f"does not state the gate decision ({label})")
    if decision == "FAIL":
        lowered = text.lower()
        result.contradictions += [
            f"says '{p}' on a FAIL" for p in FORBIDDEN_WHEN_FAIL if p in lowered
        ]
    return result
