"""Regression tests from a real run: qwen2.5:14b on the seed7 gate result (Kaggle T4).

Both answers were rejected by the first version of the grounding check. Answer 2 was
correct and was rejected by checker mistakes (a threshold quoted from the prompt, and
"passed" not matching the literal word "PASS"). Answer 1 had a real error the old check
missed: it called confirmed drops "noise".
"""

from typing import Any

from perceptiongate.agents.analyst import explain
from perceptiongate.agents.grounding import check_text
from perceptiongate.agents.llm import Reply


def chk(
    check: str, metric: str, sl: str, b: float, c: float, d: float, lo: float, hi: float
) -> dict[str, Any]:
    return {
        "check": check, "metric": metric, "slice": sl, "baseline": b, "candidate": c,
        "change_points": d, "interval_points": [lo, hi], "allowed_drop_points": 1.5,
    }  # fmt: skip


def cls(name: str, b: float, c: float, d: float, n: int) -> dict[str, Any]:
    return {"class": name, "baseline_recall": b, "candidate_recall": c,
            "change_points": d, "boxes_in_val": n}  # fmt: skip


SEED7: dict[str, Any] = {
    "candidate": "seed7",
    "decision": "PASS",
    "config_changes": ["train.seed: 42 -> 7"],
    "blocking_failures": [],
    "needs_review": [],
    "real_drops_within_tolerance": [
        chk("recall-by-condition", "recall", "highway", 0.537, 0.529, -0.8, -1.4, -0.1),
        chk("overall-map50", "map50", "all", 0.405, 0.392, -1.3, -2.3, -0.1),
    ],
    "improvements": [],
    "too_noisy_to_decide": ["vru-recall-by-condition/highway"],
    "per_class_recall": {
        "largest_drops": [
            cls("motorcycle", 0.186, 0.147, -3.9, 102),
            cls("rider", 0.290, 0.258, -3.2, 124),
            cls("bicycle", 0.222, 0.196, -2.6, 189),
        ],
        "largest_gains": [
            cls("bus", 0.357, 0.370, 1.3, 311),
            cls("pedestrian", 0.387, 0.390, 0.3, 2573),
        ],
    },  # fmt: skip
    "few_boxes_threshold": 200,
    "blocking_interval_level_percent": 98.75,
    "review_interval_level_percent": 95.0,
}

ANSWER_1 = (
    '**What happened**\nThe candidate model "seed7" passed the deterministic gate. The only '
    'config change was setting "train.seed" from 42 to 7. There were real drops within '
    'tolerance for "recall-by-condition/highway" and "overall-map50", both showing drops of '
    "-0.8 and -1.3 points respectively, which are within the allowed drop of 1.5 points. The "
    'per-class recall showed drops for "motorcycle" (-3.9 points), "rider" (-3.2 points), and '
    '"bicycle" (-2.6 points), but these are on few validation boxes (under 200), making them '
    'uncertain. There was also a gain for "bus" (+1.3 points) and a minor gain for '
    '"pedestrian" (+0.3 points).\n\n**Why it matters**\nThe drops in '
    '"recall-by-condition/highway" and "overall-map50" are within the allowed tolerance, '
    "indicating that these changes are likely due to noise rather than a significant "
    'degradation. The per-class drops for "motorcycle," "rider," and "bicycle" are substantial '
    'but on small sample sizes, making them uncertain. The gain in "bus" recall is minor and '
    'does not affect the overall decision. The check "vru-recall-by-condition/highway" was too '
    "noisy to decide, providing weak evidence of a pass.\n\n**Suggested next step**\nGiven the "
    "uncertain per-class drops and the noisy check, it is recommended to gather more data or "
    "run additional validation to confirm the stability of these metrics. If the drops "
    "persist, further investigation into the impact of the seed change on these specific "
    "classes may be necessary."
)

ANSWER_2 = (
    '**What happened**\nThe candidate model "seed7" passed the deterministic gate. The only '
    'config change was "train.seed: 42 -> 7". There were real drops within tolerance for '
    '"recall-by-condition/highway" and "overall-map50", with drops of -0.8 points and -1.3 '
    "points, respectively. The change in seed value is the most likely explanation for these "
    'drops. Additionally, the per-class recall for "motorcycle" and "rider" showed significant '
    "drops of -3.9 points and -3.2 points, respectively, but these are based on few validation "
    "boxes (102 and 124, respectively), making them uncertain.\n\n**Why it matters**\nThe "
    'drops in "recall-by-condition/highway" and "overall-map50" are within the allowed drop '
    "tolerance of 1.5 points, so they do not block the gate. However, the drops in per-class "
    'recall for "motorcycle" and "rider" are substantial, though their small sample sizes make '
    "these results uncertain. The change in seed value likely introduced some variability, but "
    "it did not cause a significant overall performance drop.\n\n**Suggested next step**\n"
    "Since the candidate passed the gate, the next step is to review the impact of the seed "
    "change on other metrics and consider whether further validation is needed, especially "
    'for the "motorcycle" and "rider" classes, given their small sample sizes.'
)


def test_answer_2_now_passes() -> None:
    g = check_text(ANSWER_2, SEED7)
    assert g.ok, (g.unverified, g.contradictions)
    assert g.checked >= 9


def test_answer_1_rejected_only_for_calling_confirmed_drops_noise() -> None:
    g = check_text(ANSWER_1, SEED7)
    assert g.unverified == []  # "200" is now a fact (few_boxes_threshold)
    assert len(g.contradictions) == 1
    assert "noise" in g.contradictions[0]


def test_wrong_decision_is_still_caught() -> None:
    wrong = ANSWER_2.replace("passed the deterministic gate", "failed the deterministic gate")
    assert any("claims FAIL" in c for c in check_text(wrong, SEED7).contradictions)
    on_fail = {**SEED7, "decision": "FAIL"}
    assert any("claims PASS" in c for c in check_text(ANSWER_2, on_fail).contradictions)


def test_passed_single_check_on_a_fail_is_fine() -> None:
    fail = {**SEED7, "decision": "FAIL", "real_drops_within_tolerance": []}
    text = "Motorcycle recall dropped 3.9 points. The VRU check passed with a 1.3 point gain."
    assert check_text(text, fail).ok


class Replay:
    """Replays the two real answers, in order."""

    name = "replay"

    def __init__(self) -> None:
        self.answers = [ANSWER_1, ANSWER_2]

    def complete(self, system: str, user: str) -> Reply:
        return Reply(self.answers.pop(0), "qwen2.5:14b", 1.0)


def test_replay_of_the_real_run_now_ends_with_the_llm_text() -> None:
    exp = explain(SEED7, Replay())
    assert exp.source == "llm"
    assert len(exp.trace["attempts"]) == 2
    assert "noise" in exp.trace["attempts"][1]["feedback_given"]
    assert exp.text.startswith("**Gate decision: PASS.** (written by code)")
