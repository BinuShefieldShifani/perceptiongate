from pathlib import Path
from typing import Any

import pytest

from perceptiongate.agents.analyst import deterministic_summary
from perceptiongate.agents.evaluate import evaluate_agent, load_cases, to_markdown
from perceptiongate.agents.grounding import check_text
from perceptiongate.agents.llm import Reply

ROOT = Path(__file__).resolve().parents[2]
CASES = load_cases(ROOT / "evals" / "agent" / "cases")


def test_cases_from_the_three_real_runs_are_present() -> None:
    assert set(CASES) == {"short-25ep", "seed7", "yolo11s"}
    assert {c["expect"]["decision"] for c in CASES.values()} == {"FAIL", "PASS"}


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_case_is_internally_consistent(name: str) -> None:
    """The code-written summary for a case must itself pass every check; otherwise the
    case asks for something impossible."""
    facts = CASES[name]["facts"]
    g = check_text(deterministic_summary(facts), facts)
    assert g.ok, (g.unverified, g.contradictions)


GOOD = {
    "short-25ep": "The gate failed: overall recall fell 2.1 points and small-object recall "
    "2.5 points. Training epochs went from 50 to 25.",
    "seed7": "The candidate passed. Only the training seed changed, from 42 to 7.",
    "yolo11s": "The candidate passed. Switching the backbone to yolo11s raised small-object "
    "recall 9.0 points and pedestrian recall 8.6 points.",
}


class ScriptedLLM:
    name = "scripted"

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = {k: list(v) for k, v in answers.items()}

    def complete(self, system: str, user: str) -> Reply:
        case = next(k for k in self.answers if f'"candidate": "{k}"' in user)
        return Reply(self.answers[case].pop(0), "scripted-model", 1.5)


def test_perfect_answers_score_full_marks() -> None:
    r = evaluate_agent(CASES, ScriptedLLM({k: [v] for k, v in GOOD.items()}))
    assert r["grounded_first_try"] == 1.0
    assert r["used_llm_text"] == 1.0
    assert r["cause_named"] == 1.0
    assert r["model"] == "scripted-model"


def test_scores_show_retries_and_fallbacks() -> None:
    answers: dict[str, list[Any]] = {
        "short-25ep": ["The gate failed, recall fell 7.7 points.", GOOD["short-25ep"]],  # retry
        "seed7": ["It passed.", "It passed, nothing changed."],  # never names the seed
        "yolo11s": ["Passed, much better.", "Passed, much better."],  # never states numbers
    }
    r = evaluate_agent(CASES, ScriptedLLM(answers))
    rows = {row["case"]: row for row in r["rows"]}
    assert rows["short-25ep"]["attempts"] == 2 and rows["short-25ep"]["used_llm_text"]
    assert "7.7" in rows["short-25ep"]["first_try_problems"]
    assert rows["seed7"]["used_llm_text"] and not rows["seed7"]["cause_named"]
    assert not rows["yolo11s"]["used_llm_text"]  # rejected twice -> code summary
    assert r["grounded_first_try"] == pytest.approx(1 / 3, abs=0.01)
    md = to_markdown(r)
    assert "| yolo11s |" in md and "headline change" in md


def test_no_llm_scores_zero_without_crashing() -> None:
    r = evaluate_agent(CASES, None)
    assert r["used_llm_text"] == 0.0 and r["model"] == "none"


@pytest.mark.parametrize("name", sorted(CASES))
def test_cause_keywords_are_not_the_candidate_name(name: str) -> None:
    """The yolo11s case once had 'yolo11s' as a cause keyword: any answer that named the
    model 'named the cause', which made the score look better than it was."""
    case = CASES[name]
    candidate = case["facts"]["candidate"].lower()
    for k in case["expect"]["cause_keywords"]:
        assert k.lower() not in candidate and candidate not in k.lower()
