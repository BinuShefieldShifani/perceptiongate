import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from perceptiongate.agents.analyst import deterministic_summary, explain, pr_comment
from perceptiongate.agents.facts import class_changes, config_changes
from perceptiongate.agents.grounding import check_text, fact_values
from perceptiongate.agents.llm import Reply, make_llm
from perceptiongate.cli import app

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


def check(
    metric: str, sl: str, b: float, c: float, d: float, lo: float, hi: float
) -> dict[str, Any]:
    return {"check": metric, "metric": metric, "slice": sl, "baseline": b, "candidate": c,
            "change_points": round(100 * d, 1),
            "interval_points": [round(100 * lo, 1), round(100 * hi, 1)],
            "allowed_drop_points": 1.0}  # fmt: skip


FACTS: dict[str, Any] = {
    "candidate": "short-25ep",
    "decision": "FAIL",
    "config_changes": ["train.epochs: 50 -> 25"],
    "blocking_failures": [check("recall", "all", 0.563, 0.542, -0.021, -0.024, -0.017)],
    "needs_review": [],
    "real_drops_within_tolerance": [
        check("vru_recall", "all", 0.366, 0.340, -0.025, -0.039, -0.012)
    ],
    "improvements": [],
    "too_noisy_to_decide": ["vru-recall-by-condition/highway"],
    "per_class_recall": {"largest_drops": [], "largest_gains": []},
    "blocking_interval_level_percent": 98.75,
    "review_interval_level_percent": 95.0,
}
GOOD = (
    "**What happened**\n- The gate says FAIL. train.epochs went from 50 to 25.\n"
    "- Recall fell from 0.563 to 0.542, -2.1 points (interval -2.4 to -1.7).\n"
    "**Why it matters**\n- VRU recall dropped 2.5 points, a real drop within tolerance.\n"
    "**Suggested next step**\n- Keep the longer schedule for mAP50 and yolo11n runs."
)


def test_fact_values_include_points_form() -> None:
    values = fact_values({"a": 0.563, "b": [-0.021], "c": "x: 50 -> 25"})
    assert {0.563, 56.3, 0.021, 2.1, 50.0, 25.0} <= {round(v, 3) for v in values}


def test_grounding_accepts_faithful_text() -> None:
    g = check_text(GOOD, FACTS)
    assert g.ok, (g.unverified, g.contradictions)
    assert g.checked >= 8  # digits inside mAP50 / yolo11n are not counted


@pytest.mark.parametrize(
    ("text", "bad"),
    [
        (GOOD + "\n- That is a relative loss of 3.7%.", "3.7"),
        (
            GOOD.replace("-2.1 points", "-2.4 points ").replace("(interval -2.4", "(interval -9.9"),
            "9.9",
        ),
        (GOOD + "\n- Expect 0.61 after retraining.", "0.61"),
    ],
)
def test_grounding_rejects_invented_numbers(text: str, bad: str) -> None:
    g = check_text(text, FACTS)
    assert not g.ok
    assert bad in [u.lstrip("+-−") for u in g.unverified]


def test_known_limit_number_attached_to_wrong_metric() -> None:
    # -2.4 exists in the facts (an interval end), so it passes even when it is used as the
    # recall change. The check proves numbers come from the facts, not that each one sits
    # next to the right metric. Documented in ADR 0007.
    assert check_text(GOOD.replace("-2.1 points", "-2.4 points"), FACTS).ok


def test_grounding_rejects_contradicting_decision() -> None:
    assert not check_text(GOOD.replace("FAIL", "PASS"), FACTS).ok
    assert not check_text(GOOD + "\n- It is safe to merge.", FACTS).ok


class FakeLLM:
    name = "fake"

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> Reply:
        self.prompts.append(user)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return Reply(r, "fake-model", 0.01, {"output_tokens": 42})


def test_explain_uses_grounded_llm_text() -> None:
    exp = explain(FACTS, FakeLLM([GOOD]))
    assert exp.source == "llm"
    assert exp.trace["attempts"][0]["unverified_numbers"] == []


def test_explain_retries_with_specific_feedback() -> None:
    llm = FakeLLM([GOOD + "\n- A relative loss of 3.7%.", GOOD])
    exp = explain(FACTS, llm)
    assert exp.source == "llm"
    assert len(exp.trace["attempts"]) == 2
    assert "3.7" in llm.prompts[1]  # the model is told exactly which number was wrong


def test_explain_falls_back_to_code_summary() -> None:
    exp = explain(FACTS, FakeLLM([GOOD + " 7.7", GOOD + " 8.8"]))
    assert exp.source == "deterministic"
    assert "fallback" in exp.trace
    assert "rejected by the grounding check" in pr_comment(FACTS, exp, "table")


def test_llm_outage_does_not_break_the_pipeline() -> None:
    exp = explain(FACTS, FakeLLM([ConnectionError("timeout")]))
    assert exp.source == "deterministic"
    assert "ConnectionError" in exp.trace["attempts"][0]["error"]


def test_deterministic_summary_is_itself_grounded() -> None:
    assert check_text(deterministic_summary(FACTS), FACTS).ok


def test_make_llm_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("PGATE_LLM", "ANTHROPIC_API_KEY", "PGATE_LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    assert make_llm() is None
    monkeypatch.setenv("PGATE_LLM", "bogus")
    with pytest.raises(ValueError):
        make_llm()


def test_config_changes_between_real_configs() -> None:
    exp = ROOT / "configs" / "experiments"
    assert config_changes(exp / "baseline.yaml", exp / "short-25ep.yaml") == [
        "train.epochs: 50 -> 25"
    ]
    assert config_changes(exp / "baseline.yaml", exp / "seed7.yaml") == ["train.seed: 42 -> 7"]


def test_class_changes_sorted_by_drop() -> None:
    def rep(recalls: dict[str, float]) -> dict[str, Any]:
        return {"slices": {"all": {"per_class": {
            k: {"recall": v, "n_gt": 100} for k, v in recalls.items()}}}}  # fmt: skip

    out = class_changes(rep({"car": 0.7, "bus": 0.4, "rider": 0.3}),
                        rep({"car": 0.71, "bus": 0.35, "rider": 0.2}))  # fmt: skip
    assert [r["class"] for r in out["largest_drops"]] == ["rider", "bus"]
    assert out["largest_gains"][0]["class"] == "car"


def test_gate_then_explain_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from perceptiongate.config import load_config
    from perceptiongate.evaluation.matching import ImageData, match_images
    from perceptiongate.evaluation.run import score

    monkeypatch.setenv("PGATE_LLM", "none")
    names = [f"{i:05d}.jpg" for i in range(200)]
    man = tmp_path / "manifest"
    man.mkdir()
    rows = ["name,split,timeofday,weather,scene,n_boxes,n_small,n_vru"]
    rows += [f"{n},val,daytime,clear,highway,2,0,1" for n in names]
    (man / "val.csv").write_text("\n".join(rows) + "\n")
    cfg_raw = yaml.safe_load((ROOT / "configs/experiments/baseline.yaml").read_text())
    cfg_raw["data"].update(manifest=str(man), classes=["car", "pedestrian"])
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg_raw))
    cfg = load_config(tmp_path / "cfg.yaml")

    box, ped = [0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.6, 0.7]

    def evaluation(rate: float, seed: int, out: Path) -> None:  # noqa: E306
        rng = np.random.default_rng(seed)
        imgs = []
        for n in names:
            det = [box, ped] if rng.random() < rate else [box]
            k = len(det)
            gt, gc = np.array([box, ped]), np.array([0, 1])
            dc, ds = np.array([0, 1][:k]), np.array([0.9, 0.8][:k])
            imgs.append(ImageData(n, gt, gc, np.array(det, float), dc, ds))
        out.mkdir()
        arrays = match_images(imgs)
        arrays.save(str(out / "eval_arrays.npz"))
        score(cfg, arrays, out, n_boot_ap=5, n_boot_count=20)

    evaluation(0.8, 1, tmp_path / "base")
    evaluation(0.4, 2, tmp_path / "cand")
    policy = {"name": "t", "data_config": str(tmp_path / "cfg.yaml"),
              "baseline": str(tmp_path / "base"), "resamples": {"recall": 200, "map": 10},
              "rules": [{"id": "vru", "metric": "vru_recall", "slices": ["all"],
                         "max_drop": 0.03, "severity": "fail"}]}  # fmt: skip
    (tmp_path / "policy.yaml").write_text(yaml.safe_dump(policy))
    pol = ["--policy", str(tmp_path / "policy.yaml")]

    gate = runner.invoke(app, ["gate", "--candidate", str(tmp_path / "cand"), *pol])
    assert gate.exit_code == 1  # FAIL
    result = runner.invoke(
        app, ["explain", "--candidate", str(tmp_path / "cand"), "--name", "cand", *pol]
    )
    assert result.exit_code == 0, result.output
    comment = (tmp_path / "cand" / "pr_comment.md").read_text()
    assert comment.startswith("## PerceptionGate: ❌ FAIL")
    assert "Full gate result" in comment
    facts = json.loads((tmp_path / "cand" / "agent_facts.json").read_text())
    assert facts["blocking_failures"][0]["metric"] == "vru_recall"
