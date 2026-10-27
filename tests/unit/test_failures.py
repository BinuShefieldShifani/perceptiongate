from pathlib import Path
from typing import Any

import numpy as np
import pytest

from perceptiongate.agents.failures import (
    mine,
    rule_based_proposal,
    run_miner,
    to_markdown,
    validate_proposal,
    write_outputs,
)
from perceptiongate.agents.grounding import check_text
from perceptiongate.agents.llm import Reply
from perceptiongate.config import load_config
from perceptiongate.data.manifest import ManifestRow
from perceptiongate.evaluation.matching import ImageData, match_images

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["car", "pedestrian"]
BASELINE_CFG = load_config(ROOT / "configs" / "experiments" / "baseline.yaml")


def scene(miss_small: bool, n: int = 120) -> tuple[Any, dict[str, ManifestRow]]:
    """Each image: one small and one large pedestrian. Optionally the small one is missed."""
    small = [0.50, 0.50, 0.508, 0.514]  # about 10x10 px at 1280x720
    large = [0.10, 0.10, 0.20, 0.30]
    images, rows = [], {}
    for i in range(n):
        det = [large] if miss_small else [large, small]
        k = len(det)
        gt, gc = np.array([large, small]), np.array([1, 1])
        dc, ds = np.array([1] * k), np.array([0.9] * k)
        images.append(ImageData(f"{i}.jpg", gt, gc, np.array(det, float), dc, ds))
        tod = "night" if i % 2 else "daytime"
        rows[f"{i}.jpg"] = ManifestRow(f"{i}.jpg", "val", tod, "clear", "highway", 2, 1, 2)
    return match_images(images), rows


def report() -> dict[str, Any]:
    cand, rows = scene(miss_small=True)
    base, _ = scene(miss_small=False)
    return mine(cand, base, rows, CLASSES, 1280, 720)


def test_mining_finds_that_small_vru_are_missed() -> None:
    r = report()
    c = r["candidate"]
    assert c["boxes"] == 240 and c["missed"] == 120 and c["recall"] == 0.5
    sizes = {row["value"]: row for row in c["by_size"]}
    assert sizes["under 16 px"]["recall"] == 0.0
    assert sizes["under 16 px"]["share_of_misses"] == 1.0
    base_sizes = {row["value"]: row for row in r["baseline"]["by_size"]}
    assert base_sizes["under 16 px"]["recall"] == 1.0
    assert any("under 16 px" in h for h in r["headline"])


def test_rule_based_proposal_targets_small_objects() -> None:
    p = rule_based_proposal(report(), BASELINE_CFG)
    assert p["key"] == "data.image_size" and p["value"] == 960
    assert "prepared dataset" in p["reason"]


@pytest.mark.parametrize(
    ("prop", "problem"),
    [
        ({"key": "train.batch", "value": 64}, "not in the allowed"),
        ({"key": "train.weights", "value": "yolo99.pt"}, "not one of"),
        ({"key": "train.epochs", "value": 999}, "outside"),
        ({"key": "train.weights", "value": "yolo11n.pt"}, "already"),
    ],
)
def test_invalid_proposals_are_rejected(prop: dict[str, Any], problem: str) -> None:
    cfg, why = validate_proposal(prop, BASELINE_CFG)
    assert cfg is None and problem in why


def test_valid_proposal_becomes_a_loadable_config() -> None:
    cfg, why = validate_proposal({"key": "train.epochs", "value": 75, "reason": "longer"},
                                 BASELINE_CFG)  # fmt: skip
    assert cfg is not None and why == ""
    assert cfg.train.epochs == 75 and cfg.name == "baseline-next"


class FakeLLM:
    name = "fake"

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, system: str, user: str) -> Reply:
        return Reply(self.text, "fake", 0.1)


def test_grounded_llm_text_and_valid_proposal_are_used() -> None:
    proposal = '{"key": "train.weights", "value": "yolo11s.pt", "reason": "bigger"}'
    prose = "All 120 missed VRU boxes are under 16 px; the baseline found them."
    text = f"{prose} ```json\n{proposal}\n```"
    res = run_miner(report(), BASELINE_CFG, FakeLLM(text))
    assert res["text"].startswith("All 120 missed")
    assert res["proposal_source"] == "llm"
    assert res["proposed_config"]["train"]["weights"] == "yolo11s.pt"


def test_invented_numbers_drop_the_text_and_bad_proposal_falls_back() -> None:
    text = 'About 87% are small. ```json\n{"key": "train.batch", "value": 8}\n```'
    res = run_miner(report(), BASELINE_CFG, FakeLLM(text))
    assert res["text"] is None  # 87 isn't in the facts
    assert res["proposal_source"] == "rule-based"
    assert res["trace"]["attempts"][0]["proposal_problem"]


def test_section_is_appended_to_the_pr_comment_once(tmp_path: Path) -> None:
    (tmp_path / "pr_comment.md").write_text("## PerceptionGate: ✅ PASS\n")
    r = report()
    res = run_miner(r, BASELINE_CFG, None)
    write_outputs(tmp_path, r, res)
    write_outputs(tmp_path, r, res)  # running twice must not duplicate the section
    body = (tmp_path / "pr_comment.md").read_text()
    assert body.count("Where the VRU misses are") == 1
    assert (tmp_path / "proposed_config.yaml").is_file()
    assert "```yaml" in to_markdown(r, res)


# ---- completeness: the real vague YOLO11s answer must be sent back ----

YOLO11S_VAGUE = (
    '**What happened**\nThe candidate model "yolo11s" passed the gate with improvements in '
    "overall recall and VRU recall, among other metrics. The most significant gains were "
    "observed in the recall for traffic lights, traffic signs, and motorcycles."
)


def test_vague_answer_without_headline_numbers_is_rejected() -> None:
    facts = {"decision": "PASS", "must_mention": [9.0, 8.6]}
    g = check_text(YOLO11S_VAGUE, facts)
    assert not g.ok
    assert any("9.0" in c for c in g.contradictions)
    better = YOLO11S_VAGUE + " Small-object recall rose 9.0 points and pedestrian recall 8.6."
    assert check_text(better, facts).ok
