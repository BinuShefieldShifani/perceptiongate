from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from perceptiongate.cli import app
from perceptiongate.evaluation.matching import ImageData, match_images
from perceptiongate.gate.compare import GateInputError, paired_bootstrap
from perceptiongate.gate.decide import FAIL, PASS, REVIEW, classify, rule_outcome, run_gate
from perceptiongate.gate.policy import Policy, load_policy

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()
CLASSES = ["car", "pedestrian"]


def test_committed_policy_is_valid() -> None:
    policy = load_policy(ROOT / "configs" / "gate" / "policy.yaml")
    assert any(r.severity == "fail" for r in policy.rules)


@pytest.mark.parametrize(
    "rule",
    [
        {"id": "x", "metric": "accuracy", "slices": ["all"], "max_drop": 0.01, "severity": "fail"},
        {"id": "x", "metric": "recall", "slices": ["moon"], "max_drop": 0.01, "severity": "fail"},
        {"id": "x", "metric": "recall", "slices": ["all"], "max_drop": 0.01, "severity": "warn"},
    ],
)
def test_bad_rules_rejected(rule: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Policy.model_validate(
            {"name": "p", "data_config": "c.yaml", "baseline": "b", "rules": [rule]}
        )


def test_classify_and_outcome() -> None:
    assert classify(-0.005, (-0.02, 0.01), 0.01) == "ok"
    # Confirmed drop smaller than the tolerance: passes, but must not look like "no change".
    assert classify(-0.025, (-0.039, -0.012), 0.03) == "within-tolerance"
    assert rule_outcome("within-tolerance", "fail") == PASS
    assert classify(0.02, (0.005, 0.03), 0.01) == "improved"
    assert classify(-0.03, (-0.05, -0.01), 0.01) == "regression"
    assert classify(-0.03, (-0.06, 0.004), 0.01) == "possible-regression"
    assert rule_outcome("regression", "fail") == FAIL
    assert rule_outcome("regression", "review") == REVIEW
    assert rule_outcome("possible-regression", "fail") == REVIEW
    assert rule_outcome("possible-regression", "review") == PASS  # shown, not escalated


def test_paired_bootstrap_sees_through_shared_difficulty() -> None:
    # Hard and easy images move both models together; only a tiny true difference remains.
    rng = np.random.default_rng(0)
    difficulty = rng.uniform(0, 1, 500)
    base = difficulty
    cand = difficulty - 0.01

    def mean_of(x: np.ndarray):  # type: ignore[no-untyped-def]
        return lambda s: float(x[s].mean())

    comp = paired_bootstrap(mean_of(base), mean_of(cand), np.arange(500), 500, seed=1)
    assert comp.delta == pytest.approx(-0.01)
    assert comp.ci95[1] < 0  # confidently negative, despite huge per-image spread


# ---- end to end on small synthetic evaluations ----

BOX = [0.1, 0.1, 0.3, 0.3]
PED = [0.5, 0.5, 0.6, 0.7]


def evaluation(names: list[str], ped_hit_rate: float, seed: int) -> object:
    rng = np.random.default_rng(seed)
    images = []
    for n in names:
        found_ped = rng.random() < ped_hit_rate
        det = [BOX, PED] if found_ped else [BOX]
        images.append(
            ImageData(
                n,
                np.array([BOX, PED]),
                np.array([0, 1]),
                np.array(det, float),
                np.array([0, 1][: len(det)]),
                np.array([0.9, 0.8][: len(det)]),
            )
        )
    return match_images(images)


@pytest.fixture
def gate_env(tmp_path: Path) -> tuple[Path, list[str]]:
    names = [f"{i:05d}.jpg" for i in range(300)]
    man = tmp_path / "manifest"
    man.mkdir()
    lines = ["name,split,timeofday,weather,scene,n_boxes,n_small,n_vru"]
    for i, n in enumerate(names):
        tod = "night" if i % 2 else "daytime"
        lines.append(f"{n},val,{tod},clear,highway,2,0,1")
    (man / "val.csv").write_text("\n".join(lines) + "\n")
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "baseline.yaml").read_text())
    cfg["data"]["manifest"] = str(man)
    cfg["data"]["classes"] = CLASSES
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    policy = {
        "name": "test",
        "data_config": str(tmp_path / "cfg.yaml"),
        "baseline": str(tmp_path / "base"),
        "resamples": {"recall": 300, "map": 20},
        "rules": [
            {"id": "vru", "metric": "vru_recall", "slices": ["all"], "max_drop": 0.03,
             "severity": "fail"},
            {"id": "vru-cond", "metric": "vru_recall", "slices": ["night", "rain"],
             "max_drop": 0.05, "severity": "review"},
            {"id": "map", "metric": "map50", "slices": ["all"], "max_drop": 0.02,
             "severity": "review"},
        ],
    }  # fmt: skip
    (tmp_path / "policy.yaml").write_text(yaml.safe_dump(policy))
    (tmp_path / "base").mkdir()
    evaluation(names, 0.8, seed=1).save(str(tmp_path / "base" / "eval_arrays.npz"))  # type: ignore[attr-defined]
    return tmp_path, names


def gate(tmp: Path, cand_dir: Path) -> tuple[int, str]:
    result = runner.invoke(
        app, ["gate", "--candidate", str(cand_dir), "--policy", str(tmp / "policy.yaml"),
              "--name", "cand"],
    )  # fmt: skip
    return result.exit_code, result.output


def test_gate_passes_equal_model(gate_env: tuple[Path, list[str]]) -> None:
    tmp, names = gate_env
    cand = tmp / "same"
    cand.mkdir()
    evaluation(names, 0.8, seed=1).save(str(cand / "eval_arrays.npz"))  # type: ignore[attr-defined]
    code, out = gate(tmp, cand)
    assert code == 0, out
    assert "PASS" in out
    assert "rain" in out and "skipped" in out  # no rainy images in this val set


def test_gate_fails_clear_regression(gate_env: tuple[Path, list[str]]) -> None:
    tmp, names = gate_env
    cand = tmp / "worse"
    cand.mkdir()
    evaluation(names, 0.4, seed=2).save(str(cand / "eval_arrays.npz"))  # type: ignore[attr-defined]
    code, out = gate(tmp, cand)
    assert code == 1, out
    assert "FAIL" in out
    result = __import__("json").loads((cand / "gate_result.json").read_text())
    assert result["decision"] == FAIL
    assert any("vru_recall on all" in r for r in result["reasons"])


def test_gate_refuses_different_val_sets(gate_env: tuple[Path, list[str]]) -> None:
    tmp, names = gate_env
    cand = tmp / "other"
    cand.mkdir()
    evaluation(names[:-1], 0.8, seed=1).save(str(cand / "eval_arrays.npz"))  # type: ignore[attr-defined]
    code, out = gate(tmp, cand)
    assert code == 2
    policy = load_policy(tmp / "policy.yaml")
    with pytest.raises(GateInputError):
        run_gate(policy, tmp / "base" / "eval_arrays.npz", cand / "eval_arrays.npz")
