import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from perceptiongate.agents.facts import _latency_facts
from perceptiongate.evaluation.latency import _summary, benchmark_pair
from perceptiongate.gate.decide import PASS, REVIEW, latency_check, to_markdown
from perceptiongate.gate.policy import Policy, load_policy

ROOT = Path(__file__).resolve().parents[2]


class FakeYOLO:
    """Speed per call depends on the model size, like the real thing."""

    def __init__(self, weights: str) -> None:
        self.ms = 12.0 if "11s" in weights else 7.0
        n = 9_400_000 if "11s" in weights else 2_600_000
        self.model = SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: n)])

    def predict(self, source: str, **kw: Any) -> list[Any]:
        return [
            SimpleNamespace(speed={"preprocess": 1.0, "inference": self.ms, "postprocess": 1.0})
        ]

    def export(self, **kw: Any) -> str:
        raise RuntimeError("no onnx in unit tests")


def test_summary() -> None:
    s = _summary([float(x) for x in range(1, 101)])
    assert s["median_ms"] == 50.5
    assert s["p95_ms"] == 96.0
    assert s["runs"] == 100


def test_benchmark_pair_measures_both_in_one_session(tmp_path: Path) -> None:
    imgs = [tmp_path / "a.jpg"]
    r = benchmark_pair(Path("yolo11n.pt"), Path("yolo11s.pt"), imgs, 640,
                       model_factory=FakeYOLO, gpu_runs=10, cpu_runs=2)  # fmt: skip
    assert r["same_session"] is True
    assert r["baseline"]["gpu"]["median_ms"] == 9.0
    assert r["candidate"]["gpu"]["median_ms"] == 14.0
    assert r["candidate"]["params_m"] == 9.4
    assert "onnx_cpu_error" in r["candidate"]  # ONNX failing is recorded, not fatal


def policy(max_ratio: float = 1.5, severity: str = "review") -> Policy:
    raw = yaml.safe_load((ROOT / "configs" / "gate" / "policy.yaml").read_text())
    raw["latency"] = {"target": "gpu", "max_ratio": max_ratio, "severity": severity}
    return Policy.model_validate(raw)


def latency_file(tmp_path: Path, base_ms: float, cand_ms: float, same: bool = True) -> Path:
    p = tmp_path / "latency.json"
    p.write_text(json.dumps({
        "same_session": same, "gpu": "Tesla T4",
        "baseline": {"params_m": 2.6, "gpu": {"median_ms": base_ms}},
        "candidate": {"params_m": 9.4, "gpu": {"median_ms": cand_ms}},
    }))  # fmt: skip
    return p


def test_committed_policy_has_a_latency_rule() -> None:
    lat = load_policy(ROOT / "configs" / "gate" / "policy.yaml").latency
    assert lat is not None and lat.max_ratio > 1


def test_slower_than_allowed_goes_to_review(tmp_path: Path) -> None:
    r = latency_check(policy(1.5), latency_file(tmp_path, 9.0, 16.2))
    assert r is not None
    assert r["ratio"] == 1.8
    assert r["status"] == "regression"
    assert r["outcome"] == REVIEW


@pytest.mark.parametrize(("cand", "status"), [(10.0, "ok"), (7.0, "improved")])
def test_within_budget_passes(tmp_path: Path, cand: float, status: str) -> None:
    r = latency_check(policy(1.5), latency_file(tmp_path, 9.0, cand))
    assert r is not None and r["status"] == status and r["outcome"] == PASS


def test_latency_needs_a_paired_measurement(tmp_path: Path) -> None:
    r = latency_check(policy(), latency_file(tmp_path, 9.0, 30.0, same=False))
    assert r is not None and r["status"] == "skipped" and r["outcome"] == PASS
    r = latency_check(policy(), tmp_path / "missing.json")
    assert r is not None and r["status"] == "skipped"


def test_latency_in_markdown_and_agent_facts(tmp_path: Path) -> None:
    lat = latency_check(policy(1.5), latency_file(tmp_path, 9.0, 16.2))
    result = {"decision": REVIEW, "policy": "p", "baseline": "b", "alpha": 0.05,
              "n_fail_checks": 4, "fail_ci_level": 0.9875, "reasons": [], "improvements": [],
              "confirmed_drops_within_tolerance": [], "underpowered": [], "results": [],
              "latency": lat}  # fmt: skip
    md = to_markdown(result, "yolo11s")
    assert "**Latency**" in md and "Tesla T4" in md and "1.80" in md
    facts = _latency_facts(lat)
    assert facts is not None
    assert facts["ratio"] == 1.8 and facts["candidate_ms"] == 16.2
