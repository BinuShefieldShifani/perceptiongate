import ast
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("kaggle_job", ROOT / "scripts" / "kaggle_job.py")
assert spec and spec.loader
kj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kj)


def test_choose_config_requires_exactly_one_experiment() -> None:
    assert kj.choose_config(["README.md", "configs/experiments/yolo11s.yaml"]) == (
        "configs/experiments/yolo11s.yaml"
    )
    with pytest.raises(SystemExit, match="exactly one"):
        kj.choose_config(["src/perceptiongate/cli.py"])
    with pytest.raises(SystemExit, match="exactly one"):
        kj.choose_config(["configs/experiments/a.yaml", "configs/experiments/b.yaml"])
    with pytest.raises(SystemExit):
        kj.choose_config(["configs/experiments/smoke.yaml"])  # CI-only config


def test_build_writes_a_valid_pinned_kernel(tmp_path: Path) -> None:
    sha = "0123456789abcdef0123456789abcdef01234567"
    out = kj.build("configs/experiments/seed7.yaml", sha, "someone", tmp_path / "job")
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/perceptiongate-run"
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert meta["dataset_sources"] == [kj.DATASET]
    run = (out / "run.py").read_text()
    ast.parse(run)
    assert f"SHA = '{sha}'" in run
    assert "CONFIG = 'configs/experiments/seed7.yaml'" in run
    assert "AGENT_MODEL = 'qwen2.5:14b'" in run


def test_build_rejects_a_non_sha(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        kj.build("configs/experiments/seed7.yaml", "main; rm -rf /", "u", tmp_path)


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ('u/perceptiongate-run has status "KernelWorkerStatus.RUNNING"', "running"),
        ('u/perceptiongate-run has status "complete"', "complete"),
        ('u/perceptiongate-run has status "KernelWorkerStatus.ERROR"', "error"),
        ('has status "cancelAcknowledged"', "error"),
        ("has status queued", "running"),
    ],
)
def test_parse_status(text: str, status: str) -> None:
    assert kj.parse_status(text) == status


def test_summary_reads_comment_and_decision(tmp_path: Path) -> None:
    ev = tmp_path / "result" / "eval"
    ev.mkdir(parents=True)
    (ev / "pr_comment.md").write_text("## PerceptionGate: ❌ FAIL\n")
    (ev / "gate_result.json").write_text(json.dumps({"decision": "FAIL"}))
    s = kj.summary(tmp_path, "x")
    assert s["decision"] == "FAIL"
    assert str(s["body"]).startswith(kj.MARKER)


def test_summary_without_outputs_reports_error(tmp_path: Path) -> None:
    s = kj.summary(tmp_path, "x")
    assert s["decision"] == "ERROR"
    assert "without producing a comment" in str(s["body"])


def test_train_workflow_is_valid_yaml_and_guards_the_label() -> None:
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "train.yml").read_text())
    job = wf["jobs"]["kaggle"]
    assert "train" in job["if"]
    assert wf["concurrency"]["cancel-in-progress"] is False
    # Inputs reach shell steps through env vars only (no ${{ inputs }} inside run:).
    for step in job["steps"]:
        assert "${{ inputs" not in step.get("run", "")
