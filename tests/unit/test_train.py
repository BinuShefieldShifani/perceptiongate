"""Training logic tested with a fake model, so these run without Ultralytics or a GPU."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from perceptiongate.cli import app
from perceptiongate.config import load_config
from perceptiongate.observability.tracking import JsonTracker
from perceptiongate.training.train import PreflightError, preflight, read_history, train

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "experiments" / "smoke.yaml"
runner = CliRunner()


class FakeModel:
    def __init__(self, weights: str, save_dir: Path) -> None:
        self.weights = weights
        self.save_dir = save_dir
        self.train_kwargs: dict[str, Any] = {}

    def train(self, **kwargs: Any) -> None:
        self.train_kwargs = kwargs
        (self.save_dir / "weights").mkdir(parents=True)
        (self.save_dir / "weights" / "best.pt").write_bytes(b"weights")
        (self.save_dir / "results.csv").write_text(
            "  epoch,  train/box_loss,  metrics/mAP50(B)\n  1,  2.0,  0.10\n  2,  1.5,  0.20\n"
        )
        self.trainer = SimpleNamespace(save_dir=str(self.save_dir))

    def val(self, **kwargs: Any) -> Any:
        box = SimpleNamespace(
            map=0.25, map50=0.45, mp=0.6, mr=0.5,
            ap_class_index=[0, 1], ap=[0.3, 0.2], ap50=[0.5, 0.4], p=[0.7, 0.5], r=[0.6, 0.4],
        )  # fmt: skip
        return SimpleNamespace(box=box)


def dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "ds" / "dataset.yaml"
    ds.parent.mkdir()
    ds.write_text("names: {0: car}\n")
    return ds


def test_train_writes_metrics_lineage_and_tracking(tmp_path: Path) -> None:
    cfg = load_config(SMOKE)
    fake = FakeModel("x", tmp_path / "out" / "ultralytics")
    tracker = JsonTracker(tmp_path / "out" / "tracking.json")
    result = train(
        cfg, dataset(tmp_path), tmp_path / "out",
        epochs=2, check_manifest=False, model_factory=lambda w: fake, tracker=tracker,
    )  # fmt: skip

    assert fake.train_kwargs["epochs"] == 2
    assert fake.train_kwargs["deterministic"] is True
    assert fake.train_kwargs["seed"] == cfg.train.seed
    assert result.weights.read_bytes() == b"weights"
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["overall"]["map50"] == 0.45
    assert metrics["per_class"]["pedestrian"]["recall"] == 0.4
    info = json.loads((tmp_path / "out" / "run_info.json").read_text())
    assert info["data_verified"] is False
    assert info["epochs_override"] is True
    tracking = json.loads((tmp_path / "out" / "tracking.json").read_text())
    assert tracking["status"] == "finished"
    assert tracking["history"][1]["metrics/mAP50(B)"] == 0.20


def test_failed_training_is_marked_failed(tmp_path: Path) -> None:
    class Broken(FakeModel):
        def train(self, **kwargs: Any) -> None:
            raise RuntimeError("CUDA out of memory")

    tracker = JsonTracker(tmp_path / "out" / "tracking.json")
    with pytest.raises(RuntimeError):
        train(
            load_config(SMOKE), dataset(tmp_path), tmp_path / "out", check_manifest=False,
            model_factory=lambda w: Broken(w, tmp_path), tracker=tracker,
        )  # fmt: skip
    assert json.loads((tmp_path / "out" / "tracking.json").read_text())["status"] == "failed"


def test_preflight_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match="not found"):
        preflight(load_config(SMOKE), tmp_path / "nope.yaml", check_manifest=False)


def test_preflight_rejects_dataset_not_built_from_manifest(
    tmp_path: Path, small_config: Path, train_labels: Path, val_labels: Path
) -> None:
    result = runner.invoke(
        app,
        ["data", "build-manifest", "--config", str(small_config),
         "--train-labels", str(train_labels), "--val-labels", str(val_labels)],
    )  # fmt: skip
    assert result.exit_code == 0
    with pytest.raises(PreflightError, match="materialize"):
        preflight(load_config(small_config), dataset(tmp_path), check_manifest=True)


def test_cli_preflight_failure_exit_code(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["train", "--config", str(SMOKE), "--dataset-yaml", str(tmp_path / "x.yaml"),
         "--out", str(tmp_path / "o")],
    )  # fmt: skip
    assert result.exit_code == 2


def test_read_history_strips_padded_headers(tmp_path: Path) -> None:
    csv = tmp_path / "results.csv"
    csv.write_text("  epoch,  metrics/mAP50(B)\n  1,  0.5\n")
    assert read_history(csv) == [{"epoch": 1.0, "metrics/mAP50(B)": 0.5}]
    assert read_history(tmp_path / "missing.csv") == []
