from pathlib import Path

import pytest
from pydantic import ValidationError

from perceptiongate.config import load_config

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "configs" / "experiments" / "baseline.yaml"


def test_baseline_config_loads() -> None:
    cfg = load_config(BASELINE)
    assert cfg.name == "baseline"
    assert "pedestrian" in cfg.data.classes


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(BASELINE.read_text().replace("epochs: 30", "epochs: 30\n  epoch: 5"))
    with pytest.raises(ValidationError):
        load_config(bad)


def test_negative_epochs_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(BASELINE.read_text().replace("epochs: 30", "epochs: -1"))
    with pytest.raises(ValidationError):
        load_config(bad)


def test_duplicate_classes_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(BASELINE.read_text().replace("[car, truck,", "[car, car,"))
    with pytest.raises(ValidationError):
        load_config(bad)


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")
