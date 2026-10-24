from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from perceptiongate.config import load_config

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "configs" / "experiments" / "baseline.yaml"


def write_variant(tmp_path: Path, change: Any) -> Path:
    """Load the baseline config, apply a change to the dict, save it as a new file.
    Editing the parsed YAML (not the text) keeps these tests valid when values change."""
    raw = yaml.safe_load(BASELINE.read_text())
    change(raw)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_baseline_config_loads() -> None:
    cfg = load_config(BASELINE)
    assert cfg.name == "baseline"
    assert "pedestrian" in cfg.data.classes


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = write_variant(tmp_path, lambda r: r["train"].update({"epoch": 5}))
    with pytest.raises(ValidationError):
        load_config(path)


def test_negative_epochs_rejected(tmp_path: Path) -> None:
    path = write_variant(tmp_path, lambda r: r["train"].update({"epochs": -1}))
    with pytest.raises(ValidationError):
        load_config(path)


def test_duplicate_classes_rejected(tmp_path: Path) -> None:
    path = write_variant(
        tmp_path, lambda r: r["data"].update({"classes": ["car", "car", *r["data"]["classes"][2:]]})
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_auto_optimizer_is_rejected(tmp_path: Path) -> None:
    path = write_variant(tmp_path, lambda r: r["train"].update({"optimizer": "auto"}))
    with pytest.raises(ValidationError):
        load_config(path)


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")
