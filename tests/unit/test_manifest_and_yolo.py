from pathlib import Path

import yaml
from typer.testing import CliRunner

from perceptiongate.cli import app
from perceptiongate.data.manifest import verify_manifest
from perceptiongate.data.schema import Box
from perceptiongate.data.yolo import to_yolo_line

runner = CliRunner()


def build(cfg: Path, train: Path, val: Path) -> Path:
    result = runner.invoke(
        app,
        [
            "data",
            "build-manifest",
            "--config",
            str(cfg),
            "--train-labels",
            str(train),
            "--val-labels",
            str(val),
        ],
    )
    assert result.exit_code == 0, result.output
    return cfg.parent / "manifest"


def test_build_and_verify_manifest(
    small_config: Path, train_labels: Path, val_labels: Path
) -> None:
    manifest = build(small_config, train_labels, val_labels)
    assert verify_manifest(manifest) == []
    assert (manifest / "validation_report.json").is_file()
    assert len((manifest / "train.csv").read_text().splitlines()) == 31  # header + 30


def test_build_is_reproducible(small_config: Path, train_labels: Path, val_labels: Path) -> None:
    m = build(small_config, train_labels, val_labels)
    first = (m / "manifest.json").read_text()
    build(small_config, train_labels, val_labels)
    assert (m / "manifest.json").read_text() == first


def test_hand_edit_is_detected(small_config: Path, train_labels: Path, val_labels: Path) -> None:
    manifest = build(small_config, train_labels, val_labels)
    csv = manifest / "train.csv"
    csv.write_text(csv.read_text() + "extra.jpg,train,night,clear,highway,1,0,0\n")
    problems = verify_manifest(manifest)
    assert any("hash mismatch" in p for p in problems)
    result = runner.invoke(app, ["data", "verify-manifest", str(manifest)])
    assert result.exit_code == 1


def test_yolo_line_normalises_and_clips() -> None:
    line = to_yolo_line(Box("car", -10, 0, 640, 360), 3, 1280, 720)
    assert line == "3 0.250000 0.250000 0.500000 0.500000"
    assert to_yolo_line(Box("car", 2000, 0, 2100, 10), 0, 1280, 720) is None


def test_materialize_writes_yolo_dataset(
    tmp_path: Path, small_config: Path, train_labels: Path, val_labels: Path
) -> None:
    manifest = build(small_config, train_labels, val_labels)
    img_dirs = {}
    for split in ("train", "val"):
        d = tmp_path / f"raw_{split}"
        d.mkdir()
        for line in (manifest / f"{split}.csv").read_text().splitlines()[1:]:
            (d / line.split(",")[0]).write_bytes(b"fake")
        img_dirs[split] = d
    out = tmp_path / "yolo"
    result = runner.invoke(
        app,
        [
            "data",
            "materialize",
            "--config",
            str(small_config),
            "--train-labels",
            str(train_labels),
            "--val-labels",
            str(val_labels),
            "--train-images",
            str(img_dirs["train"]),
            "--val-images",
            str(img_dirs["val"]),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    ds = yaml.safe_load((out / "dataset.yaml").read_text())
    assert ds["names"][0] == "car"
    labels = list((out / "labels" / "train").glob("*.txt"))
    assert len(labels) == 30
    assert labels[0].read_text().count("\n") == 2  # car + pedestrian
