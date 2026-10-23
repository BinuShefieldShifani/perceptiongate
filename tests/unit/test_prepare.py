"""Prepared dataset round trip with real JPEGs. Needs Pillow (runs in the smoke job)."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from perceptiongate.cli import app
from perceptiongate.config import load_config
from perceptiongate.training.train import PreflightError, preflight

Image = pytest.importorskip("PIL.Image")
runner = CliRunner()


def write_images(manifest: Path, split: str, root: Path) -> None:
    # Put half the images in a subfolder, like the Kaggle mirror's trainA/trainB.
    for i, line in enumerate((manifest / f"{split}.csv").read_text().splitlines()[1:]):
        folder = root / ("A" if i % 2 else "B")
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1280, 720), (90, 120, 150)).save(folder / line.split(",")[0])


def test_prepare_unpack_and_preflight(
    tmp_path: Path, small_config: Path, train_labels: Path, val_labels: Path
) -> None:
    args = ["--config", str(small_config), "--train-labels", str(train_labels),
            "--val-labels", str(val_labels)]  # fmt: skip
    assert runner.invoke(app, ["data", "build-manifest", *args]).exit_code == 0
    manifest = tmp_path / "manifest"
    for split in ("train", "val"):
        write_images(manifest, split, tmp_path / f"raw_{split}")

    result = runner.invoke(
        app,
        ["data", "prepare", *args,
         "--train-images", str(tmp_path / "raw_train"), "--val-images", str(tmp_path / "raw_val"),
         "--out-dir", str(tmp_path / "prepared"), "--workers", "4"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    tar = tmp_path / "prepared" / "manifest-640.tar"
    assert tar.is_file()

    result = runner.invoke(app, ["data", "unpack", "--tar", str(tar), "--out", str(tmp_path / "y")])
    assert result.exit_code == 0, result.output
    ds_yaml = tmp_path / "y" / "manifest-640" / "dataset.yaml"
    ds = yaml.safe_load(ds_yaml.read_text())
    assert "path" not in ds  # relocatable

    jpgs = sorted((ds_yaml.parent / "images" / "train").glob("*.jpg"))
    assert len(jpgs) == 30
    with Image.open(jpgs[0]) as im:
        assert im.size == (640, 360)  # long side = image_size, aspect kept
    label = (ds_yaml.parent / "labels" / "train" / f"{jpgs[0].stem}.txt").read_text()
    assert label.count("\n") == 2
    info = json.loads((ds_yaml.parent / "prepared.json").read_text())
    assert info["images"] == {"train": 30, "val": 18}

    cfg = load_config(small_config)
    preflight(cfg, ds_yaml, check_manifest=True)  # matches the manifest: OK

    # Rebuild the manifest with a different seed: the old tar must now be refused.
    text = small_config.read_text().replace("seed: 42", "seed: 7", 1)
    small_config.write_text(text)
    assert runner.invoke(app, ["data", "build-manifest", *args]).exit_code == 0
    with pytest.raises(PreflightError, match="different manifest"):
        preflight(load_config(small_config), ds_yaml, check_manifest=True)
