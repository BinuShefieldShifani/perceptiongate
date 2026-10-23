from pathlib import Path

import yaml

from perceptiongate.data.synthetic import make_synthetic
from perceptiongate.data.yolo import index_images


def test_index_finds_images_in_subfolders(tmp_path: Path) -> None:
    (tmp_path / "trainA").mkdir()
    (tmp_path / "trainB").mkdir()
    (tmp_path / "trainA" / "a.jpg").write_bytes(b"x")
    (tmp_path / "trainB" / "b.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not an image")
    index = index_images(tmp_path)
    assert set(index) == {"a.jpg", "b.jpg"}
    assert index["b.jpg"].parent.name == "trainB"


def test_synthetic_dataset_layout(tmp_path: Path) -> None:
    ds = make_synthetic(tmp_path, ["car", "pedestrian"], n_train=4, n_val=2, size=32)
    assert yaml.safe_load(ds.read_text())["names"] == {0: "car", 1: "pedestrian"}
    pngs = sorted((tmp_path / "images" / "train").glob("*.png"))
    assert len(pngs) == 4
    assert pngs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    for label in (tmp_path / "labels" / "train").glob("*.txt"):
        for line in label.read_text().splitlines():
            cls, *coords = line.split()
            assert cls in {"0", "1"}
            assert all(0.0 <= float(c) <= 1.0 for c in coords)
