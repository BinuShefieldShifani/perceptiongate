from pathlib import Path

from conftest import write_labels
from perceptiongate.data.inspect import summarize
from perceptiongate.data.schema import load_frames


def test_legacy_and_new_layouts_normalise_the_same(tmp_path: Path) -> None:
    old = load_frames(write_labels(tmp_path / "old.json", 3, legacy=True))
    new = load_frames(write_labels(tmp_path / "new.json", 3, legacy=False))
    assert [b.category for b in old[0].boxes] == ["car", "pedestrian"]
    assert old == new


def test_non_box_labels_are_ignored(tmp_path: Path) -> None:
    frames = load_frames(write_labels(tmp_path / "l.json", 1))
    assert len(frames[0].boxes) == 2  # lanes and drivable area dropped


def test_summary_counts(tmp_path: Path) -> None:
    s = summarize(load_frames(write_labels(tmp_path / "l.json", 9)))
    assert s["frames"] == 9
    assert s["categories"] == {"car": 9, "pedestrian": 9}
    assert set(s["timeofday"]) == {"daytime", "night", "dawn/dusk"}
