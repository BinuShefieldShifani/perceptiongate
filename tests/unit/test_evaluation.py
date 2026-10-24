import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from perceptiongate.cli import app
from perceptiongate.data.manifest import ManifestRow, read_rows
from perceptiongate.evaluation.matching import ImageData, MatchedArrays, box_iou, match_images
from perceptiongate.evaluation.metrics import (
    Index,
    average_precision,
    bootstrap,
    class_ap,
    map_values,
)
from perceptiongate.evaluation.report import build_report, to_markdown

CLASSES = ["car", "pedestrian"]
runner = CliRunner()


def img(
    name: str,
    gt: list[list[float]],
    gc: list[int],
    det: list[list[float]],
    dc: list[int],
    ds: list[float],
) -> ImageData:
    return ImageData(
        name,
        np.array(gt, float).reshape(-1, 4),
        np.array(gc, int),
        np.array(det, float).reshape(-1, 4),
        np.array(dc, int),
        np.array(ds, float),
    )


BOX = [0.1, 0.1, 0.3, 0.3]


def test_box_iou() -> None:
    iou = box_iou(np.array([BOX]), np.array([BOX, [0.2, 0.1, 0.4, 0.3], [0.5, 0.5, 0.6, 0.6]]))
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(1 / 3)
    assert iou[0, 2] == 0.0


def test_perfect_detection_gives_ap_one() -> None:
    arr = match_images([img("a", [BOX], [0], [BOX], [0], [0.9])])
    ap = class_ap(Index(arr, 2), np.array([0]))
    assert ap[0] == pytest.approx(np.ones(10))
    assert np.isnan(ap[1, 0])  # no pedestrians: excluded, not zero
    assert map_values(ap) == (pytest.approx(1.0), pytest.approx(1.0))


def test_duplicate_detection_is_a_false_positive() -> None:
    arr = match_images([img("a", [BOX], [0], [BOX, BOX], [0, 0], [0.9, 0.8])])
    assert arr.det_tp[:, 0].tolist() == [True, False]


def test_wrong_class_does_not_match() -> None:
    arr = match_images([img("a", [BOX], [0], [BOX], [1], [0.9])])
    assert not arr.det_tp.any()
    assert not arr.gt_hit.any()


def test_iou_thresholds_and_operating_point() -> None:
    # IoU 1/3 with the GT: below 0.5, so a miss at every threshold.
    arr = match_images([img("a", [BOX], [0], [[0.2, 0.1, 0.4, 0.3]], [0], [0.9])])
    assert not arr.det_tp.any()
    # Correct box but low confidence: counts for mAP, not for operating-point recall.
    arr = match_images([img("a", [BOX], [0], [BOX], [0], [0.1])], op_conf=0.25)
    assert arr.det_tp[0, 0] and not arr.gt_hit[0]


def test_average_precision_known_value() -> None:
    # 2 GT; ranked TP, FP, TP -> precision envelope 1.0 up to recall 0.5, 2/3 after.
    ap = average_precision(np.array([0.9, 0.8, 0.7]), np.array([[True], [False], [True]]), 2)
    expected = (51 * 1.0 + 50 * (2 / 3)) / 101
    assert ap[0] == pytest.approx(expected)
    assert average_precision(np.zeros(0), np.zeros((0, 10), bool), 3).tolist() == [0.0] * 10


def test_bootstrap_is_deterministic_and_brackets_value() -> None:
    values = np.arange(100, dtype=float)

    def stat(s: np.ndarray) -> float:
        return float(values[s].mean())

    a = bootstrap(stat, np.arange(100), 200, seed=3)
    assert a == bootstrap(stat, np.arange(100), 200, seed=3)
    assert a[0] < 49.5 < a[1]


def scene(n: int) -> tuple[list[ImageData], dict[str, ManifestRow]]:
    images, rows = [], {}
    for i in range(n):
        night = i % 2 == 1
        ped = [0.5, 0.5, 0.52, 0.53]  # tiny: 0.02*0.03*1280*720 = 553 px^2 < 32^2
        # Day: both found. Night: the pedestrian is missed.
        det = [BOX] if night else [BOX, ped]
        images.append(
            img(f"{i}.jpg", [BOX, ped], [0, 1], det, [0, 1][: len(det)], [0.9, 0.8][: len(det)])
        )
        tod = "night" if night else "daytime"
        rows[f"{i}.jpg"] = ManifestRow(f"{i}.jpg", "val", tod, "clear", "highway", 2, 1, 1)
    return images, rows


def test_report_finds_the_night_pedestrian_problem() -> None:
    images, rows = scene(80)
    report = build_report(
        match_images(images), rows, CLASSES, 1280, 720, n_boot_ap=20, n_boot_count=200
    )
    day, night = report["slices"]["day"], report["slices"]["night"]
    assert day["vru_recall"]["value"] == 1.0
    assert night["vru_recall"]["value"] == 0.0
    assert night["small_recall"]["value"] == 0.0
    assert night["per_class"]["pedestrian"]["recall"] == 0.0
    assert night["per_class"]["car"]["recall"] == 1.0
    assert report["slices"]["all"]["n_images"] == 80
    lo, hi = report["slices"]["all"]["vru_recall"]["ci95"]
    assert lo <= 0.5 <= hi
    assert "rain" not in report["slices"]  # no rainy images: slice omitted
    md = to_markdown(report)
    assert "| night | 40 |" in md


def test_small_slices_get_no_intervals() -> None:
    images, rows = scene(10)
    report = build_report(
        match_images(images), rows, CLASSES, 1280, 720, n_boot_ap=5, n_boot_count=5
    )
    assert report["slices"]["night"]["recall"]["ci95"] == [None, None]
    assert "note" in report["slices"]["night"]


def test_arrays_round_trip(tmp_path: Path) -> None:
    arr = match_images(scene(6)[0])
    arr.save(str(tmp_path / "a.npz"))
    back = MatchedArrays.load(str(tmp_path / "a.npz"))
    assert back.names == arr.names
    assert np.array_equal(back.det_tp, arr.det_tp)
    assert back.op_conf == arr.op_conf


class FakeModel:
    """Predicts the GT box for every label line, except pedestrians (class 1)."""

    def __init__(self, labels_dir: Path) -> None:
        self.labels_dir = labels_dir

    def predict(self, source: list[str], **kwargs: Any) -> Any:
        for path in source:
            rows = np.loadtxt(self.labels_dir / f"{Path(path).stem}.txt", ndmin=2)
            rows = rows[rows[:, 0] == 0]
            cx, cy, w, h = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
            xyxyn = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            yield SimpleNamespace(
                boxes=SimpleNamespace(xyxyn=xyxyn, cls=rows[:, 0], conf=np.full(len(rows), 0.9))
            )


def test_evaluate_and_score_end_to_end(
    tmp_path: Path, small_config: Path, train_labels: Path, val_labels: Path
) -> None:
    from perceptiongate.config import load_config
    from perceptiongate.data.build import load_clean
    from perceptiongate.data.yolo import to_yolo_line
    from perceptiongate.evaluation.run import evaluate

    args = ["--config", str(small_config), "--train-labels", str(train_labels),
            "--val-labels", str(val_labels)]  # fmt: skip
    assert runner.invoke(app, ["data", "build-manifest", *args]).exit_code == 0
    cfg = load_config(small_config)
    ds = tmp_path / "ds"
    (ds / "images" / "val").mkdir(parents=True)
    (ds / "labels" / "val").mkdir(parents=True)
    frames = {f.name: f for f in load_clean(cfg, val_labels, "val")[0]}
    index = {c: i for i, c in enumerate(cfg.data.classes)}
    for row in read_rows(cfg.data.manifest, "val"):
        stem = Path(row.name).stem
        (ds / "images" / "val" / f"{stem}.jpg").write_bytes(b"x")
        lines = [to_yolo_line(b, index[b.category], 1280, 720) for b in frames[row.name].boxes]
        (ds / "labels" / "val" / f"{stem}.txt").write_text("\n".join(x for x in lines if x) + "\n")

    out = tmp_path / "eval"
    report = evaluate(
        cfg, Path("unused.pt"), ds, out, n_boot_ap=5, n_boot_count=20,
        model_factory=lambda w: FakeModel(ds / "labels" / "val"),
    )  # fmt: skip
    s = report["slices"]["all"]
    assert s["per_class"]["car"]["recall"] == 1.0
    assert s["per_class"]["pedestrian"]["recall"] == 0.0
    assert s["vru_recall"]["value"] == 0.0
    assert (out / "slice_report.md").is_file()

    # Re-score from saved arrays on CPU only: same numbers.
    result = runner.invoke(
        app,
        ["score", "--config", str(small_config), "--arrays", str(out / "eval_arrays.npz"),
         "--out", str(tmp_path / "rescored"), "--ap-resamples", "5"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    rescored = json.loads((tmp_path / "rescored" / "slice_report.json").read_text())
    assert rescored["slices"]["all"]["map50"]["value"] == s["map50"]["value"]
