import json
from pathlib import Path
from typing import Any

import pytest

from perceptiongate.dashboard.data import baseline, discover, overview, vru_heatmap

ROOT = Path(__file__).resolve().parents[2]


def slice_report(name: str, vru: float) -> dict[str, Any]:
    def m(v: float) -> dict[str, Any]:
        return {"value": v, "ci95": [v - 0.01, v + 0.01]}

    return {"experiment": name, "slices": {
        "all": {"vru_recall": m(vru), "recall": m(0.6), "small_recall": m(0.4), "map50": m(0.45)},
        "highway": {"vru_recall": m(vru - 0.1)},
    }}  # fmt: skip


def make_run(root: Path, folder: str, name: str, decision: str, vru: float,
             latency: Any = None) -> Path:  # fmt: skip
    ev = root / folder / "result" / "eval"
    ev.mkdir(parents=True)
    results = [
        {"rule": "vru-recall", "slice": "all", "candidate": vru, "delta": vru - 0.366},
        {"rule": "overall-recall", "slice": "all", "candidate": 0.6, "delta": 0.037},
    ]
    gate = {"decision": decision, "results": results, "latency": latency}
    (ev / "gate_result.json").write_text(json.dumps(gate))
    (ev / "gate_result.md").write_text(f"## Gate: {decision}\n")
    (ev / "slice_report.json").write_text(json.dumps(slice_report(name, vru)))
    (ev.parent / "run_info.json").write_text(json.dumps({"git_commit": "abcdef1234567"}))
    return ev


def test_discover_reads_runs_and_old_formats(tmp_path: Path) -> None:
    make_run(
        tmp_path,
        "yolo11s",
        "yolo11s",
        "PASS",
        0.448,
        latency={"target": "gpu", "status": "ok", "ratio": 1.03},
    )  # old single-rule format
    ev = make_run(tmp_path, "final", "yolo11s", "NEEDS_REVIEW", 0.45,
                  latency=[{"id": "cpu-onnx", "target": "onnx_cpu", "status": "regression",
                            "ratio": 2.7}])  # fmt: skip
    (ev / "agent_eval").mkdir()
    (ev / "agent_eval" / "agent_eval.json").write_text(json.dumps({"model": "m", "rows": []}))
    runs = discover(tmp_path)
    assert len(runs) == 2
    by_run = {r["run"]: r for r in runs}
    final = by_run["yolo11s · final"]
    assert final["decision"] == "NEEDS_REVIEW"
    assert final["latency cpu-onnx (x)"] == 2.7
    assert final["Δ VRU recall (pt)"] == 8.4
    assert final["commit"] == "abcdef1"
    assert final["agent_eval"]["model"] == "m"
    assert by_run["yolo11s · yolo11s"]["latency gpu (x)"] == 1.03
    assert by_run["yolo11s · yolo11s"]["failure"] is None


def test_overview_and_heatmap_include_the_baseline(tmp_path: Path) -> None:
    make_run(tmp_path, "a", "seed7", "PASS", 0.364)
    bdir = tmp_path / "base"
    bdir.mkdir()
    (bdir / "slice_report.json").write_text(json.dumps(slice_report("baseline", 0.366)))
    base = baseline(bdir)
    rows = overview(discover(tmp_path), base)
    assert rows[-1]["run"] == "baseline" and rows[-1]["decision"] == "reference"
    assert all(not isinstance(v, (dict, list)) for r in rows for v in r.values())
    heat = vru_heatmap(discover(tmp_path), base)
    assert list(heat) == ["baseline", "seed7 · a"]
    assert heat["baseline"]["highway"] == pytest.approx(0.266)


def test_app_renders_with_real_streamlit(tmp_path: Path) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    make_run(tmp_path, "a", "short-25ep", "FAIL", 0.340)
    app = testing.AppTest.from_file(str(ROOT / "src" / "perceptiongate" / "dashboard" / "app.py"))
    import sys

    old = sys.argv
    sys.argv = ["app.py", "--runs", str(tmp_path), "--baseline", str(tmp_path / "missing")]
    try:
        app.run(timeout=60)
    finally:
        sys.argv = old
    assert not app.exception, app.exception
    assert app.title[0].value == "PerceptionGate"
    assert any("short-25ep" in str(df.value) for df in app.dataframe)
