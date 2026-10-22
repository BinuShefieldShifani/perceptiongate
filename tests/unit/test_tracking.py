import json
from pathlib import Path

import pytest

from perceptiongate.observability.tracking import JsonTracker, flatten, make_tracker


def test_flatten_nested() -> None:
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_json_tracker_records_everything(tmp_path: Path) -> None:
    t = JsonTracker(tmp_path / "tracking.json")
    t.start("run-1", {"git_commit": "abc"})
    t.log_params({"train": {"epochs": 3}})
    t.log_metrics({"loss": 1.5}, step=1)
    t.log_metrics({"final.map50": 0.4})
    t.end("finished")
    data = json.loads((tmp_path / "tracking.json").read_text())
    assert data["status"] == "finished"
    assert data["params"] == {"train.epochs": 3}
    assert data["history"] == [{"step": 1, "loss": 1.5}]
    assert data["metrics"] == {"final.map50": 0.4}


def test_json_only_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGATE_TRACKING", "json")
    tracker = make_tracker(tmp_path)
    tracker.start("r", {})
    tracker.end("finished")
    assert (tmp_path / "tracking.json").is_file()
    assert not (tmp_path / "mlruns").exists()


def test_broken_secondary_tracker_does_not_stop_the_run(tmp_path: Path) -> None:
    from perceptiongate.observability.tracking import MultiTracker

    class Down:
        def start(self, run_name: str, tags: dict[str, str]) -> None:
            raise ConnectionError("server unreachable")

    primary = JsonTracker(tmp_path / "tracking.json")
    tracker = MultiTracker(primary, [Down()])  # type: ignore[list-item]
    tracker.start("r", {})
    tracker.log_metrics({"m": 1.0})
    tracker.end("finished")
    data = json.loads((tmp_path / "tracking.json").read_text())
    assert data["status"] == "finished"
    assert "server unreachable" in data["tracking_warnings"][0]
