"""Experiment tracking.

Every run always writes tracking.json next to its outputs. That file is what later
pipeline steps (CI, the gate, the agents) read, so they never depend on a server.
If MLflow is installed, the same calls also go to MLflow:
  * MLFLOW_TRACKING_URI set   -> that server (e.g. DagsHub)
  * not set                   -> a local file store inside the run folder
Set PGATE_TRACKING=json to turn MLflow off.

Tracking must never kill a training run. If MLflow fails (server down, bad credentials),
we print a warning, record it in tracking.json, and carry on with JSON only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

EXPERIMENT = "perceptiongate"


class Tracker(Protocol):
    def start(self, run_name: str, tags: dict[str, str]) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def end(self, status: str) -> None: ...


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


class JsonTracker:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {
            "run_name": None,
            "status": "running",
            "tags": {},
            "params": {},
            "metrics": {},
            "history": [],
            "artifacts": [],
        }

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str) + "\n")

    def start(self, run_name: str, tags: dict[str, str]) -> None:
        self.data["run_name"] = run_name
        self.data["tags"].update(tags)
        self._flush()

    def log_params(self, params: dict[str, Any]) -> None:
        self.data["params"].update(flatten(params))
        self._flush()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if step is None:
            self.data["metrics"].update(metrics)
        else:
            self.data["history"].append({"step": step, **metrics})
        self._flush()

    def log_artifact(self, path: Path) -> None:
        self.data["artifacts"].append(str(path))
        self._flush()

    def end(self, status: str) -> None:
        self.data["status"] = status
        self._flush()


class MlflowTracker:
    def __init__(self, local_dir: Path) -> None:
        import mlflow

        self.mlflow = mlflow
        uri = os.environ.get("MLFLOW_TRACKING_URI") or (local_dir / "mlruns").resolve().as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(EXPERIMENT)
        self.uri = uri

    def start(self, run_name: str, tags: dict[str, str]) -> None:
        self.mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: dict[str, Any]) -> None:
        flat = {k: str(v)[:500] for k, v in flatten(params).items()}
        self.mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k.replace("(", "_").replace(")", ""): v for k, v in metrics.items()}
        self.mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: Path) -> None:
        self.mlflow.log_artifact(str(path))

    def end(self, status: str) -> None:
        self.mlflow.end_run(status="FINISHED" if status == "finished" else "FAILED")


class MultiTracker:
    """JSON tracker is primary and must succeed; the others are best effort."""

    def __init__(self, primary: JsonTracker, others: list[Tracker]) -> None:
        self.primary = primary
        self.others = others

    def _each(self, method: str, *args: Any) -> None:
        getattr(self.primary, method)(*args)
        for t in list(self.others):
            try:
                getattr(t, method)(*args)
            except Exception as exc:  # noqa: BLE001 - tracking must not stop training
                msg = f"{type(t).__name__}.{method} failed, continuing without it: {exc}"
                print(f"WARNING: {msg}", flush=True)
                self.primary.data.setdefault("tracking_warnings", []).append(msg)
                self.others.remove(t)

    def start(self, run_name: str, tags: dict[str, str]) -> None:
        self._each("start", run_name, tags)

    def log_params(self, params: dict[str, Any]) -> None:
        self._each("log_params", params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._each("log_metrics", metrics, step)

    def log_artifact(self, path: Path) -> None:
        self._each("log_artifact", path)

    def end(self, status: str) -> None:
        self._each("end", status)


def make_tracker(run_dir: Path) -> MultiTracker:
    primary = JsonTracker(run_dir / "tracking.json")
    others: list[Tracker] = []
    if os.environ.get("PGATE_TRACKING", "auto") != "json":
        try:
            others.append(MlflowTracker(run_dir))
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: MLflow unavailable, using JSON tracking only: {exc}", flush=True)
            primary.data["tracking_warnings"] = [f"mlflow init failed: {exc}"]
    return MultiTracker(primary, others)
