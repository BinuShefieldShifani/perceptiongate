"""Lineage for every run: which code, which config, which data, which machine."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from perceptiongate import __version__
from perceptiongate.config import ExperimentConfig


def config_hash(cfg: ExperimentConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:16]


def git_commit() -> str:
    if sha := os.environ.get("GITHUB_SHA") or os.environ.get("PGATE_GIT_SHA"):
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except ImportError:
        pass
    return "none"


def collect_run_info(cfg: ExperimentConfig, manifest_dir: Path | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "experiment": cfg.name,
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        "perceptiongate": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _version("torch"),
        "ultralytics": _version("ultralytics"),
        "gpu": gpu_name(),
        "manifest": None,
    }
    if manifest_dir is not None and (manifest_dir / "manifest.json").is_file():
        summary = json.loads((manifest_dir / "manifest.json").read_text())
        info["manifest"] = {
            "path": str(manifest_dir),
            "train_sha256": summary["splits"]["train"]["sha256"],
            "val_sha256": summary["splits"]["val"]["sha256"],
        }
    return info
