"""Run a local Ollama server for the analyst agent (Kaggle, CI, or a laptop).

    ensure_ollama(["qwen2.5:14b"])

Installs Ollama if it's missing (Linux, using the official install script), starts
`ollama serve` if nothing answers on the port, waits until it is healthy, and pulls the
models. Each step is skipped when it is already done, so calling it twice is cheap.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

DEFAULT_URL = "http://localhost:11434"
INSTALL_CMD = "curl -fsSL https://ollama.com/install.sh | sh"

Runner = Callable[..., subprocess.CompletedProcess]  # type: ignore[type-arg]


def server_version(url: str = DEFAULT_URL, timeout: float = 2.0) -> str | None:
    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=timeout) as resp:  # noqa: S310
            return str(json.loads(resp.read()).get("version"))
    except (OSError, ValueError):
        return None


def wait_for_server(url: str = DEFAULT_URL, seconds: float = 60.0) -> str:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if version := server_version(url):
            return version
        time.sleep(1)
    raise RuntimeError(f"Ollama did not answer on {url} within {seconds:.0f}s")


def install(run: Runner = subprocess.run) -> None:
    if shutil.which("ollama"):
        return
    if shutil.which("zstd") is None and shutil.which("apt-get"):
        # Recent Ollama release archives are zstd-compressed.
        run(["apt-get", "install", "-y", "-q", "zstd"], check=False, capture_output=True)
    print("Installing Ollama...", flush=True)
    run(INSTALL_CMD, shell=True, check=True)  # noqa: S602 - fixed official command


def start(log_path: Path, url: str = DEFAULT_URL) -> str:
    if version := server_version(url):
        return version
    log = log_path.open("a")
    subprocess.Popen(["ollama", "serve"], stdout=log, stderr=subprocess.STDOUT)  # noqa: S603,S607
    return wait_for_server(url)


def pull(model: str, run: Runner = subprocess.run) -> float:
    start_t = time.monotonic()
    print(f"Pulling {model}...", flush=True)
    run(["ollama", "pull", model], check=True)
    return round(time.monotonic() - start_t, 1)


def ensure_ollama(models: list[str], log_path: Path = Path("ollama.log")) -> dict[str, object]:
    """Install, start and pull. Returns timings for the run log."""
    t0 = time.monotonic()
    install()
    version = start(log_path)
    pulls = {m: pull(m) for m in models}
    return {
        "version": version,
        "pull_seconds": pulls,
        "setup_seconds": round(time.monotonic() - t0, 1),
    }
