"""Latency of the baseline and the candidate, measured in the same session on the same GPU.

Latency from different machines can't be compared: Kaggle hands out T4s and P100s. So each
run benchmarks both models side by side and stores both numbers in one latency.json.
The gate then compares within that file only.

Two measurements per model:
  * gpu: Ultralytics predict, batch 1, FP16, per-image time including pre- and
    post-processing (what a single camera stream would see)
  * onnx_cpu: the model exported to ONNX and run with ONNX Runtime on CPU, a rough
    stand-in for a deployment target without a big GPU
"""

from __future__ import annotations

import contextlib
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ModelFactory = Callable[[str], Any]


def _summary(ms: list[float]) -> dict[str, float]:
    s = sorted(ms)
    return {
        "median_ms": round(statistics.median(s), 2),
        "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 2),
        "runs": len(s),
    }


def _default_factory(weights: str) -> Any:
    from ultralytics import YOLO

    return YOLO(weights)


def gpu_latency(
    model: Any, images: list[Path], image_size: int, runs: int, warmup: int
) -> dict[str, Any]:
    device = "0"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    half = device != "cpu"
    for i in range(warmup):
        model.predict(
            str(images[i % len(images)]), imgsz=image_size, device=device, half=half, verbose=False
        )
    total, infer = [], []
    for i in range(runs):
        r = model.predict(
            str(images[i % len(images)]), imgsz=image_size, device=device, half=half, verbose=False
        )[0]
        total.append(float(sum(r.speed.values())))
        infer.append(float(r.speed.get("inference", 0.0)))
    # Per-image total is what one camera stream sees, but at batch 1 it is dominated by
    # fixed pre/post-processing. Inference alone shows the model's own cost.
    return {
        **_summary(total),
        "inference_median_ms": _summary(infer)["median_ms"],
        "device": device,
        "half": half,
    }


def onnx_cpu_latency(model: Any, image_size: int, runs: int, warmup: int) -> dict[str, Any]:
    import onnxruntime as ort

    path = Path(
        model.export(format="onnx", imgsz=image_size, dynamic=False, simplify=False, verbose=False)
    )
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.default_rng(0).random((1, 3, image_size, image_size), dtype=np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    ms = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {name: x})
        ms.append((time.perf_counter() - t) * 1000)
    return {**_summary(ms), "onnx_mb": round(path.stat().st_size / 1e6, 1)}


def benchmark_model(
    weights: Path,
    images: list[Path],
    image_size: int,
    gpu_runs: int = 200,
    cpu_runs: int = 30,
    model_factory: ModelFactory | None = None,
    with_onnx: bool = True,
) -> dict[str, Any]:
    # ONNX export writes next to the weights. Kaggle inputs are read-only, so work on a
    # writable copy (this is what made the baseline's CPU number fail on the first run).
    import shutil
    import tempfile

    local = Path(tempfile.mkdtemp(prefix="bench-")) / Path(weights).name
    if Path(weights).is_file():
        shutil.copy(weights, local)
    else:
        local = Path(weights)
    model = (model_factory or _default_factory)(str(local))
    out: dict[str, Any] = {"weights": str(weights)}
    with contextlib.suppress(AttributeError):
        out["params_m"] = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    out["gpu"] = gpu_latency(model, images, image_size, gpu_runs, warmup=20)
    if with_onnx:
        try:
            out["onnx_cpu"] = onnx_cpu_latency(model, image_size, cpu_runs, warmup=3)
        except Exception as exc:  # noqa: BLE001 - ONNX is a bonus measurement
            out["onnx_cpu_error"] = f"{type(exc).__name__}: {exc}"
    return out


def benchmark_pair(
    baseline: Path,
    candidate: Path,
    images: list[Path],
    image_size: int,
    model_factory: ModelFactory | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    from perceptiongate.observability.run_info import gpu_name

    # Alternate the order would be fairer for thermal effects; with warm-up and 200 runs
    # each, baseline-then-candidate is stable enough on a T4 and keeps the code simple.
    return {
        "same_session": True,
        "gpu": gpu_name(),
        "image_size": image_size,
        "baseline": benchmark_model(
            baseline, images, image_size, model_factory=model_factory, **kwargs
        ),
        "candidate": benchmark_model(
            candidate, images, image_size, model_factory=model_factory, **kwargs
        ),
    }
