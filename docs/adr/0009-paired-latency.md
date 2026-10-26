# ADR 0009: Latency is measured in pairs, in the same session

Status: accepted

## Context
YOLO11s passed the gate with 18 of 19 checks improved. It also has about 3.5x the
parameters of the baseline, and the gate knew nothing about speed. "Better" is only a
decision once you know what it costs.

Latency numbers from different machines are not comparable, and Kaggle jobs land on
different GPUs (T4, P100). A baseline latency measured once and stored would be compared
against candidates measured on other hardware.

## Decision
- Every PR job attaches the baseline model (Kaggle dataset
  `perceptiongate-baseline-bdd100k-v1`) and benchmarks baseline and candidate one after
  the other, in the same session, on the same GPU: `pgate benchmark` -> latency.json.
- Two numbers per model: GPU (Ultralytics predict, batch 1, FP16, per-image time with
  pre- and post-processing, median of 200 after warm-up) and ONNX Runtime on CPU (median
  of 30), as a rough stand-in for a target without a large GPU.
- The gate compares medians within that file only. Policy: candidate GPU latency may be
  at most 1.5x the baseline's; beyond that the result is NEEDS_REVIEW, not FAIL. The latency
  budget belongs to the deployment target, which the gate doesn't know.
- If the pair wasn't measured together, the latency check is skipped and says so.
- A benchmark failure is recorded and doesn't stop the gate. ONNX failing doesn't stop
  the GPU measurement.

## Consequences
About two extra minutes per run. The comment can now say "better by X, slower by Y", and
a person makes the trade-off. Real hardware targets (TensorRT, an embedded SoC) would need
their own benchmark step; this one only says how the two models compare on the same box.
