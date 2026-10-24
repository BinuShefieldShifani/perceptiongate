# ADR 0005: Slice metrics are computed by our code, checked against COCO

Status: accepted

## Context
Ultralytics reports overall and per-class mAP. The baseline showed why that is not enough:
overall mAP50 was 0.42 while motorcycle recall was 0.18. We need the same numbers per
condition (night, rain, highway) and per object group (vulnerable road users, small
objects), with uncertainty, so the gate can tell a real regression from noise.

## Decision
- Predictions are made once per model at low confidence (0.001) and matched to ground
  truth per image, COCO style, at IoU 0.50:0.95. The result is saved as flat arrays
  (`eval_arrays.npz`).
- Every metric for every slice is computed from those arrays with NumPy. Because
  matching only depends on one image, any subset of images can be scored by summing and
  sorting, which makes bootstrap intervals cheap.
- Intervals are 95% percentile bootstrap over images: 100 resamples for mAP, 1000 for
  recall-type metrics. Slices under 30 images get no interval.
- Recall and precision are reported at an operating point (confidence 0.25, IoU 0.5),
  because that is what a deployed detector actually uses. mAP is threshold-free.
- Scoring needs no GPU and no model: `pgate score` rebuilds a report from saved arrays.

## Verification
On five random datasets of 200 images, our mAP50 and mAP50-95 equal pycocotools'
results to floating-point precision (differences below 1e-16).

## Consequences
Our mAP matches COCO's definition, so it can differ slightly from Ultralytics' own
validator, which interpolates differently. Within this project, all comparisons use our
numbers only.
