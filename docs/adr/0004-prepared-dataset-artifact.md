# ADR 0004: Train from a prepared dataset tar, not the raw mount

Status: accepted

## Context
The first 3-epoch GPU run on Kaggle took 50 minutes. About 45 of them were disk reads:
Ultralytics measured 0.2 MB/s from the mounted BDD100K dataset (dataset scan 29 min,
epoch 1 14.5 min). Once the files were cached, an epoch took 1.5 minutes. Every new
Kaggle session pays the slow reads again.

## Decision
- `pgate data prepare` reads the manifest's images once, resizes them so the long side
  equals `data.image_size`, and packs everything into one tar
  (`<manifest>-<size>.tar`, with dataset.yaml, labels and source_manifest.json inside).
- The tar is published as a private Kaggle dataset and attached to training runs.
  `pgate data unpack` extracts it to local disk in seconds.
- Training's preflight still compares source_manifest.json against the committed
  manifest, so a tar from an older manifest is refused.

## Why resizing is safe
Ultralytics letterboxes every image to `imgsz` when loading it, so a 1280x720 image is
already shrunk to 640x360 before the model sees it. Pre-resizing does the same work once.
YOLO labels are normalised to 0..1 and don't change.

## Consequences
One extra build step per manifest version, and a second artifact to keep in sync. In
return, each training run starts in seconds instead of ~45 minutes.
