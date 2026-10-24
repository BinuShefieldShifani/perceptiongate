# PerceptionGate

CI/CD for a driving-perception object detector. Every change is trained, evaluated by
slice (night, rain, small objects, vulnerable road users), checked against a release
policy, and explained by agents that also propose the next experiment.

> Status: step 4, evaluation by slice. Built in public, one step at a time.

## Scope, honestly
One public dataset subset (BDD100K), a small YOLO model, free GPU time on Kaggle. The
point is the workflow around the model, not the model's absolute accuracy.

## Design principles
- Configs are typed and validated before any GPU time is spent.
- The release gate is deterministic code. Agents advise, they don't decide.
  See [ADR 0001](docs/adr/0001-deterministic-gate.md).
- Every dataset version is a manifest in git, verified in CI.
  See [ADR 0002](docs/adr/0002-dataset-manifests-in-git.md).

## Data pipeline
```
raw BDD100K labels (Kaggle)
  -> normalise both label layouts (2018 and 2020 releases)
  -> clean: drop zero-size, out-of-image and tiny boxes; fail if too many are bad
  -> stratified sample by time of day x weather, with a floor for rare conditions
  -> manifest (CSV + hashes) committed to git, verified in CI
  -> materialize: YOLO labels + symlinked images, only on the training machine
```
```bash
pgate data inspect path/to/labels.json
pgate data build-manifest --config configs/experiments/baseline.yaml \
    --train-labels .../train.json --val-labels .../val.json
pgate data verify-manifest manifests/bdd100k-v1
```

### bdd100k-v1 at a glance
Built from the full BDD100K labels (69,863 train and 10,000 val frames, 1.47M boxes).

| | Images | Boxes | Small (<32² px) | Vulnerable road users |
|---|---|---|---|---|
| train | 10,000 | 184,116 | 101,564 (55%) | 15,234 (8%) |
| val | 2,000 | 36,438 | 20,115 (55%) | 2,988 (8%) |

Label cleaning dropped 437 boxes out of 1.47M (0.03%), all under 2 px on a side.
No inverted or out-of-image boxes were found. Over half the boxes are small objects
and fewer than 1 in 10 are pedestrians, riders or cyclists, which is why both get
their own evaluation slice and their own gate rule.

## Training
```
pgate train
  preflight: dataset.yaml exists, manifest verifies, dataset folder was built
             from that exact manifest (source_manifest.json hash)  -> else exit 2
  lineage:   run_info.json  (git commit, config hash, manifest hashes, torch,
             ultralytics, GPU)
  train:     Ultralytics YOLO, fixed seed, deterministic=True
  outputs:   metrics.json (overall + per class), model/best.pt, tracking.json
  tracking:  always tracking.json; MLflow too if installed
             (MLFLOW_TRACKING_URI or a local file store). MLflow failures warn,
             they never stop training.
```
CI runs a real one-epoch training on CPU with synthetic images on every PR
(`smoke-train` job), so broken training code fails in minutes, not on the GPU.

### Getting data to the GPU fast
The first GPU run spent 45 of its 50 minutes reading images from Kaggle's mounted
dataset (0.2 MB/s). With the files cached, one epoch took 1.5 minutes. So images are
prepared once into a single tar and every run unpacks that instead
([ADR 0004](docs/adr/0004-prepared-dataset-artifact.md)):

```bash
# once per manifest version (Kaggle, CPU)
python scripts/kaggle_prepare.py            # -> bdd100k-v1-640.tar, publish as a dataset
# every training run (Kaggle, GPU)
python scripts/kaggle_train.py --prepared /kaggle/input/<your dataset> --epochs 30
```

### Things the first GPU run caught
- `optimizer=auto` made Ultralytics ignore the configured `lr0` and pick its own
  optimizer. The optimizer is now an explicit config field.
- MLflow's file store refused to start. Training carried on with JSON tracking, as
  ADR 0003 intends. Local tracking now uses SQLite.

## Baseline (YOLO11n, 50 epochs, bdd100k-v1)
1.19 GPU hours on a Kaggle T4, about 75 s per epoch once data comes from the prepared
tar. Full numbers in `baselines/bdd100k-v1/`.

| Class | mAP50 | Recall |
|---|---:|---:|
| car | 0.699 | 0.666 |
| truck | 0.502 | 0.480 |
| traffic_sign | 0.491 | 0.467 |
| traffic_light | 0.462 | 0.476 |
| bus | 0.435 | 0.383 |
| pedestrian | 0.433 | 0.402 |
| rider | 0.313 | 0.306 |
| motorcycle | 0.237 | 0.176 |
| bicycle | 0.210 | 0.228 |
| **all** | **0.420** | **0.398** |

Overall mAP50 of 0.42 hides that the model misses about 82% of motorcycles and 6 in 10
pedestrians. That is why evaluation goes per slice and the gate has a separate rule for
vulnerable road users.

## Evaluation by slice
```bash
pgate evaluate --config ... --weights best.pt --dataset-dir <yolo dataset> --out eval/
pgate score    --config ... --arrays eval/eval_arrays.npz --out eval/   # CPU, no model
```
For each slice (all, day, night, dawn_dusk, clear, rain, snow, fog, highway,
city_street): mAP50 and mAP50-95 with 95% bootstrap intervals, recall and precision at
the operating point, vulnerable-road-user recall, small-object recall, and per-class AP
and recall. Output: `slice_report.json` for machines, `slice_report.md` for people.

The mAP code matches pycocotools exactly on random test data
([ADR 0005](docs/adr/0005-own-slice-metrics.md)). Scoring 2,000 images with 600k
detections takes about 1.5 minutes on CPU, including all bootstrap intervals.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
make install
make check
pgate validate-config configs/experiments/baseline.yaml
```

## Layout
```
configs/       experiment and gate configs (YAML)
manifests/     versioned dataset definitions (committed)
src/perceptiongate/
  config.py    typed config loading
  cli.py       `pgate` command line
  data/        subset building and label validation
  training/    training entry point (runs on Kaggle GPU)
  evaluation/  slice metrics, confidence intervals, latency
  gate/        release policy
  agents/      analyst and failure-mining agents
  observability/
scripts/       thin wrappers called from CI and Kaggle
kaggle/        Kaggle kernel metadata
tests/
docs/adr/      architecture decision records
```

## Data license
BDD100K is released by UC Berkeley for non-commercial research and education. This repo
contains only image names and derived counts, never images or original label files.
