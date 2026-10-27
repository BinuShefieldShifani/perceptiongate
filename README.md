# PerceptionGate

CI/CD for a driving-perception object detector. Every change is trained, evaluated by
slice (night, rain, small objects, vulnerable road users), checked against a release
policy, and explained by agents that also propose the next experiment.

> Status: step 9, failure-mining agent and paired latency (GPU and CPU). Built in public, one step at a time.

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

## Baseline by slice
From `baselines/bdd100k-v1/slice_report.md` (operating point: confidence 0.25, IoU 0.5):

| Slice | Images | mAP50 [95% CI] | VRU recall [95% CI] |
|---|---:|---|---|
| all | 2000 | 0.405 [0.391, 0.426] | 0.366 [0.340, 0.391] |
| night | 691 | 0.440 [0.378, 0.503] | 0.357 [0.302, 0.414] |
| day | 918 | 0.406 [0.386, 0.436] | 0.373 [0.342, 0.405] |
| **highway** | 501 | **0.332** [0.283, 0.424] | **0.248** [0.160, 0.340] |

The weakest slice is highway, not night: pedestrians and cyclists there are rare and
usually far away. Night is as good as day.

## Release gate
```bash
pgate gate --candidate outputs/<run>/eval --name <run>     # exit 0 PASS/REVIEW, 1 FAIL
```
Compares a candidate with the baseline on the same images (paired bootstrap), under
`configs/gate/policy.yaml`. Blocking checks: overall, VRU, pedestrian and small-object
recall, with a Bonferroni correction. Per-condition recall and mAP can only ask for
review. Every result shows its noise floor. Thresholds were set from simulated
seed-to-seed noise ([ADR 0006](docs/adr/0006-release-gate.md)):

| True VRU drop | Gate (simulated) |
|---|---|
| none | 40 of 40 PASS |
| 3 pt | FAIL 5 of 10 |
| 5 pt | FAIL 8 of 10 |
| 8 pt | FAIL 10 of 10 |

The first version of the policy flagged 9 of 10 equal-quality models. The val set
can't resolve VRU changes under ~3 points, so the thresholds now say so.

Two calibration runs check the gate on real models:
`configs/experiments/short-25ep.yaml` (half the schedule, should FAIL) and
`configs/experiments/seed7.yaml` (same config, other seed, should PASS).

## Analyst agent
```bash
pgate gate    --candidate outputs/<run>/eval --name <run>
pgate explain --candidate outputs/<run>/eval --name <run> \
              --candidate-config configs/experiments/<run>.yaml
```
Writes `pr_comment.md`: the decision (from the gate), a three-part explanation, and the
full gate table. The explanation comes from an LLM that only sees a fact sheet built by
code, and every number it writes is checked against that sheet. A reply with a number
that isn't there is retried once with the exact numbers named; after that, a code-written
summary is used instead ([ADR 0007](docs/adr/0007-grounded-analyst-agent.md)).

The agent runs on local open-weights models through Ollama: no API keys, and no data
sent to a model provider. On Kaggle it uses the same T4 GPU as training:
```bash
python scripts/kaggle_agent.py --name short-25ep --models qwen2.5:7b,qwen2.5:14b
python scripts/kaggle_train.py --config ... --prepared ... --agent-model qwen2.5:14b
```
`scripts/kaggle_agent.py` installs and starts Ollama, pulls the models, runs the gate
and the agent once per model, and writes a comparison of grounding pass rate and
latency. `PGATE_LLM_TIMEOUT` (default 300 s) covers slow CPU inference. Any
OpenAI-compatible endpoint works the same way (`PGATE_LLM=openai`,
`PGATE_LLM_BASE_URL`); `ANTHROPIC_API_KEY` is supported too. With no LLM, the comment
uses the code-written summary.

First real case, short-25ep (half the training schedule): the gate failed it on overall
recall (−2.1 points) and small-object recall (−2.5). VRU recall also dropped 2.5 points.
That is within the 3-point tolerance, so it passes, but the drop is real, and the
report now shows it as "real drop, within tolerance" instead of a green tick.

## From PR to comment
```
PR changes one config in configs/experiments/  +  label "train"
  -> GitHub Actions pushes a Kaggle GPU job pinned to the PR's commit
  -> Kaggle: train -> evaluate -> gate -> qwen2.5:14b writes the explanation
  -> Actions downloads the outputs, comments on the PR, fails the check on a gate FAIL
```
See [ADR 0008](docs/adr/0008-pr-triggered-kaggle-runs.md). Runs queue one at a time.
What agents are and aren't allowed to do here, and why:
[docs/where-agents-should-not-go.md](docs/where-agents-should-not-go.md).

### Failure mining
After the gate and the explanation, a second agent works out where the candidate still
misses vulnerable road users (by size, condition and class, against the baseline) and
proposes one next experiment from a fixed menu. Code validates the proposal and puts a
ready-to-run config in the PR comment; a person decides whether to run it
([ADR 0010](docs/adr/0010-failure-mining-agent.md)).

### Latency
Every PR job also benchmarks the baseline and the candidate side by side, on the same GPU
in the same session. Two rules: GPU per image and ONNX Runtime on CPU; more than 1.5x
slower than the baseline goes to review, whatever the accuracy
([ADR 0009](docs/adr/0009-paired-latency.md)).

The first YOLO11s run showed why one measure isn't enough: 3.6x the parameters, but only
3% slower per image on the T4 (11.37 -> 11.66 ms), because at batch 1 that time is mostly
pre/post-processing. On CPU, YOLO11s took 144 ms per image.

### Gate calibration on real models
| Candidate | What changed | Overall recall | VRU recall | Gate |
|---|---|---:|---:|---|
| short-25ep | half the training schedule | −2.1 pt | −2.5 pt (real, within tolerance) | ❌ FAIL |
| seed7 | training seed only | −0.3 pt | −0.2 pt | ✅ PASS |

A real change moved recall about 7× more than retraining with another seed, and the
thresholds sit between the two. mAP50 dropped 1.3 points from the seed change alone,
which is why mAP only asks for review. Rare classes (about 100 boxes each) moved 3 to 4
points between seeds, so single-run per-class changes need several seeds before they mean
much.

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
