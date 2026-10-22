# PerceptionGate

CI/CD for a driving-perception object detector. Every change is trained, evaluated by
slice (night, rain, small objects, vulnerable road users), checked against a release
policy, and explained by agents that also propose the next experiment.

> Status: step 2, dataset pipeline. Built in public, one step at a time.

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
