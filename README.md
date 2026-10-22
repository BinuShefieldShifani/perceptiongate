# PerceptionGate

CI/CD for a driving-perception object detector. Every change is trained, evaluated by
slice (night, rain, small objects, vulnerable road users), checked against a release
policy, and explained by agents that also propose the next experiment.

> Status: step 1, project skeleton and CI. Built in public, one step at a time.

## Scope, honestly
One public dataset subset (BDD100K), a small YOLO model, free GPU time on Kaggle. The
point is the workflow around the model, not the model's absolute accuracy.

## Design principles
- Configs are typed and validated before any GPU time is spent.
- The release gate is deterministic code. Agents advise, they don't decide.
  See [ADR 0001](docs/adr/0001-deterministic-gate.md).

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
