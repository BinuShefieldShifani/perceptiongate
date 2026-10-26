# ADR 0008: PRs trigger GPU runs on Kaggle; results come back as a PR comment

Status: accepted

## Context
Training needs a GPU; GitHub's free runners have none. Kaggle has free T4/P100 GPUs and
an API that can push a script, run it, and return its outputs.

## Decision
- A PR that changes exactly one file in `configs/experiments/` and gets the label `train`
  starts `.github/workflows/train.yml`. `workflow_dispatch` runs any config by hand.
- The workflow builds a Kaggle script kernel (`scripts/kaggle_job.py build`): GPU and
  internet on, the prepared dataset attached, and a `run.py` that clones the repo at the
  PR's exact commit sha. Lineage in run_info.json therefore points at that commit.
- On Kaggle: train, evaluate, gate, and the analyst agent with a local qwen2.5:14b.
- Actions polls the job, downloads its outputs, uploads them as an artifact, and posts or
  updates one PR comment (found by a hidden marker). The check fails on a gate FAIL.
- One concurrency group (`kaggle-gpu`), no cancelling: runs queue, because there is one
  kernel slug, one GPU and one weekly quota.
- The Kaggle token lives in GitHub secrets. User input reaches shell steps only through
  environment variables, and the commit sha is validated before it is written into run.py.

## Consequences
A run costs the PR author nothing but a label, and about 80 minutes of GPU quota for a
50-epoch model. The Actions job waits the whole time; on a public repo those minutes are
free. A job that crashes still produces a comment saying there was no result, with the
Kaggle log in the artifact.
