# ADR 0003: Tracking is JSON-first; MLflow is best effort

Status: accepted

## Context
Training runs on Kaggle and is orchestrated from GitHub Actions. A tracking server can be
down, rate limited or misconfigured. Losing a two-hour GPU run because a dashboard was
unreachable is a bad trade.

## Decision
- Every run writes tracking.json, metrics.json and run_info.json into its output folder.
  These files are the contract: the evaluation, gate and agents read them.
- MLflow receives the same data when available (`MLFLOW_TRACKING_URI`, e.g. DagsHub, or a
  local file store). Any MLflow error is logged as a warning in tracking.json and the run
  continues.
- Ultralytics' own integrations (MLflow, W&B, ...) are switched off so each run is logged
  exactly once, in our format.

## Consequences
Pipeline steps never depend on a server. The MLflow UI is for people; the JSON files are
for machines.
