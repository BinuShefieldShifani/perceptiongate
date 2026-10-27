# ADR 0012: JSON files are the record; MLflow server and Langfuse are optional

Status: accepted

## Context
The plan listed DagsHub (hosted MLflow) and Langfuse (LLM tracing). After building the
pipeline, the question is what each would add on top of what is already there.

## What exists
- Every run writes run_info.json (commit, config hash, manifest hashes, versions, GPU),
  metrics.json, tracking.json (all epochs) and a local MLflow SQLite store, and every
  PR job uploads them as an Actions artifact, linked from the PR.
- Every LLM call writes a trace: model, prompt size, tokens, seconds, the reply, which
  numbers were rejected and why, retries and fallbacks (agent_trace.json,
  failure_trace.json), plus agent_eval.json for the agent regression cases.
- The dashboard compares runs, slices, latency and agent scores across all of that.

## Decision
- **MLflow server (DagsHub or other): supported, not wired into PR jobs.** Setting
  `MLFLOW_TRACKING_URI` sends every run to any MLflow server with no code change. I did
  not put tracking-server credentials into the Kaggle jobs: it adds a secret to every job
  for a view the artifacts and the dashboard already give. A team with many people and
  hundreds of runs would want the server; one person with a handful of runs doesn't.
- **Langfuse: not added.** The traces already hold what Langfuse would show for one or
  two calls per run. It earns its place with many calls per day, several agents, or a
  need to search and annotate traces across runs. The trace files are plain JSON with
  stable fields, so exporting them later (to Langfuse or OpenTelemetry) is a converter,
  not a redesign.

## Consequences
Nothing depends on a service being up: a run's record travels with the run. The cost is
no central, searchable history across machines; that is the point at which to add them.
