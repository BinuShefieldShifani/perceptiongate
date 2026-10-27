# ADR 0011: Every pipeline run also evaluates the agent

Status: accepted

## Context
The analyst agent is a component like any other, and it changes: prompts get edited,
models get swapped (7B vs 14B), checks get stricter. One comment per PR shows how it did
on one case; it doesn't show whether a change made it worse on the cases that used to
work. The real runs produced three cases with known right answers.

## Decision
- `evals/agent/cases/` holds fact sheets from the three real runs (short-25ep FAIL,
  seed7 PASS on noise, yolo11s PASS with improvements), each with expectations:
  the decision, the headline numbers to state, and keywords for the cause.
- `pgate agent-eval` runs the agent on every case exactly as the pipeline does and
  scores: grounded on the first try, LLM text used (possibly after a retry), cause named.
- Every Kaggle job runs it after writing the PR comment, with the same local model, so
  each pipeline run is also an agent regression test. Results land in the artifact and
  in the dashboard.
- A test checks each case is internally consistent: the code-written summary for the
  case must itself pass every check. A case that no answer could pass is a broken case.

## Consequences
Three cases are a smoke test, not a benchmark. New cases come from real runs that showed
a new failure (as the "noise" answer and the number-free answer did). Temperature 0
makes reruns of the same model repeat themselves, so variation shows up across models
and prompt changes, which is what we want to catch.
