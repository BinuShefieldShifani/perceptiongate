# ADR 0007: Analyst agent with a grounding check

Status: accepted

## Context
The gate produces a correct decision and a 19-row table. Reviewers need the short
version: what changed, what got worse, whether it matters, what to do. That is a good
job for an LLM, and a risky one: LLMs restate numbers wrongly and compute new ones.

## Decision
- Code builds a fact sheet (`agent_facts.json`) from gate_result.json, both slice
  reports and the config diff: the decision, blocking and review checks, confirmed drops
  within tolerance, too-noisy checks, the largest per-class recall changes. Deltas are
  in percentage points.
- One LLM call writes the explanation in three fixed sections. One call, because small
  models handle single calls reliably and drop multi-step tool chains.
- Grounding check on the reply: every number must equal a fact value at the text's
  precision (0.021 may appear as 2.1 points, 0.563 as 56.3%); the text must state the
  gate decision and must not call a FAIL mergeable.
- On failure, one retry that names the exact numbers that were rejected. If that also
  fails, or the LLM is unreachable, the comment uses a summary written by code and says
  so. The pipeline never blocks on the LLM.
- The comment header (the decision) and the full gate table always come from code.
- Every attempt is logged to `agent_trace.json`: model, latency, tokens, the reply,
  which numbers were checked and which were rejected.
- Providers: Claude API, any OpenAI-compatible endpoint (Ollama, Groq, vLLM), or none.
  Standard-library HTTP only, no SDK dependency.

## Tested against real output
On the short-25ep gate result, a faithful explanation passed with 14 of 14 numbers
verified. A version adding "a relative loss of about 5.4%" (a plausible number the
model computed itself) was rejected. "Safe to merge" on a FAIL was rejected.

## First real run: what the trace showed
qwen2.5:14b on Kaggle's T4, explaining the seed7 PASS. Both of its answers were
rejected and the comment fell back to the code summary. The trace showed why:
- Answer 2 was correct (9 of 9 numbers verified) and was rejected by two checker bugs:
  it quoted the "200 boxes" threshold from the prompt, which was not in the facts, and
  it wrote "passed the gate", which the check did not accept as stating PASS.
- Answer 1 had a real error that the check missed: it called the confirmed drops
  within tolerance "likely due to noise". Every number was right; the meaning was not.

Changes:
- Thresholds the prompt mentions are in the fact sheet.
- The decision line is written by code. The check now asks the opposite question: does
  the text claim a different decision for the gate, the candidate or the model?
- A targeted meaning check: when the facts list confirmed drops, the text must not
  dismiss them as noise.
- The model is warmed up after pulling. The first call took 104 s, the retry 12 s;
  the difference was loading 9 GB into GPU memory, not generating text.
Both real answers are kept as regression tests (tests/unit/test_agent_regressions.py).

## Known limit
The check proves each number comes from the facts, not that it sits next to the right
metric. Writing the interval end (-2.4) as the recall change (-2.1) passes, because -2.4
is a fact. Closing this needs structured output (the model fills slots that code
renders), which would also make the text stiffer. For now the full gate table under the
explanation is the reference, and the limit is covered by a test so it stays visible.
