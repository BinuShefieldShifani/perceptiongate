# Where agents help in this pipeline, and where they don't

Written after building it, from what actually happened on real runs.

## Agents do
- **Explain results.** Turning a 19-row gate table into three short paragraphs for a
  reviewer is useful work, and it is checked: every number must exist in a fact sheet
  built by code (ADR 0007).
- **Say what probably caused a change.** The config diff is in the facts, so "the epochs
  went from 50 to 25" comes for free and is grounded.

## Agents don't
- **Decide whether a model ships.** The gate is deterministic code under a reviewed
  policy (ADR 0001, ADR 0006). The decision line in every comment is written by code, and
  the check that the explanation doesn't contradict it is also code.
- **Compute metrics.** mAP, recall, intervals and the paired comparison are NumPy,
  verified against pycocotools. An LLM restating a number is the risky part, so it never
  produces one.
- **Set thresholds.** Thresholds came from measured noise (simulation, then a real
  seed-to-seed run). An agent could suggest a change, but changing one is a reviewed PR.
- **Start GPU runs on its own.** A person adds the `train` label. Runs cost quota and
  time; that budget belongs to a person.

## What the real runs taught
- A 14B model, given correct facts, still inverted a meaning once ("drops within
  tolerance are likely noise") and once put a correct number next to the wrong metric
  (an allowed drop of 1.0 written for a check whose limit is 1.5). The first is now caught
  by a targeted check. The second is not: number grounding can't see which metric a number
  belongs to. That is why the full gate table always sits under the explanation, and why
  the comment says which parts were written by code.
- The checker itself was wrong twice on the first real run (too strict). Both real answers
  became regression tests. An agent's safety net needs the same testing as the agent.
