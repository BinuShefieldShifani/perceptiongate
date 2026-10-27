# Agent evaluation cases

Fact sheets from real pipeline runs, with what a correct explanation must do:
grounded numbers, no contradiction of the gate decision, the headline changes stated
(`must_mention`), and the cause named (`cause_keywords`, any one of them).

`pgate agent-eval --cases evals/agent/cases --out <dir>` runs every case with the
configured LLM. Every Kaggle job runs it too, after writing the PR comment, so each
pipeline run also regression-tests the agent. Add a case whenever a real run shows a
new way for the agent to get something wrong.
