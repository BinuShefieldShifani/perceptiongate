# ADR 0001: The release gate is deterministic; agents advise

Status: accepted

## Context
The pipeline uses LLM agents to summarise evaluation results and propose follow-up
experiments. An LLM can misread numbers or be confidently wrong. In a safety-related
domain like driver assistance, the decision to promote a model must be reproducible and
auditable.

## Decision
- Metrics are computed by code. The pass/fail decision comes from a versioned YAML policy
  evaluated by plain Python.
- Agents read the metrics and the gate result. They explain, compare and suggest. They do
  not change the gate result.
- Every number an agent writes is checked against the metrics file before it is posted.
- Any experiment an agent proposes is opened as a draft PR. A human decides whether it runs.

## Consequences
The same inputs always give the same gate result. Agent output can be wrong without
changing what ships. The cost is that agents cannot catch problems the policy does not
encode; when that happens, the fix is a policy change reviewed like code.
