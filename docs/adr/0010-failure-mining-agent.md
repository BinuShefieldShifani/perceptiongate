# ADR 0010: Failure-mining agent proposes, code validates, a person runs

Status: accepted

## Context
The gate says whether a candidate is better; it doesn't say where it still fails or what
to try next. The baseline report pointed at highway VRU recall, and YOLO11s improved
almost everything except that slice. The next experiment should come from where the
misses actually are.

## Decision
- Code mines the saved predictions (no images, no extra GPU): every vulnerable-road-user
  box in val, found or missed at the operating point, grouped by size (geometric mean
  side in original pixels), time of day, weather, scene and class, for candidate and
  baseline. Groups under 30 boxes are reported but not used for conclusions.
- An LLM names the main pattern from those numbers (grounding check as in ADR 0007) and
  proposes one change from a fixed menu: backbone size, epochs, learning rate, optimizer,
  input size. Input size is flagged as needing a new prepared dataset.
- Code validates the proposal against the menu and the config schema, and writes it as
  `proposed_config.yaml`. An invalid or missing proposal falls back to a rule: if most
  missed VRU boxes are small, propose a larger input; else a larger backbone; else more
  epochs.
- The result is a section appended to the PR comment, with the proposed config ready to
  commit. A person opens it as a PR and adds the `train` label. The agent never starts a
  run: GPU time is a budget a person owns.

## Completeness check (analyst)
The first YOLO11s explanation passed the grounding check by containing no numbers at all.
The fact sheet now lists the headline changes (blocking failures, or the two largest
changes on blocking checks), and the explanation must state each of them, or it is sent
back like an invented number would be.

## Consequences
Metadata-based mining finds patterns in size and conditions, not in appearance (glare,
occlusion, unusual poses). Looking at the missed crops with a vision model is the next
step, and the hardest one to ground.
