# ADR 0006: Paired, noise-aware release gate

Status: accepted (thresholds provisional until the seed7 run)

## Context
The baseline slice report showed two things that shape the gate:
- mAP intervals are wide (night: 0.38 to 0.50), because mAP averages over classes and
  some are rare (102 motorcycles, 124 riders in val). Blocking on mAP would be noise.
- The safety-relevant numbers are recall of vulnerable road users (VRU) and of small
  objects, and they are measured on far fewer boxes than overall recall.

## Decision
- The candidate and the baseline are compared on the same 2,000 val images with a
  paired bootstrap of the difference. Hard and easy images move both models together,
  so the paired interval is much narrower than two separate intervals.
- A result is a **regression** only if the drop exceeds `max_drop` AND the interval is
  entirely below zero. It is a **possible regression** if the drop exceeds `max_drop` but
  the interval includes zero.
- Four blocking ("fail") checks: overall recall, VRU recall, pedestrian recall,
  small-object recall. Their family-wise false-alarm rate is held at 5% with a
  Bonferroni correction (each uses a 98.75% interval).
- Per-condition checks and mAP are "review" checks: only a confirmed regression sends
  the candidate to a human; a possible one is shown but does not escalate.
- Every result reports its noise floor (interval half-width). Checks whose noise floor
  exceeds their `max_drop` are listed as "too noisy to decide on".
- The gate refuses to compare evaluations made on different images or at different
  operating points (exit code 2).

## How the thresholds were chosen
Simulation at the real scale (2,000 images, ~18 objects each, class mix and recalls
like the baseline). Seed-to-seed noise, as the smallest drop a 95% test can catch:

| Check | Detectable drop |
|---|---:|
| overall recall | 0.7 pt |
| per-condition recall | ~1.1 pt |
| small-object recall | 1.3 pt |
| VRU recall | 2.3 pt |
| pedestrian recall | 2.5 pt |
| VRU recall per condition | 4-5 pt |

Each `max_drop` sits just above its noise.

The first policy (1-point VRU threshold, no correction) failed or flagged 9 of 10
equal-quality candidates. After the change:

| True VRU drop | Gate result (independent baseline/candidate pairs) |
|---|---|
| none | 40 of 40 PASS |
| 3 pt | FAIL 5 of 10 |
| 5 pt | FAIL 8 of 10, review 1 |
| 8 pt | FAIL 10 of 10 |

## Lesson: baseline luck
Against one fixed baseline, 6 of 60 equal-quality candidates still failed. That baseline
had come out 1.5 points high on pedestrians by chance, so everything compared with it
looked worse. With an independent baseline per pair, false alarms went to 0 of 40.
In practice: check the baseline against a second seed (configs/experiments/seed7.yaml).
If they differ by more than the noise above, compare candidates against the mean of
several seeds instead of a single run.

## Consequences
With this val set, VRU drops under ~3 points are mostly invisible. Detecting a 1-point
VRU change would need roughly 5x more VRU boxes. That is a data decision, not a code one.
