"""Deterministic stratified sampling.

Plain random sampling of BDD100K gives very few foggy or snowy images, so the model and
the evaluation would barely see them. Here every stratum (e.g. night+rainy) first gets up
to `min_per_stratum` images, and the rest is shared in proportion to stratum size.
The same inputs and seed always give the same selection.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from perceptiongate.data.schema import Frame

Stratum = tuple[str, ...]


def stratum_of(frame: Frame, keys: Sequence[str]) -> Stratum:
    return tuple(frame.attr(k) for k in keys)


def allocate(sizes: dict[Stratum, int], n: int, floor: int) -> dict[Stratum, int]:
    """Decide how many items to take from each stratum."""
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    alloc = dict.fromkeys(sizes, 0)
    order = sorted(sizes)

    # Phase 1: guarantee a floor per stratum (round-robin if floors exceed n).
    remaining = n
    level = 0
    while remaining > 0 and level < floor:
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if alloc[key] < sizes[key] and alloc[key] <= level:
                alloc[key] += 1
                remaining -= 1
                progressed = True
        level += 1
        if not progressed:
            break

    # Phase 2: share the rest in proportion to the spare capacity of each stratum.
    while remaining > 0:
        spare = {k: sizes[k] - alloc[k] for k in order if sizes[k] > alloc[k]}
        spare_total = sum(spare.values())
        shares = {k: (remaining * s) // spare_total for k, s in spare.items()}
        if sum(shares.values()) == 0:
            for key in sorted(spare, key=lambda k: (-spare[k], k)):
                if remaining == 0:
                    break
                alloc[key] += 1
                remaining -= 1
            continue
        for key, share in shares.items():
            take = min(share, spare[key])
            alloc[key] += take
            remaining -= take
    return alloc


def stratified_sample(
    frames: Sequence[Frame],
    n: int,
    keys: Sequence[str],
    min_per_stratum: int,
    seed: int,
    require_objects: bool = True,
) -> list[Frame]:
    pool = [f for f in frames if f.boxes] if require_objects else list(frames)
    groups: dict[Stratum, list[Frame]] = defaultdict(list)
    for frame in sorted(pool, key=lambda f: f.name):
        groups[stratum_of(frame, keys)].append(frame)

    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])

    alloc = allocate({k: len(v) for k, v in groups.items()}, n, min_per_stratum)
    chosen = [f for key in sorted(groups) for f in groups[key][: alloc[key]]]
    return sorted(chosen, key=lambda f: f.name)
