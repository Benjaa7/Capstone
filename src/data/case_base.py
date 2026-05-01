"""Case Base instance builder.

Selects a stratified random subsample of passengers from a full
:class:`Instance` so that the MILP can be solved in reasonable time as an
oracle for the ALNS. The fleet (Common + Large) is preserved per the
professor's instruction; only ``time_dependent`` may be turned off.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from src.data.instance import Instance


def _stratified_targets(counts: pd.Series, n: int) -> dict[int, int]:
    """Distribute ``n`` slots across categories proportional to ``counts``.

    Uses Hamilton's largest-remainder method to keep the totals exact and the
    distribution closest to the original.
    """
    total = counts.sum()
    raw = {cat: n * c / total for cat, c in counts.items()}
    floor = {cat: int(math.floor(v)) for cat, v in raw.items()}
    remainder = n - sum(floor.values())
    # Sort categories by largest fractional part and add 1 to the top
    # ``remainder`` of them.
    fracs = sorted(raw.items(), key=lambda kv: kv[1] - floor[kv[0]], reverse=True)
    for cat, _ in fracs[:remainder]:
        floor[cat] += 1
    return floor


def select_case_base(
    instance: Instance,
    n: int = 30,
    seed: int = 42,
) -> Instance:
    """Return a new :class:`Instance` with a stratified subsample of ``n`` passengers.

    The strata are the four union priorities (1..4); the sample preserves the
    relative frequencies as closely as possible. The vehicle fleet, union
    constraints and travel-time matrices are kept intact (the MILP simply has
    fewer pickup/delivery nodes).
    """
    rng = np.random.default_rng(seed)
    counts = instance.category_counts()
    targets = _stratified_targets(counts, n)

    chosen_ids: list[int] = []
    for cat, k in targets.items():
        pool = instance.passengers.loc[instance.passengers["priority"] == cat, "id"].to_numpy()
        if k > len(pool):
            raise ValueError(
                f"Cannot draw {k} passengers from category {cat} (pool size {len(pool)})."
            )
        sampled = rng.choice(pool, size=k, replace=False)
        chosen_ids.extend(int(x) for x in sampled)

    sub_pass = (
        instance.passengers[instance.passengers["id"].isin(chosen_ids)]
        .copy()
        .reset_index(drop=True)
    )
    # Renumber so the subset has compact ids 1..n (helpful for the MILP).
    sub_pass["original_id"] = sub_pass["id"]
    sub_pass["id"] = np.arange(1, len(sub_pass) + 1)

    return replace(
        instance,
        label=f"{instance.label}_caso_base_n{n}",
        passengers=sub_pass,
        # Reset cached lookups to avoid carrying state from the parent.
        _time_lookup={},
        _avg_lookup={},
    )
