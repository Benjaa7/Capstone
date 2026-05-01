"""ALNS destroy operators.

Each operator removes ``k`` requests from a :class:`Solution` and returns
the (now partial) solution together with the list of removed passenger ids.
The repair operators (in :mod:`src.alns.repair`) then re-insert these
passengers somewhere — possibly into a different route.

All operators are pure functions ``(sol, k, rng, eval) -> (sol', removed)``
and do not mutate the input solution.

Catalogue (matching Table 1 of ``informe/chapters/metodologia.tex``):

* ``random_removal``        — diversification baseline.
* ``worst_removal``         — removes requests with the highest insertion cost.
* ``related_removal``       — Shaw-style: similar in distance + time + category.
* ``load_violation_removal``— targets capacity-excess regions.
* ``time_based_removal``    — targets ride-time / advance violations.
* ``category_violation_removal`` (novel) — targets sindical incompatibilities.
* ``route_removal``         — empties the smallest active route entirely.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.alns.evaluation import evaluate_route
from src.alns.solution import Route, Solution
from src.data.instance import Instance


# Type alias for an operator
DestroyFn = Callable[[Solution, int, np.random.Generator, Instance], tuple[Solution, list[int]]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _route_passengers(route: Route, n: int) -> list[int]:
    return [node for node in route.nodes if 1 <= node <= n]


def _remove_passengers_from_solution(sol: Solution, pids: list[int], n: int) -> Solution:
    """Return a *new* :class:`Solution` with ``pids`` removed from every route.

    Empty routes (no remaining passengers) are dropped.
    """
    pid_set = set(pids)
    new_routes: list[Route] = []
    for route in sol.routes:
        kept = [
            node
            for node in route.nodes
            if node not in pid_set and (node - n) not in pid_set
        ]
        # Drop the route if it has no passengers left.
        passengers_left = [node for node in kept if 1 <= node <= n]
        if not passengers_left:
            continue
        new_routes.append(
            Route(
                vehicle_id=route.vehicle_id,
                vehicle_type=route.vehicle_type,
                nodes=kept,
                start_times={k: v for k, v in route.start_times.items() if k in kept},
                loads={k: v for k, v in route.loads.items() if k in kept},
            )
        )
    new_sol = Solution(
        instance_label=sol.instance_label,
        n_passengers=sol.n_passengers,
        routes=new_routes,
        total_cost=sol.total_cost,
        fixed_cost=sol.fixed_cost,
        variable_cost=sol.variable_cost,
        is_feasible=False,
        metadata=dict(sol.metadata),
    )
    return new_sol


def _all_passengers(sol: Solution, n: int) -> list[int]:
    out: list[int] = []
    for route in sol.routes:
        out.extend(_route_passengers(route, n))
    return out


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def random_removal(
    sol: Solution, k: int, rng: np.random.Generator, instance: Instance
) -> tuple[Solution, list[int]]:
    """Remove ``k`` passengers chosen uniformly at random."""
    n = sol.n_passengers
    pool = _all_passengers(sol, n)
    if not pool:
        return sol, []
    k = min(k, len(pool))
    chosen = list(rng.choice(pool, size=k, replace=False))
    chosen = [int(x) for x in chosen]
    return _remove_passengers_from_solution(sol, chosen, n), chosen


def worst_removal(
    sol: Solution, k: int, rng: np.random.Generator, instance: Instance
) -> tuple[Solution, list[int]]:
    """Remove ``k`` passengers whose round-trip cost is highest in their route.

    For each passenger we compute its detour cost: the additional distance
    contributed by the (pickup, delivery) pair beyond a straight-line skip.
    Passengers with the largest detour are removed first; mild stochasticity
    is added by sampling among the top candidates.
    """
    n = sol.n_passengers
    cost_per_meter = float(instance.others["cost_per_meter"])
    pax = instance.passengers.set_index("id")
    candidates: list[tuple[float, int]] = []

    for route in sol.routes:
        # Compute per-passenger detour cost as: distance from prev pickup-node
        # to current pickup-node + distance from current delivery-node to next.
        nodes = route.nodes
        for idx, node in enumerate(nodes):
            if not (1 <= node <= n):
                continue
            pid = node
            prev = nodes[idx - 1] if idx > 0 else None
            nxt = nodes[idx + 1] if idx + 1 < len(nodes) else None
            cost = 0.0
            if prev is not None and prev != 0 and prev != 2 * n + 1:
                p_lat, p_lon = _coord_for_node(instance, prev, n)
                c_lat, c_lon = _coord_for_node(instance, node, n)
                cost += cost_per_meter * instance.distance((p_lat, p_lon), (c_lat, c_lon))
            if nxt is not None and nxt != 0 and nxt != 2 * n + 1:
                c_lat, c_lon = _coord_for_node(instance, node, n)
                nx_lat, nx_lon = _coord_for_node(instance, nxt, n)
                cost += cost_per_meter * instance.distance((c_lat, c_lon), (nx_lat, nx_lon))
            candidates.append((cost, pid))

    if not candidates:
        return sol, []

    # Sort descending by detour cost
    candidates.sort(key=lambda x: x[0], reverse=True)
    k = min(k, len(candidates))
    # Random pick from top-2k to inject diversification (Shaw-style).
    top = candidates[: min(2 * k, len(candidates))]
    chosen_idx = rng.choice(len(top), size=k, replace=False)
    chosen = [int(top[i][1]) for i in chosen_idx]
    return _remove_passengers_from_solution(sol, chosen, n), chosen


def related_removal(
    sol: Solution, k: int, rng: np.random.Generator, instance: Instance
) -> tuple[Solution, list[int]]:
    """Shaw removal: remove a seed plus the most similar requests.

    Similarity is a weighted sum of normalized distance, time-window proximity
    and same-category bonus, following Ropke & Pisinger (2006).
    """
    n = sol.n_passengers
    pool = _all_passengers(sol, n)
    if not pool:
        return sol, []
    k = min(k, len(pool))
    pax = instance.passengers.set_index("id")

    seed_pid = int(rng.choice(pool))
    seed_row = pax.loc[seed_pid]
    seed_pcoord = (float(seed_row["pickup_lat"]), float(seed_row["pickup_lon"]))
    seed_dcoord = (float(seed_row["delivery_lat"]), float(seed_row["delivery_lon"]))
    seed_e, _seed_sigma = instance.effective_pickup_window(seed_pid)
    seed_cat = int(seed_row["priority"])

    scores: list[tuple[float, int]] = []
    for pid in pool:
        if pid == seed_pid:
            continue
        row = pax.loc[pid]
        pcoord = (float(row["pickup_lat"]), float(row["pickup_lon"]))
        dcoord = (float(row["delivery_lat"]), float(row["delivery_lon"]))
        e_pid, _sigma = instance.effective_pickup_window(int(pid))
        cat = int(row["priority"])

        d_pp = instance.distance(seed_pcoord, pcoord)
        d_dd = instance.distance(seed_dcoord, dcoord)
        time_diff = abs(seed_e - e_pid)
        cat_diff = abs(seed_cat - cat)

        # Normalised weighted sum (lower = more similar).
        score = 0.5 * (d_pp + d_dd) / 1000.0 + 0.3 * time_diff / 600.0 + 0.2 * cat_diff
        scores.append((score, pid))

    scores.sort(key=lambda x: x[0])
    chosen = [seed_pid, *(int(pid) for _s, pid in scores[: k - 1])]
    return _remove_passengers_from_solution(sol, chosen, n), chosen


def load_violation_removal(
    sol: Solution, k: int, rng: np.random.Generator, instance: Instance
) -> tuple[Solution, list[int]]:
    """Target the routes whose load profile exceeds capacity. If no route
    has a capacity violation, falls back to :func:`worst_removal`."""
    n = sol.n_passengers
    bad_pids: list[int] = []
    for route in sol.routes:
        ev = evaluate_route(route, instance)
        if ev.violations.q <= 1e-6:
            continue
        bad_pids.extend(_route_passengers(route, n))
    if not bad_pids:
        return worst_removal(sol, k, rng, instance)
    k = min(k, len(bad_pids))
    chosen_idx = rng.choice(len(bad_pids), size=k, replace=False)
    chosen = [int(bad_pids[i]) for i in chosen_idx]
    return _remove_passengers_from_solution(sol, chosen, n), chosen


def time_based_removal(
    sol: Solution, k: int, rng: np.random.Generator, instance: Instance
) -> tuple[Solution, list[int]]:
    """Target the routes whose ride-time / advance violations are highest.

    Falls back to :func:`worst_removal` if no time violation exists.
    """
    n = sol.n_passengers
    bad_pids: list[int] = []
    for route in sol.routes:
        ev = evaluate_route(route, instance)
        if ev.violations.r <= 1e-6 and ev.violations.d <= 1e-6 and ev.violations.t <= 1e-6:
            continue
        bad_pids.extend(_route_passengers(route, n))
    if not bad_pids:
        return worst_removal(sol, k, rng, instance)
    k = min(k, len(bad_pids))
    chosen_idx = rng.choice(len(bad_pids), size=k, replace=False)
    chosen = [int(bad_pids[i]) for i in chosen_idx]
    return _remove_passengers_from_solution(sol, chosen, n), chosen


def category_violation_removal(
    sol: Solution, k: int, rng: np.random.Generator, instance: Instance
) -> tuple[Solution, list[int]]:
    """Novel operator: remove passengers from routes that violate the
    sindical "two consecutive categories" rule. Falls back to
    :func:`worst_removal` if all routes are sindical-clean."""
    n = sol.n_passengers
    pax = instance.passengers.set_index("id")
    bad_pids: list[int] = []
    for route in sol.routes:
        cats = sorted(
            {
                int(pax.loc[node, "priority"])
                for node in route.nodes
                if 1 <= node <= n
            }
        )
        violates = (
            len(cats) > 2
            or any(abs(a - b) > 1 for a, b in zip(cats, cats[1:], strict=False))
        )
        if not violates:
            continue
        # Prefer to remove the minority category(ies) to bring the route to compliance.
        cat_count: dict[int, int] = {}
        for node in route.nodes:
            if 1 <= node <= n:
                c = int(pax.loc[node, "priority"])
                cat_count[c] = cat_count.get(c, 0) + 1
        # Sort categories by count ascending — minority first.
        sorted_cats = sorted(cat_count.items(), key=lambda kv: kv[1])
        # Remove passengers from the top minority categories until at most 2 consecutive remain.
        for cat, _cnt in sorted_cats[:-2] if len(sorted_cats) > 2 else sorted_cats[:1]:
            for node in route.nodes:
                if 1 <= node <= n and int(pax.loc[node, "priority"]) == cat:
                    bad_pids.append(int(node))

    if not bad_pids:
        return worst_removal(sol, k, rng, instance)
    bad_pids = list(dict.fromkeys(bad_pids))  # dedup preserving order
    k = min(k, len(bad_pids))
    chosen = bad_pids[:k]
    if len(bad_pids) > k:
        # Random subset for diversification
        chosen_idx = rng.choice(len(bad_pids), size=k, replace=False)
        chosen = [int(bad_pids[i]) for i in chosen_idx]
    return _remove_passengers_from_solution(sol, chosen, n), chosen


def route_removal(
    sol: Solution, k: int, rng: np.random.Generator, instance: Instance
) -> tuple[Solution, list[int]]:
    """Empty the route with the lowest occupancy entirely.

    This attacks the fixed cost ``F^{t(k)} y_k``, since a route with one or
    two passengers is often cheaper to redistribute than to keep open.
    """
    n = sol.n_passengers
    if not sol.routes:
        return sol, []
    target_route = min(sol.routes, key=lambda r: len([x for x in r.nodes if 1 <= x <= n]))
    chosen = _route_passengers(target_route, n)
    if not chosen:
        return sol, []
    if k < len(chosen):
        # Sample a subset (rare; usually we want to empty the whole route)
        chosen_idx = rng.choice(len(chosen), size=k, replace=False)
        chosen = [int(chosen[i]) for i in chosen_idx]
    return _remove_passengers_from_solution(sol, chosen, n), chosen


# ---------------------------------------------------------------------------
# Coord helper
# ---------------------------------------------------------------------------
def _coord_for_node(instance: Instance, node: int, n: int) -> tuple[float, float]:
    if node == 0 or node == 2 * n + 1:
        return (0.0, 0.0)
    pax = instance.passengers.set_index("id")
    if 1 <= node <= n:
        row = pax.loc[node]
        return float(row["pickup_lat"]), float(row["pickup_lon"])
    pid = node - n
    row = pax.loc[pid]
    return float(row["delivery_lat"]), float(row["delivery_lon"])


# Registry
DESTROY_OPERATORS: dict[str, DestroyFn] = {
    "random": random_removal,
    "worst": worst_removal,
    "related": related_removal,
    "load_violation": load_violation_removal,
    "time_based": time_based_removal,
    "category_violation": category_violation_removal,
    "route": route_removal,
}
