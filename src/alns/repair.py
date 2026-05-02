"""ALNS repair operators.

Each operator takes a partial :class:`Solution` (some passengers removed by
a destroy operator) plus the list of removed passenger ids, and re-inserts
all of them. The resulting solution may still be infeasible — the dynamic
penalties of the evaluator and subsequent ALNS iterations are responsible
for converging to feasibility.

Catalogue (Table 1 of ``informe/chapters/metodologia.tex``):

* ``best_insertion``  — for each removed pid, find its lowest-cost insertion.
* ``random_insertion``— random order and random insertion position.
* ``regret2_insertion`` — maximises the regret (1st vs 2nd best position).
* ``regret3_insertion`` — like regret2 but using the 3rd-best gap.
* ``zero_load_insertion`` (Pilati 2025) — prefers slots where the vehicle
  load is currently zero, easing capacity and category feasibility.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.alns.evaluation import evaluate_route, penalised_score
from src.alns.evaluation import WeightTracker
from src.alns.solution import Route, Solution
from src.data.instance import Instance


RepairFn = Callable[
    [Solution, list[int], np.random.Generator, Instance, WeightTracker],
    Solution,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _insert_into_route(route: Route, pid: int, p_pos: int, d_pos: int, n: int) -> Route:
    """Return a clone of ``route`` with pickup at ``p_pos`` and delivery at
    ``d_pos`` (after pickup).

    ``p_pos`` and ``d_pos`` are indices *into the original node list*.
    After inserting the pickup at ``p_pos``, every subsequent index shifts
    by 1, so the delivery must be placed at ``d_pos + 1`` whenever
    ``d_pos >= p_pos`` (which is always true since d_pos >= p_pos by
    construction in the search loops).
    """
    new_nodes = list(route.nodes)
    new_nodes.insert(p_pos, pid)
    # Pickup inserted at p_pos ⟹ all positions >= p_pos shift by 1.
    # Because d_pos >= p_pos always holds, we unconditionally add 1.
    new_nodes.insert(d_pos + 1, pid + n)
    return Route(
        vehicle_id=route.vehicle_id,
        vehicle_type=route.vehicle_type,
        nodes=new_nodes,
        start_times={},
        loads={},
    )


def _try_insertion_in_route(
    route: Route, pid: int, n: int, instance: Instance, weights: WeightTracker
) -> tuple[int, float, int, int] | None:
    """Find the best (p_pos, d_pos) pair to insert ``pid`` in ``route``.

    Returns ``(new_viols, score_increment, p_pos, d_pos)`` for the best
    placement, where ``new_viols`` counts violation *types* introduced
    (0 = strictly feasibility-non-worsening). Returns ``None`` only if the
    route has no valid insertion positions.

    Primary sort key is ``(new_viols, score_increment)`` so that a placement
    that introduces no new violations is always preferred over a cheaper-but-
    infeasible one, regardless of the current penalty weight magnitudes.
    """
    base_eval = evaluate_route(route, instance)
    base_score = penalised_score(base_eval.cost, base_eval.violations, weights)
    base_v = base_eval.violations
    best: tuple[tuple[int, float], int, int] | None = None
    n_nodes = len(route.nodes)

    for p_pos in range(1, n_nodes):
        for d_pos in range(p_pos, n_nodes):
            new_route = _insert_into_route(route, pid, p_pos, d_pos, n)
            ev = evaluate_route(new_route, instance)
            v_new = ev.violations
            new_viols = (
                int(v_new.q > base_v.q + 1e-6)
                + int(v_new.r > base_v.r + 1e-6)
                + int(v_new.d > base_v.d + 1e-6)
                + int(v_new.t > base_v.t + 1e-6)
                + int(v_new.u > base_v.u + 1e-6)
            )
            new_score = penalised_score(ev.cost, v_new, weights)
            inc = new_score - base_score
            key = (new_viols, float(inc))
            if best is None or key < best[0]:
                best = (key, p_pos, d_pos)

    if best is None:
        return None
    key, p, d = best
    return key[0], key[1], p, d


def _open_new_route(sol: Solution, instance: Instance) -> Route:
    """Allocate a fresh route id and pick a Common (preferred) or Large vehicle."""
    n = sol.n_passengers
    used_ids = {r.vehicle_id for r in sol.routes}
    next_vid = max(used_ids, default=-1) + 1
    used_common = sum(1 for r in sol.routes if r.vehicle_type == "Common")
    if used_common < instance.vehicle_stock("Common"):
        vtype = "Common"
    else:
        vtype = "Large"
    return Route(
        vehicle_id=next_vid,
        vehicle_type=vtype,
        nodes=[0, 2 * n + 1],
        start_times={},
        loads={},
    )


def _replace_route(sol: Solution, route: Route) -> Solution:
    new_routes: list[Route] = []
    replaced = False
    for r in sol.routes:
        if r.vehicle_id == route.vehicle_id:
            new_routes.append(route)
            replaced = True
        else:
            new_routes.append(r)
    if not replaced:
        new_routes.append(route)
    return Solution(
        instance_label=sol.instance_label,
        n_passengers=sol.n_passengers,
        routes=new_routes,
        total_cost=sol.total_cost,
        fixed_cost=sol.fixed_cost,
        variable_cost=sol.variable_cost,
        is_feasible=False,
        metadata=dict(sol.metadata),
    )
_MAX_ROUTES_TO_SEARCH = 80  # geographic pre-filter for best_insertion


def _route_proximity_key(
    route: "Route", pid_pickup: tuple[float, float], instance: "Instance", n: int
) -> float:
    """Cheap squared-Euclidean proxy from ``pid_pickup`` to the last passenger
    node in ``route`` (used for geographic pre-filtering in best_insertion)."""
    last_pax = next((nd for nd in reversed(route.nodes) if 1 <= nd <= 2 * n), None)
    if last_pax is None:
        return float("inf")
    info_nd = last_pax if last_pax <= n else last_pax - n
    info = instance.pax_dict(info_nd)
    coord = (info["pickup_lat"], info["pickup_lon"]) if last_pax <= n else (info["delivery_lat"], info["delivery_lon"])
    dlat = pid_pickup[0] - coord[0]
    dlon = pid_pickup[1] - coord[1]
    return dlat * dlat + dlon * dlon


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def best_insertion(
    sol: Solution,
    removed: list[int],
    rng: np.random.Generator,
    instance: Instance,
    weights: WeightTracker,
) -> Solution:
    """Insert each removed pid in the lowest-scoring position over all routes.

    Sort key: ``(new_viols, score_increment)`` — feasibility-non-worsening
    placements are always preferred. A new route is opened only when NO
    existing route has any valid insertion position (best is None), keeping
    the route count stable and letting the penalty mechanism fix violations.

    For large fleets (> ``_MAX_ROUTES_TO_SEARCH`` routes), only the
    geographically nearest routes are evaluated for speed.
    """
    n = sol.n_passengers
    pending = list(removed)
    rng.shuffle(pending)

    for pid in pending:
        info = instance.pax_dict(pid)
        pid_pickup = (info["pickup_lat"], info["pickup_lon"])

        if len(sol.routes) > _MAX_ROUTES_TO_SEARCH:
            routes_to_search = sorted(
                range(len(sol.routes)),
                key=lambda i: _route_proximity_key(sol.routes[i], pid_pickup, instance, n),
            )[:_MAX_ROUTES_TO_SEARCH]
        else:
            routes_to_search = list(range(len(sol.routes)))

        best: tuple[tuple[int, float], int, int, int] | None = None
        for r_idx in routes_to_search:
            route = sol.routes[r_idx]
            placement = _try_insertion_in_route(route, pid, n, instance, weights)
            if placement is None:
                continue
            nv, inc, p_pos, d_pos = placement
            key: tuple[int, float] = (nv, float(inc))
            if best is None or key < best[0]:
                best = (key, r_idx, p_pos, d_pos)

        if best is not None:
            _, r_idx, p_pos, d_pos = best
            sol = _replace_route(sol, _insert_into_route(sol.routes[r_idx], pid, p_pos, d_pos, n))
        else:
            # Last resort: open a new route.
            new_route = _open_new_route(sol, instance)
            new_route = _insert_into_route(new_route, pid, 1, 1, n)
            sol = _replace_route(sol, new_route)

    return sol


def random_insertion(
    sol: Solution,
    removed: list[int],
    rng: np.random.Generator,
    instance: Instance,
    weights: WeightTracker,
) -> Solution:
    """Insert each removed pid at a random valid position."""
    n = sol.n_passengers
    pending = list(removed)
    rng.shuffle(pending)

    for pid in pending:
        if sol.routes:
            r_idx = int(rng.integers(0, len(sol.routes)))
            route = sol.routes[r_idx]
            n_nodes = len(route.nodes)
            p_pos = int(rng.integers(1, max(2, n_nodes)))
            d_pos = int(rng.integers(p_pos, max(p_pos + 1, n_nodes)))
            new_route = _insert_into_route(route, pid, p_pos, d_pos, n)
            sol = _replace_route(sol, new_route)
        else:
            new_route = _open_new_route(sol, instance)
            new_route = _insert_into_route(new_route, pid, 1, 1, n)
            sol = _replace_route(sol, new_route)
    return sol


def _regret_insertion(
    sol: Solution,
    removed: list[int],
    rng: np.random.Generator,
    instance: Instance,
    weights: WeightTracker,
    k_regret: int,
) -> Solution:
    """Generic regret-k insertion: at each step, pick the request whose gap
    between the best and the k-th best insertion is largest, and place it
    in its best slot. Uses geographic pre-filtering for large fleets."""
    n = sol.n_passengers
    pending = list(removed)

    while pending:
        best_overall: tuple[float, int, int, int, int] | None = None
        for pid in pending:
            info = instance.pax_dict(pid)
            pid_pickup = (info["pickup_lat"], info["pickup_lon"])
            if len(sol.routes) > _MAX_ROUTES_TO_SEARCH:
                routes_to_search = sorted(
                    range(len(sol.routes)),
                    key=lambda i: _route_proximity_key(sol.routes[i], pid_pickup, instance, n),
                )[:_MAX_ROUTES_TO_SEARCH]
            else:
                routes_to_search = list(range(len(sol.routes)))

            increments: list[tuple[float, int, int, int]] = []
            for r_idx in routes_to_search:
                route = sol.routes[r_idx]
                placement = _try_insertion_in_route(route, pid, n, instance, weights)
                if placement is None:
                    continue
                _nv, inc, p_pos, d_pos = placement
                increments.append((inc, r_idx, p_pos, d_pos))
            if not increments:
                # Force a new route — high penalty placeholder so this pid is processed next.
                new_route = _open_new_route(sol, instance)
                new_route = _insert_into_route(new_route, pid, 1, 1, n)
                sol = _replace_route(sol, new_route)
                pending.remove(pid)
                break
            increments.sort(key=lambda x: x[0])
            if len(increments) >= k_regret:
                regret = increments[k_regret - 1][0] - increments[0][0]
            else:
                regret = -increments[0][0]  # no second option: prioritise heavily
            score = (-regret, increments[0][0])  # tiebreak by lower insertion cost
            if best_overall is None or score < (-best_overall[0], best_overall[1]):
                best_overall = (regret, increments[0][0], pid, increments[0][2], increments[0][3])
        else:
            # 'else' on the for-loop runs when no break happened.
            if best_overall is None:
                break  # nothing more to insert
            regret, _inc, pid, p_pos, d_pos = best_overall
            # Re-scan with geo-filter to find consistent (route_idx, p_pos, d_pos).
            info = instance.pax_dict(pid)
            pid_pickup = (info["pickup_lat"], info["pickup_lon"])
            if len(sol.routes) > _MAX_ROUTES_TO_SEARCH:
                rescan_routes = sorted(
                    range(len(sol.routes)),
                    key=lambda i: _route_proximity_key(sol.routes[i], pid_pickup, instance, n),
                )[:_MAX_ROUTES_TO_SEARCH]
            else:
                rescan_routes = list(range(len(sol.routes)))
            best_route_idx = -1
            best_inc = float("inf")
            best_p_pos = p_pos
            best_d_pos = d_pos
            for r_idx in rescan_routes:
                route = sol.routes[r_idx]
                placement = _try_insertion_in_route(route, pid, n, instance, weights)
                if placement is None:
                    continue
                _nv, inc, rp, rd = placement
                if inc < best_inc:
                    best_inc = inc
                    best_route_idx = r_idx
                    best_p_pos = rp
                    best_d_pos = rd
            if best_route_idx == -1:
                new_route = _open_new_route(sol, instance)
                new_route = _insert_into_route(new_route, pid, 1, 1, n)
                sol = _replace_route(sol, new_route)
            else:
                new_route = _insert_into_route(
                    sol.routes[best_route_idx], pid, best_p_pos, best_d_pos, n
                )
                sol = _replace_route(sol, new_route)
            pending.remove(pid)
    return sol


def regret2_insertion(
    sol: Solution,
    removed: list[int],
    rng: np.random.Generator,
    instance: Instance,
    weights: WeightTracker,
) -> Solution:
    return _regret_insertion(sol, removed, rng, instance, weights, k_regret=2)


def regret3_insertion(
    sol: Solution,
    removed: list[int],
    rng: np.random.Generator,
    instance: Instance,
    weights: WeightTracker,
) -> Solution:
    return _regret_insertion(sol, removed, rng, instance, weights, k_regret=3)


def zero_load_insertion(
    sol: Solution,
    removed: list[int],
    rng: np.random.Generator,
    instance: Instance,
    weights: WeightTracker,
) -> Solution:
    """Pilati 2025 novel operator: prefer slots where the vehicle load is zero.

    For each removed pid, locate the route where the load profile has the
    most "empty" slots (nodes with load 0) and insert there. Falls back to
    :func:`best_insertion` if no zero-load slot is available.
    """
    n = sol.n_passengers
    pending = list(removed)
    rng.shuffle(pending)

    for pid in pending:
        best: tuple[float, int, int, int] | None = None
        for r_idx, route in enumerate(sol.routes):
            ev = evaluate_route(route, instance)
            zero_load_nodes = [node for node, load in ev.loads.items() if load == 0]
            if not zero_load_nodes:
                continue
            # Try inserting pickup right after a zero-load node.
            for zln in zero_load_nodes:
                if zln not in route.nodes:
                    continue
                p_pos = route.nodes.index(zln) + 1
                if p_pos >= len(route.nodes):
                    continue
                # Insert delivery right after pickup (length 1 route segment).
                d_pos = p_pos
                new_route = _insert_into_route(route, pid, p_pos, d_pos, n)
                ev_new = evaluate_route(new_route, instance)
                new_score = penalised_score(ev_new.cost, ev_new.violations, weights)
                base_score = penalised_score(ev.cost, ev.violations, weights)
                inc = new_score - base_score
                if best is None or inc < best[0]:
                    best = (inc, r_idx, p_pos, d_pos)

        if best is not None:
            _, r_idx, p_pos, d_pos = best
            new_route = _insert_into_route(sol.routes[r_idx], pid, p_pos, d_pos, n)
            sol = _replace_route(sol, new_route)
        else:
            # Fall back: best insertion or open a new route.
            sol = best_insertion(sol, [pid], rng, instance, weights)
    return sol


# Registry
REPAIR_OPERATORS: dict[str, RepairFn] = {
    "best": best_insertion,
    "random": random_insertion,
    "regret2": regret2_insertion,
    "regret3": regret3_insertion,
    "zero_load": zero_load_insertion,
}

# Faster subset for large instances (n > ~100) where regret operators are
# prohibitively slow even with geographic pre-filtering.
FAST_REPAIR_OPERATORS: dict[str, RepairFn] = {
    "best": best_insertion,
    "random": random_insertion,
    "zero_load": zero_load_insertion,
}
