"""Initial-solution constructor for the ALNS.

Implements the Pilati et al. (2025) randomized constructive heuristic
described in Section 4.5.2 of ``informe/chapters/metodologia.tex``:

1. Order requests by ``e_i`` ascending (effective earliest pickup).
2. Seed one passenger per route.
3. For every remaining passenger, try to insert into the route that
   minimises **one of four distance metrics chosen at random** for that
   passenger; the eventual infeasibility is absorbed by the dynamic
   penalties of the ALNS evaluator.
   The four metrics, between the last node of route and the new request:
       - last.pickup  -> new.pickup
       - last.pickup  -> new.delivery
       - last.delivery -> new.pickup
       - last.delivery -> new.delivery
4. If no existing route accepts (cost is finite under the heuristic), open a
   new route choosing a vehicle type from the pool. Common is preferred when
   stock allows.

The constructor produces a :class:`Solution` whose ``nodes`` lists are full
sequences ``[0, p_a, d_a, p_b, d_b, ..., 2n+1]`` (pickup-deliver pattern per
request, no batching) so the ALNS engine can immediately mutate them with
its destroy/repair operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.alns.evaluation import refresh_solution_metrics
from src.alns.solution import Route, Solution
from src.data.instance import Instance


@dataclass
class FleetPool:
    """Available stock per vehicle type (defaults to the instance's stock)."""

    common: int
    large: int

    @classmethod
    def from_instance(cls, instance: Instance, common: int | None = None, large: int | None = None) -> "FleetPool":
        return cls(
            common=common if common is not None else instance.vehicle_stock("Common"),
            large=large if large is not None else instance.vehicle_stock("Large"),
        )


@dataclass
class _RouteDraft:
    """Mutable container while building a route."""

    vehicle_id: int
    vehicle_type: str
    nodes: list[int] = field(default_factory=lambda: [0])  # always starts at depot
    last_pickup_coord: tuple[float, float] | None = None
    last_delivery_coord: tuple[float, float] | None = None


def _coord(instance: Instance, pid: int, kind: str) -> tuple[float, float]:
    row = instance.passengers.set_index("id").loc[pid]
    if kind == "pickup":
        return float(row["pickup_lat"]), float(row["pickup_lon"])
    return float(row["delivery_lat"]), float(row["delivery_lon"])


def _distance_metric(
    instance: Instance,
    draft: _RouteDraft,
    pid: int,
    metric: int,
) -> float | None:
    """Return the distance under one of the four Pilati metrics, or None
    if this metric is not applicable yet (e.g. route has no prior pickup)."""
    p_new = _coord(instance, pid, "pickup")
    d_new = _coord(instance, pid, "delivery")

    if draft.last_pickup_coord is None and draft.last_delivery_coord is None:
        # Empty route: every metric collapses to 0 (virtual depot).
        return 0.0

    if metric == 0:
        if draft.last_pickup_coord is None:
            return None
        return instance.distance(draft.last_pickup_coord, p_new)
    if metric == 1:
        if draft.last_pickup_coord is None:
            return None
        return instance.distance(draft.last_pickup_coord, d_new)
    if metric == 2:
        if draft.last_delivery_coord is None:
            return None
        return instance.distance(draft.last_delivery_coord, p_new)
    if metric == 3:
        if draft.last_delivery_coord is None:
            return None
        return instance.distance(draft.last_delivery_coord, d_new)
    raise ValueError(f"Unknown metric: {metric}")


def _append_passenger(draft: _RouteDraft, pid: int, n: int, instance: Instance) -> None:
    """Append a (pickup, delivery) pair at the end of the route, updating
    the cached coordinates."""
    pickup_node = pid
    delivery_node = pid + n
    draft.nodes.append(pickup_node)
    draft.nodes.append(delivery_node)
    draft.last_pickup_coord = _coord(instance, pid, "pickup")
    draft.last_delivery_coord = _coord(instance, pid, "delivery")


def construct_initial_solution(
    instance: Instance,
    fleet: FleetPool | None = None,
    seed: int = 0,
    n_seed_routes: int | None = None,
) -> Solution:
    """Build an initial Pilati-style solution. May be infeasible at this
    stage; the ALNS engine will repair it through the dynamic penalties.

    Algorithm (Pilati et al. 2025, §4.1):

    1. Sort passengers by effective ``e_i``.
    2. Seed ``K`` routes with the first ``K`` passengers (one each).
    3. For the remaining passengers, insert into the route minimising one
       of four distance metrics chosen at random per request. Ties are
       broken uniformly.
    """
    rng = np.random.default_rng(seed)
    fleet = fleet or FleetPool.from_instance(instance)
    n = instance.n_passengers()

    # 1. Order passengers by effective e_i.
    order: list[tuple[float, int]] = []
    for pid in instance.passengers["id"].astype(int).tolist():
        e_eff, _sigma = instance.effective_pickup_window(int(pid))
        order.append((e_eff, int(pid)))
    order.sort(key=lambda x: (x[0], x[1]))

    # 2. Decide how many routes to seed up front. Default uses a sensible
    # heuristic of ceil(n / capacity_common) so the average load matches a
    # Common vehicle's capacity. The user can override via ``n_seed_routes``.
    cap_common = instance.vehicle_capacity("Common")
    if n_seed_routes is None:
        n_seed_routes = max(1, (n + cap_common - 1) // cap_common)
    n_seed_routes = min(n_seed_routes, n, fleet.common + fleet.large)

    drafts: list[_RouteDraft] = []
    next_vid = 0
    common_left = fleet.common
    large_left = fleet.large

    # 2b. Open the seed routes by selecting a vehicle type for each.
    for seed_idx in range(n_seed_routes):
        if common_left > 0:
            vtype = "Common"
            common_left -= 1
        elif large_left > 0:
            vtype = "Large"
            large_left -= 1
        else:
            break
        draft = _RouteDraft(vehicle_id=next_vid, vehicle_type=vtype)
        next_vid += 1
        _, pid = order[seed_idx]
        _append_passenger(draft, pid, n, instance)
        drafts.append(draft)

    # 3. Insert the remaining passengers.
    for _e_i, pid in order[len(drafts):]:
        metric = int(rng.integers(0, 4))
        best: tuple[float, _RouteDraft] | None = None
        for draft in drafts:
            d = _distance_metric(instance, draft, pid, metric)
            if d is None:
                for fb in (0, 1, 2, 3):
                    if fb == metric:
                        continue
                    d = _distance_metric(instance, draft, pid, fb)
                    if d is not None:
                        break
            if d is None:
                continue
            if best is None or d < best[0]:
                best = (float(d), draft)

        if best is not None:
            _append_passenger(best[1], pid, n, instance)
            continue

        # Fall back: open a fresh route if the heuristic insertion failed.
        if common_left > 0:
            vtype = "Common"
            common_left -= 1
        elif large_left > 0:
            vtype = "Large"
            large_left -= 1
        else:
            raise RuntimeError(
                f"Fleet exhausted at passenger {pid}. Increase FleetPool stocks."
            )
        draft = _RouteDraft(vehicle_id=next_vid, vehicle_type=vtype)
        next_vid += 1
        _append_passenger(draft, pid, n, instance)
        drafts.append(draft)

    # Close every route at depot end.
    routes: list[Route] = []
    for d in drafts:
        d.nodes.append(2 * n + 1)
        routes.append(
            Route(
                vehicle_id=d.vehicle_id,
                vehicle_type=d.vehicle_type,
                nodes=list(d.nodes),
                start_times={},
                loads={},
            )
        )

    sol = Solution(
        instance_label=f"{instance.label}__pilati_init",
        n_passengers=n,
        routes=routes,
        total_cost=0.0,
        fixed_cost=0.0,
        variable_cost=0.0,
        is_feasible=False,
        metadata={"constructor": "pilati", "seed": seed},
    )
    refresh_solution_metrics(sol, instance)
    return sol
