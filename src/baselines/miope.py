"""Greedy myopic baseline for the TD-HDARP.

Implements the heuristic described in Section 4.4.2 of
``informe/chapters/metodologia.tex``:

    Para cada pasajero, en orden ascendente por σ_i, se asigna al
    vehículo activo (Common o Large) que minimice el costo marginal de
    insertar pickup y delivery dentro de las ventanas factibles. Si
    ningún vehículo activo acepta, se abre uno nuevo.

The output is a :class:`Solution` consistent with what the MILP and the
ALNS produce, so KPIs and visualizations are computed uniformly across
solvers.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.alns.solution import Route, Solution
from src.data.instance import Instance


@dataclass
class _RouteState:
    """Lightweight mutable representation used while building each route."""

    vehicle_id: int
    vehicle_type: str
    capacity: int
    fixed_cost: float
    nodes: list[int]
    start_times: dict[int, float]
    loads: dict[int, int]
    distance: float
    categories: set[int]


class MiopeSolver:
    """Greedy assignment respecting all hard constraints of the TD-HDARP."""

    def __init__(self, instance: Instance) -> None:
        pax = instance.passengers.copy()
        if not (pax["id"].min() == 1 and pax["id"].max() == len(pax)):
            raise ValueError(
                "MiopeSolver requires passengers with compact ids 1..n. "
                "Use src.data.case_base.select_case_base."
            )
        self.instance = instance
        self.passengers = pax.set_index("id", drop=False)
        self.n = len(pax)
        self.depot_start = 0
        self.depot_end = 2 * self.n + 1

        self.s_pickup = float(instance.others["pickup_duration"])
        self.s_delivery = float(instance.others["delivery_duration"])
        self.cost_per_meter = float(instance.others["cost_per_meter"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _coord(self, pid: int, kind: str) -> tuple[float, float]:
        row = self.passengers.loc[pid]
        if kind == "pickup":
            return float(row["pickup_lat"]), float(row["pickup_lon"])
        return float(row["delivery_lat"]), float(row["delivery_lon"])

    def _tau(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return self.instance.tau(p1, p2, t=None)

    def _dist(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return self.instance.distance(p1, p2, t=None)

    def _categories_compatible(self, current: set[int], new_cat: int) -> bool:
        """Adding ``new_cat`` keeps the route within "at most 2 consecutive categories"."""
        candidate = current | {new_cat}
        if len(candidate) > 2:
            return False
        if len(candidate) == 2:
            a, b = sorted(candidate)
            if abs(a - b) > 1:
                return False
        return True

    def _try_insert(self, state: _RouteState, pid: int) -> tuple[float, _RouteState] | None:
        """Try to append ``pid`` to ``state``. Returns (incremental_cost, new_state)
        if feasible, ``None`` otherwise. The myopic heuristic appends pickup and
        delivery at the END of the current route (no internal insertion search)
        to keep the baseline clean and fast."""
        row = self.passengers.loc[pid]
        priority = int(row["priority"])
        e_i, sigma = self.instance.effective_pickup_window(pid)
        W_r, M_r = self.instance.union[priority]

        # Sindical compatibility
        if not self._categories_compatible(state.categories, priority):
            return None

        # Capacity: load before pickup must be 0 (since we append at the end and
        # all previous pairs are completed before).
        last_node = state.nodes[-1]
        if state.loads.get(last_node, 0) >= state.capacity:
            return None

        # Compute pickup time
        if last_node == self.depot_start:
            pickup_time = e_i
            from_coord = self._coord(pid, "pickup")  # distance from "depot" is 0 in our model
            arc_dist_to_pickup = 0.0
            arc_tau_to_pickup = 0.0
        else:
            last_pid = last_node if last_node <= self.n else last_node - self.n
            last_kind = "pickup" if last_node <= self.n else "delivery"
            from_coord = self._coord(last_pid, last_kind)
            to_coord = self._coord(pid, "pickup")
            arc_tau_to_pickup = self._tau(from_coord, to_coord)
            arc_dist_to_pickup = self._dist(from_coord, to_coord)
            pickup_time = max(
                e_i,
                state.start_times[last_node] + self._service_time(last_node) + arc_tau_to_pickup,
            )

        if pickup_time > sigma:  # cannot pickup after turn starts
            return None

        # Pickup -> delivery arc
        pickup_coord = self._coord(pid, "pickup")
        delivery_coord = self._coord(pid, "delivery")
        tau_pd = self._tau(pickup_coord, delivery_coord)
        dist_pd = self._dist(pickup_coord, delivery_coord)
        delivery_time = pickup_time + self.s_pickup + tau_pd

        # Ride time check (skipped for alone passengers, but the myopic baseline
        # does not implement the alone fallback — those passengers will fail and
        # force a new vehicle).
        ride_time = delivery_time - (pickup_time + self.s_pickup)
        if ride_time > M_r:
            return None

        # Delivery time window: [sigma - W_r - s_delivery, sigma - s_delivery]
        delivery_lb = max(0.0, sigma - W_r - self.s_delivery)
        delivery_ub = sigma - self.s_delivery
        if delivery_time < delivery_lb:
            delivery_time = delivery_lb  # may wait at delivery point
        if delivery_time > delivery_ub:
            return None

        # Build the updated state
        pickup_node = pid
        delivery_node = pid + self.n
        new_nodes = list(state.nodes) + [pickup_node, delivery_node]
        new_start_times = dict(state.start_times)
        new_start_times[pickup_node] = pickup_time
        new_start_times[delivery_node] = delivery_time
        new_loads = dict(state.loads)
        new_loads[pickup_node] = state.loads.get(last_node, 0) + 1
        new_loads[delivery_node] = new_loads[pickup_node] - 1
        new_distance = state.distance + arc_dist_to_pickup + dist_pd
        new_categories = state.categories | {priority}

        marginal_cost = self.cost_per_meter * (arc_dist_to_pickup + dist_pd)

        new_state = _RouteState(
            vehicle_id=state.vehicle_id,
            vehicle_type=state.vehicle_type,
            capacity=state.capacity,
            fixed_cost=state.fixed_cost,
            nodes=new_nodes,
            start_times=new_start_times,
            loads=new_loads,
            distance=new_distance,
            categories=new_categories,
        )
        return marginal_cost, new_state

    def _service_time(self, node: int) -> float:
        if node == self.depot_start or node == self.depot_end:
            return 0.0
        return self.s_pickup if node <= self.n else self.s_delivery

    def _new_route(self, vehicle_id: int, vehicle_type: str) -> _RouteState:
        return _RouteState(
            vehicle_id=vehicle_id,
            vehicle_type=vehicle_type,
            capacity=self.instance.vehicle_capacity(vehicle_type),
            fixed_cost=self.instance.vehicle_fixed_cost(vehicle_type),
            nodes=[self.depot_start],
            start_times={self.depot_start: 0.0},
            loads={self.depot_start: 0},
            distance=0.0,
            categories=set(),
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self) -> Solution:
        # Sort passengers by sigma_i (turn start time) ascending.
        order = self.passengers.sort_values(by=["sigma_i", "e_i"])["id"].astype(int).tolist()

        routes: list[_RouteState] = []
        next_vehicle_id = 0

        # Stocks remaining
        stock_common = self.instance.vehicle_stock("Common")
        stock_large = self.instance.vehicle_stock("Large")

        for pid in order:
            best: tuple[float, int, _RouteState] | None = None  # (cost, route_idx, new_state)
            for idx, state in enumerate(routes):
                result = self._try_insert(state, pid)
                if result is None:
                    continue
                cost, new_state = result
                if best is None or cost < best[0]:
                    best = (cost, idx, new_state)
            if best is not None:
                _, idx, new_state = best
                routes[idx] = new_state
                continue

            # No active route accepts: open a new one. Prefer Common (cheaper) unless capacity warrants Large.
            opened: _RouteState | None = None
            for vtype in ("Common", "Large"):
                if vtype == "Common" and stock_common <= 0:
                    continue
                if vtype == "Large" and stock_large <= 0:
                    continue
                fresh = self._new_route(next_vehicle_id, vtype)
                result = self._try_insert(fresh, pid)
                if result is None:
                    continue
                _, new_state = result
                opened = new_state
                if vtype == "Common":
                    stock_common -= 1
                else:
                    stock_large -= 1
                next_vehicle_id += 1
                break
            if opened is None:
                raise RuntimeError(
                    f"No available vehicle could serve passenger {pid}. "
                    "The fleet stock is exhausted or constraints are infeasible."
                )
            routes.append(opened)

        return self._finalize(routes)

    def _finalize(self, route_states: list[_RouteState]) -> Solution:
        out_routes: list[Route] = []
        fixed_cost = 0.0
        variable_cost = 0.0
        for state in route_states:
            # Close route at depot_end with zero arc cost (virtual depot).
            nodes = list(state.nodes) + [self.depot_end]
            start_times = dict(state.start_times)
            start_times[self.depot_end] = 0.0
            loads = dict(state.loads)
            loads[self.depot_end] = 0
            fixed_cost += state.fixed_cost
            variable_cost += self.cost_per_meter * state.distance
            out_routes.append(
                Route(
                    vehicle_id=state.vehicle_id,
                    vehicle_type=state.vehicle_type,
                    nodes=nodes,
                    start_times=start_times,
                    loads=loads,
                )
            )

        total = fixed_cost + variable_cost
        return Solution(
            instance_label=f"{self.instance.label}__miope",
            n_passengers=self.n,
            routes=out_routes,
            total_cost=total,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
            is_feasible=True,
            metadata={
                "solver": "miope",
                "n_vehicles_used": len(out_routes),
                "n_passengers": self.n,
            },
        )
