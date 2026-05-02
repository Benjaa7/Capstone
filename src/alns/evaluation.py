"""ALNS evaluation function with dynamic penalty weights.

Implements the formulation from Pilati et al. (2025), extended with the
sindical category-violation term we adapted for this project (see
``informe/chapters/metodologia.tex`` Section 4.5):

    f(s) = c(s) + α·q(s) + β·r(s) + γ·d(s) + ε·t(s) + φ·u(s)

The five violation magnitudes are:

* ``q(s)``: total capacity excess (units) summed over all routes.
* ``r(s)``: total ride-time excess (seconds) over ``M_{r_i}``.
* ``d(s)``: total advance excess (seconds) over ``W_{r_i}`` (AM convention).
* ``t(s)``: total time-window violation (seconds) at any node.
* ``u(s)``: number of disallowed sindical category pairs across vehicles.

Weights start at 1 and are multiplied by ``δ ~ U(δ_lo, δ_hi)`` if the
corresponding violation is non-zero in the candidate, or divided otherwise.
A solution is incumbent-eligible only when ``Violations.is_clean()`` (i.e.
``f(s) = c(s)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.alns.solution import Route, Solution

if TYPE_CHECKING:
    from src.data.instance import Instance


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------
@dataclass
class Violations:
    q: float = 0.0  # capacity excess (units)
    r: float = 0.0  # ride-time excess (s)
    d: float = 0.0  # advance excess (s)
    t: float = 0.0  # time-window violation (s)
    u: float = 0.0  # sindical pair violations (count)

    def is_clean(self, eps: float = 1e-6) -> bool:
        return (self.q + self.r + self.d + self.t + self.u) <= eps

    def __add__(self, other: "Violations") -> "Violations":
        return Violations(
            q=self.q + other.q,
            r=self.r + other.r,
            d=self.d + other.d,
            t=self.t + other.t,
            u=self.u + other.u,
        )

    def to_dict(self) -> dict[str, float]:
        return {"q": self.q, "r": self.r, "d": self.d, "t": self.t, "u": self.u}


# ---------------------------------------------------------------------------
# Adaptive weight tracker
# ---------------------------------------------------------------------------
@dataclass
class WeightTracker:
    """Five Pilati-style penalty weights with multiplicative updates."""

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    epsilon: float = 1.0
    phi: float = 1.0

    delta_lo: float = 1.05
    delta_hi: float = 1.10
    w_min: float = 1e-3
    w_max: float = 1e4
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def _adjust(self, w: float, violated: bool) -> float:
        delta = float(self.rng.uniform(self.delta_lo, self.delta_hi))
        new = w * delta if violated else w / delta
        return float(np.clip(new, self.w_min, self.w_max))

    def update(self, v: Violations) -> None:
        self.alpha = self._adjust(self.alpha, v.q > 1e-6)
        self.beta = self._adjust(self.beta, v.r > 1e-6)
        self.gamma = self._adjust(self.gamma, v.d > 1e-6)
        self.epsilon = self._adjust(self.epsilon, v.t > 1e-6)
        self.phi = self._adjust(self.phi, v.u > 1e-6)


# ---------------------------------------------------------------------------
# Route walker
# ---------------------------------------------------------------------------
def _coord_of_node(instance: "Instance", node: int, n: int) -> tuple[float, float]:
    """Return ``(lat, lon)`` for a node in the MILP indexing convention."""
    if node == 0 or node == 2 * n + 1:
        return (0.0, 0.0)
    if 1 <= node <= n:
        info = instance.pax_dict(node)
        return info["pickup_lat"], info["pickup_lon"]
    info = instance.pax_dict(node - n)
    return info["delivery_lat"], info["delivery_lon"]


@dataclass
class _RouteEval:
    cost: float
    violations: Violations
    start_times: dict[int, float]
    loads: dict[int, int]


def evaluate_route(route: Route, instance: "Instance") -> _RouteEval:
    """Walk a single route and compute cost + violations + scheduled times.

    The route's ``nodes`` list is expected to be a fully-specified sequence
    starting at ``0`` (depot) and ending at ``2n+1`` (depot end). Pickup and
    delivery nodes are interleaved in any order; this function checks that
    each delivery comes after its pickup.

    Solo-trip clause (MILP constraint): if ``τ_direct(i) > M_r_i``, passenger
    ``i`` must travel alone (load ≤ 1 between their pickup and delivery).
    Violations are counted in ``v.q`` (capacity-type).
    The set of such passengers is cached on the instance for performance.
    """
    n = instance.n_passengers()
    s_pickup = float(instance.others["pickup_duration"])
    s_delivery = float(instance.others["delivery_duration"])
    cost_per_meter = float(instance.others["cost_per_meter"])

    # Solo-trip passengers: cached on instance to avoid recomputing each call.
    solo_pids: set[int] = instance.solo_passenger_ids

    nodes = list(route.nodes)
    if not nodes:
        return _RouteEval(0.0, Violations(), {}, {})

    cap = float(instance.vehicle_capacity(route.vehicle_type))
    fixed_cost = float(instance.vehicle_fixed_cost(route.vehicle_type))

    cost = fixed_cost
    v = Violations()
    times: dict[int, float] = {}
    loads: dict[int, int] = {}

    pickup_times: dict[int, float] = {}
    categories_seen: set[int] = set()
    load = 0
    time_now = 0.0
    last_coord: tuple[float, float] | None = None

    for idx, node in enumerate(nodes):
        if node == 0:
            times[node] = 0.0
            loads[node] = 0
            continue
        if node == 2 * n + 1:
            times[node] = time_now
            loads[node] = load
            continue

        is_pickup = 1 <= node <= n
        pid = node if is_pickup else node - n
        coord = _coord_of_node(instance, node, n)
        info = instance.pax_dict(pid)
        priority = info["priority"]
        sigma_i = info["sigma_i"]
        W_r, M_r = instance.union[priority]

        # First real visit: arc from depot is free (virtual depot).
        if last_coord is None:
            arc_tau = 0.0
            arc_dist = 0.0
        else:
            arc_tau = instance.tau(last_coord, coord)
            arc_dist = instance.distance(last_coord, coord)
        cost += cost_per_meter * arc_dist
        arrival = time_now + arc_tau

        if is_pickup:
            categories_seen.add(priority)
            e_i, l_i = instance.effective_pickup_window(pid)
            start = max(arrival, e_i)
            if start > l_i:
                v.t += start - l_i
            load += 1
            if load > cap:
                v.q += load - cap
            # Solo-trip clause: passenger with τ_direct > M_r must travel alone.
            if pid in solo_pids and load > 1:
                v.q += load - 1  # penalise every co-passenger present at pickup
            pickup_times[pid] = start
            times[node] = start
            loads[node] = load
            time_now = start + s_pickup
        else:  # delivery
            e_d = max(0.0, sigma_i - W_r - s_delivery)
            l_d = sigma_i - s_delivery
            start = max(arrival, e_d)
            if start > l_d:
                v.t += start - l_d
            advance = sigma_i - (start + s_delivery)
            if advance > W_r:
                v.d += advance - W_r
            if pid in pickup_times:
                ride = start - (pickup_times[pid] + s_pickup)
                # Solo-trip clause: passengers with τ_direct > M_r are
                # exempt from the ride-time constraint (they travel alone).
                if pid not in solo_pids and ride > M_r:
                    v.r += ride - M_r
            else:
                # Delivery without a prior pickup in this route — invalid
                # ordering. Penalise heavily through time-window slack.
                v.t += sigma_i  # large penalty
            load -= 1
            times[node] = start
            loads[node] = load
            time_now = start + s_delivery
        last_coord = coord

    # Sindical: count disallowed violations.
    # Rule: at most 2 categories, and they must be consecutive (|a-b| <= 1).
    # We count one violation per broken rule instance.
    cats = sorted(categories_seen)
    u_count = 0
    if len(cats) > 2:
        # Each category beyond the first two is a separate violation.
        u_count += len(cats) - 2
    for a, b in zip(cats, cats[1:], strict=False):
        if abs(a - b) > 1:
            u_count += 1
    v.u += u_count

    return _RouteEval(cost=cost, violations=v, start_times=times, loads=loads)


# ---------------------------------------------------------------------------
# Solution-level evaluator
# ---------------------------------------------------------------------------
def evaluate_solution(sol: Solution, instance: "Instance") -> tuple[float, Violations]:
    """Return ``(cost, violations)`` for a complete solution."""
    cost = 0.0
    v = Violations()
    for route in sol.routes:
        ev = evaluate_route(route, instance)
        cost += ev.cost
        v = v + ev.violations
    return cost, v


def penalised_score(cost: float, v: Violations, w: WeightTracker) -> float:
    """Return ``f(s) = c(s) + α q + β r + γ d + ε t + φ u``."""
    return (
        cost
        + w.alpha * v.q
        + w.beta * v.r
        + w.gamma * v.d
        + w.epsilon * v.t
        + w.phi * v.u
    )


def refresh_solution_metrics(sol: Solution, instance: "Instance") -> None:
    """Recompute cost, fixed/variable cost, feasibility and per-route start
    times/loads in-place so the Solution reflects its current ``nodes``."""
    total_cost = 0.0
    total_var = 0.0
    total_fix = 0.0
    feasible = True
    for route in sol.routes:
        ev = evaluate_route(route, instance)
        route.start_times = ev.start_times
        route.loads = ev.loads
        total_cost += ev.cost
        total_fix += float(instance.vehicle_fixed_cost(route.vehicle_type))
        total_var += ev.cost - float(instance.vehicle_fixed_cost(route.vehicle_type))
        if not ev.violations.is_clean():
            feasible = False
    sol.total_cost = total_cost
    sol.fixed_cost = total_fix
    sol.variable_cost = total_var
    sol.is_feasible = feasible
