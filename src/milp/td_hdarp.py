"""Mixed-Integer Linear Programming model for the TD-HDARP.

Implements the formulation given in ``informe/chapters/metodologia.tex``,
equations (4.1)–(4.14). The Caso Base uses ``time_dependent=False`` (static
:math:`\\tau_{ij}`, computed as the cross-hour mean of the available matrices)
per the professor's instruction; the time-dependent variant is left as a hook
for the Entrega 3 final pass.

The model produces a :class:`~src.alns.solution.Solution` so that the same
downstream tooling (KPIs, plots, JSON persistence) can be reused for the ALNS.

Note on ``pickup_from`` interpretation
--------------------------------------
The raw data contains passengers whose ``pickup_from`` is too late to allow
on-time arrival even with direct travel (e.g. pickup_from at 04:30, turn at
04:55, direct travel 30 min). Operationally these workers must be picked up
earlier than their stated preference: the hard requirement is on-time arrival
(within ``W_r`` of ``sigma_i``), not on the preferred pickup hour. The model
therefore uses ``e_i_eff = min(pickup_from, sigma - W_r - tau_direct - s_p -
s_d)`` so direct travel is always feasible. Cases where this relaxation kicks
in are reported in :attr:`TDHDARPModel.relaxed_pickup_passengers`.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.alns.solution import Route, Solution
from src.data.instance import Instance


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class FleetSize:
    """Pre-allocated fleet pool for the MILP.

    Choose enough vehicles to absorb the worst-case scenario implied by the
    sindical constraints (priority 4 alone, etc.). Defaults are tuned for the
    30-passenger Caso Base.
    """

    common: int = 12
    large: int = 4


@dataclass
class MilpConfig:
    """Solver-level options."""

    time_dependent: bool = False
    big_m_time: float = 24 * 3600.0  # tight: any time delta fits in a 24h day
    big_m_load: float = 10.0  # tight: max capacity is 6
    tight_per_arc_big_m: bool = True  # use l_i + s_i + tau_ij - e_j per arc when smaller
    time_limit_s: int = 12 * 3600  # 12 h default
    mip_gap: float = 0.01
    threads: int = 0  # 0 = use all available
    verbose: bool = True
    symmetry_breaking: bool = True
    eliminate_infeasible_arcs: bool = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TDHDARPModel:
    """Build, solve and extract solutions for the TD-HDARP MILP."""

    def __init__(
        self,
        instance: Instance,
        fleet_size: FleetSize | None = None,
        config: MilpConfig | None = None,
        warm_start: Solution | None = None,
    ) -> None:
        self.warm_start = warm_start
        self.instance = instance
        self.fleet_size = fleet_size or FleetSize()
        self.config = config or MilpConfig()

        # ---- Passengers (must use compact 1..n ids; case_base.py guarantees this) ----
        pax = instance.passengers.copy()
        if not (pax["id"].min() == 1 and pax["id"].max() == len(pax)):
            raise ValueError(
                "TDHDARPModel requires passengers with compact ids 1..n. "
                "Use src.data.case_base.select_case_base or renumber manually."
            )
        self.passengers = pax.set_index("id", drop=False)
        self.n = len(pax)

        # ---- Node indexing ----
        self.depot_start = 0
        self.depot_end = 2 * self.n + 1
        self.P = list(range(1, self.n + 1))
        self.D = list(range(self.n + 1, 2 * self.n + 1))
        self.V = [self.depot_start, *self.P, *self.D, self.depot_end]

        # ---- Vehicle pool ----
        self.vehicles = self._build_fleet()
        self.K = list(range(len(self.vehicles)))

        # ---- Travel time / distance matrices ----
        self.tau, self.dist = self._build_matrices()

        # ---- Node attributes (depend on tau for the e_i relaxation) ----
        self.relaxed_pickup_passengers: list[int] = []
        self.e, self.l, self.q, self.s, self.r = self._build_node_attrs()

        # ---- Sindical compatibility ----
        self.cat_compat = self._build_compat_pairs()

        # ---- Cost coefficients ----
        self.cost_per_meter = float(instance.others["cost_per_meter"])

        # ---- Gurobi model (built on demand) ----
        self.model: gp.Model | None = None
        self.x: dict[tuple[int, int, int], gp.Var] | None = None
        self.y: dict[int, gp.Var] | None = None
        self.B: dict[tuple[int, int], gp.Var] | None = None
        self.L: dict[tuple[int, int], gp.Var] | None = None
        self.u: dict[tuple[int, int], gp.Var] | None = None

        # ---- Detect alone-required passengers (direct travel exceeds M_r) ----
        self.alone_passengers = self._detect_alone_passengers()

        if self.relaxed_pickup_passengers:
            warnings.warn(
                f"Relaxed e_i for {len(self.relaxed_pickup_passengers)} passenger(s) "
                f"with infeasible pickup_from window: {self.relaxed_pickup_passengers}",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_fleet(self) -> list[dict[str, Any]]:
        veh: list[dict[str, Any]] = []
        kid = 0
        for vtype, count in (("Common", self.fleet_size.common), ("Large", self.fleet_size.large)):
            stock = self.instance.vehicle_stock(vtype)
            actual = min(count, stock)
            cap = self.instance.vehicle_capacity(vtype)
            fixed = self.instance.vehicle_fixed_cost(vtype)
            for _ in range(actual):
                veh.append({"id": kid, "type": vtype, "capacity": cap, "fixed_cost": fixed})
                kid += 1
        if not veh:
            raise ValueError("Empty fleet pool: configure FleetSize with at least one vehicle.")
        return veh

    def _node_coord(self, node: int) -> tuple[float, float]:
        if node == self.depot_start or node == self.depot_end:
            # Virtual depot: use centroid of pickups (only matters if τ/d are non-zero).
            return (
                float(self.passengers["pickup_lat"].mean()),
                float(self.passengers["pickup_lon"].mean()),
            )
        if node in self.P:
            row = self.passengers.loc[node]
            return float(row["pickup_lat"]), float(row["pickup_lon"])
        # Delivery
        pid = node - self.n
        row = self.passengers.loc[pid]
        return float(row["delivery_lat"]), float(row["delivery_lon"])

    def _build_node_attrs(
        self,
    ) -> tuple[dict[int, float], dict[int, float], dict[int, int], dict[int, float], dict[int, int]]:
        e: dict[int, float] = {}
        l: dict[int, float] = {}
        q: dict[int, int] = {}
        s: dict[int, float] = {}
        r: dict[int, int] = {}

        s_pickup = float(self.instance.others["pickup_duration"])
        s_delivery = float(self.instance.others["delivery_duration"])

        # Depots
        for d in (self.depot_start, self.depot_end):
            e[d] = 0.0
            l[d] = 24 * 3600.0
            q[d] = 0
            s[d] = 0.0
            r[d] = 0  # not used

        # Pickups: time window [pickup_from, sigma_i].
        # Use the effective_pickup_window helper from Instance so the MILP and
        # the miope baseline see the same relaxed e_i values.
        for i in self.P:
            row = self.passengers.loc[i]
            priority = int(row["priority"])
            r[i] = priority
            q[i] = +1
            s[i] = s_pickup
            e_eff, sigma_i = self.instance.effective_pickup_window(i)
            if self.instance.is_pickup_relaxed(i):
                self.relaxed_pickup_passengers.append(i)
            e[i] = e_eff
            l[i] = sigma_i  # loose upper bound; tightened by eq:r11 elsewhere

        # Deliveries: must finish service by sigma_i, no earlier than sigma_i - W_r
        for j in self.D:
            pid = j - self.n
            row = self.passengers.loc[pid]
            priority = int(row["priority"])
            r[j] = priority
            q[j] = -1
            s[j] = s_delivery
            sigma = float(row["sigma_i"])
            W_r, _M_r = self.instance.union[priority]
            e[j] = max(0.0, sigma - W_r - s_delivery)
            l[j] = max(e[j], sigma - s_delivery)

        return e, l, q, s, r

    def _build_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        N = self.depot_end + 1
        tau = np.zeros((N, N))
        dist = np.zeros((N, N))
        for i in self.V:
            for j in self.V:
                if i == j:
                    continue
                # Virtual depots: zero travel time and distance.
                if i == self.depot_start or j == self.depot_end:
                    continue
                if i == self.depot_end or j == self.depot_start:
                    continue
                co_i = self._node_coord(i)
                co_j = self._node_coord(j)
                tau[i, j] = self.instance.tau(co_i, co_j, t=None)
                dist[i, j] = self.instance.distance(co_i, co_j, t=None)
        return tau, dist

    def _build_compat_pairs(self) -> set[tuple[int, int]]:
        """Return ``{(r, r') : |r - r'| <= 1}`` over priorities 1..4."""
        return {(r1, r2) for r1 in (1, 2, 3, 4) for r2 in (1, 2, 3, 4) if abs(r1 - r2) <= 1}

    def _detect_alone_passengers(self) -> list[int]:
        """Passengers whose direct trip already exceeds their max ride time."""
        alone: list[int] = []
        for i in self.P:
            direct = self.tau[i, i + self.n]
            _W, M_r = self.instance.union[self.r[i]]
            if direct > M_r:
                alone.append(i)
        return alone

    def _arc_is_feasible(self, i: int, j: int) -> bool:
        """Quick filter: arc (i, j) is infeasible if even the earliest start at i
        cannot reach j within its time window."""
        if i == j:
            return False
        # Self-loop, return-to-start, leave-from-end are forbidden by construction.
        if i == self.depot_end or j == self.depot_start:
            return False
        # Cannot visit pickup of i then immediately depot end before delivery.
        # Cannot deliver before pickup of same passenger.
        if i in self.D and j == i - self.n:
            return False
        if i in self.P and j == i:
            return False
        if i == self.depot_start and j == self.depot_end:
            return False
        # Time-window arc filter.
        if i in self.P or i in self.D:
            earliest_arrival_at_j = self.e[i] + self.s[i] + self.tau[i, j]
            if earliest_arrival_at_j > self.l[j] + 1e-6:
                return False
        return True

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self) -> gp.Model:
        m = gp.Model("TD_HDARP")
        if not self.config.verbose:
            m.Params.OutputFlag = 0
        m.Params.TimeLimit = self.config.time_limit_s
        m.Params.MIPGap = self.config.mip_gap
        if self.config.threads:
            m.Params.Threads = self.config.threads

        # --- Variables -----------------------------------------------------
        x: dict[tuple[int, int, int], gp.Var] = {}
        for k in self.K:
            for i in self.V:
                for j in self.V:
                    if not self._arc_is_feasible(i, j):
                        continue
                    if (
                        self.config.eliminate_infeasible_arcs
                        and i not in (self.depot_start, self.depot_end)
                        and j not in (self.depot_start, self.depot_end)
                    ):
                        # Keep only feasible arcs
                        pass
                    x[i, j, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

        y = {k: m.addVar(vtype=GRB.BINARY, name=f"y_{k}") for k in self.K}

        B = {
            (i, k): m.addVar(lb=self.e[i], ub=self.l[i], name=f"B_{i}_{k}")
            for k in self.K
            for i in self.V
        }

        L = {
            (i, k): m.addVar(
                lb=0.0,
                ub=float(self.vehicles[k]["capacity"]),
                vtype=GRB.INTEGER,
                name=f"L_{i}_{k}",
            )
            for k in self.K
            for i in self.V
        }

        u = {(r, k): m.addVar(vtype=GRB.BINARY, name=f"u_{r}_{k}") for k in self.K for r in (1, 2, 3, 4)}

        # --- Objective -----------------------------------------------------
        m.setObjective(
            gp.quicksum(self.vehicles[k]["fixed_cost"] * y[k] for k in self.K)
            + gp.quicksum(
                self.cost_per_meter * self.dist[i, j] * x[i, j, k]
                for (i, j, k) in x
            ),
            GRB.MINIMIZE,
        )

        # --- (eq:r1) Each pickup served exactly once ----------------------
        for i in self.P:
            m.addConstr(
                gp.quicksum(
                    x[i, j, k]
                    for k in self.K
                    for j in self.V
                    if j != self.depot_start and (i, j, k) in x
                ) == 1,
                name=f"r1_serve_{i}",
            )

        # --- (eq:r2) Pickup and delivery in the same vehicle --------------
        for i in self.P:
            for k in self.K:
                m.addConstr(
                    gp.quicksum(x[i, j, k] for j in self.V if (i, j, k) in x)
                    - gp.quicksum(x[j, i + self.n, k] for j in self.V if (j, i + self.n, k) in x)
                    == 0,
                    name=f"r2_pair_{i}_{k}",
                )

        # --- (eq:r3) Vehicle activation: depot exit/entry equals y_k ------
        for k in self.K:
            m.addConstr(
                gp.quicksum(x[self.depot_start, j, k] for j in self.V if (self.depot_start, j, k) in x)
                == y[k],
                name=f"r3_start_{k}",
            )
            m.addConstr(
                gp.quicksum(x[i, self.depot_end, k] for i in self.V if (i, self.depot_end, k) in x)
                == y[k],
                name=f"r3_end_{k}",
            )

        # --- (eq:r4) Flow conservation at non-depot nodes -----------------
        for h in self.P + self.D:
            for k in self.K:
                m.addConstr(
                    gp.quicksum(x[i, h, k] for i in self.V if (i, h, k) in x)
                    - gp.quicksum(x[h, j, k] for j in self.V if (h, j, k) in x)
                    == 0,
                    name=f"r4_flow_{h}_{k}",
                )

        # --- (eq:r5) Time consistency along arcs --------------------------
        # Two-sided big-M to force equality when x=1. Per-arc tight Big-M:
        #   M_ij = l_i + s_i + tau_ij - e_j  (the maximum feasible slack)
        for (i, j, k), var in x.items():
            if i == self.depot_start or j == self.depot_end:
                continue
            tight_lb = self.l[i] + self.s[i] + self.tau[i, j] - self.e[j]
            tight_ub = self.l[j] - self.e[i] - self.s[i] - self.tau[i, j]
            if self.config.tight_per_arc_big_m:
                Mt_lb = max(0.0, min(self.config.big_m_time, tight_lb))
                Mt_ub = max(0.0, min(self.config.big_m_time, tight_ub))
            else:
                Mt_lb = self.config.big_m_time
                Mt_ub = self.config.big_m_time
            m.addConstr(
                B[j, k] >= B[i, k] + self.s[i] + self.tau[i, j] - Mt_lb * (1 - var),
                name=f"r5_time_lb_{i}_{j}_{k}",
            )
            m.addConstr(
                B[j, k] <= B[i, k] + self.s[i] + self.tau[i, j] + Mt_ub * (1 - var),
                name=f"r5_time_ub_{i}_{j}_{k}",
            )

        # --- (eq:r6) Pickup-delivery precedence ---------------------------
        for i in self.P:
            for k in self.K:
                m.addConstr(
                    B[i, k] + self.s[i] + self.tau[i, i + self.n] <= B[i + self.n, k],
                    name=f"r6_prec_{i}_{k}",
                )

        # --- (eq:r7) Time windows are encoded as bounds on B_ik (above).

        # --- (eq:r8) Capacity flow ---------------------------------------
        # Two-sided big-M to force equality when x=1; tight to capacity + 1.
        Ml = self.config.big_m_load
        for (i, j, k), var in x.items():
            if i == self.depot_start or j == self.depot_end:
                continue
            cap = float(self.vehicles[k]["capacity"])
            Ml_arc = min(Ml, cap + 1.0)
            m.addConstr(
                L[j, k] >= L[i, k] + self.q[j] - Ml_arc * (1 - var),
                name=f"r8_load_lb_{i}_{j}_{k}",
            )
            m.addConstr(
                L[j, k] <= L[i, k] + self.q[j] + Ml_arc * (1 - var),
                name=f"r8_load_ub_{i}_{j}_{k}",
            )

        # --- (eq:r9) Initial load is zero, depot end load is zero ---------
        for k in self.K:
            m.addConstr(L[self.depot_start, k] == 0, name=f"r9_start_load_{k}")

        # --- (eq:r10) Max ride time per category --------------------------
        for i in self.P:
            if i in self.alone_passengers:
                continue  # Relaxed: alone passengers may exceed M_r.
            _W, M_r = self.instance.union[self.r[i]]
            for k in self.K:
                m.addConstr(
                    B[i + self.n, k] - (B[i, k] + self.s[i]) <= M_r,
                    name=f"r10_ride_{i}_{k}",
                )

        # --- (eq:r11) Max wait/advance vs sigma_i (AM convention) ---------
        for i in self.P:
            sigma = float(self.passengers.loc[i, "sigma_i"])
            W_r, _M = self.instance.union[self.r[i]]
            for k in self.K:
                m.addConstr(
                    sigma - (B[i + self.n, k] + self.s[i + self.n]) <= W_r,
                    name=f"r11_wait_{i}_{k}",
                )

        # --- (eq:r12) Category indicator linking --------------------------
        for i in self.P:
            for k in self.K:
                m.addConstr(
                    u[self.r[i], k] >= gp.quicksum(x[i, j, k] for j in self.V if (i, j, k) in x),
                    name=f"r12_cat_{i}_{k}",
                )

        # --- (eq:r13) At most two categories per vehicle ------------------
        for k in self.K:
            m.addConstr(
                gp.quicksum(u[r, k] for r in (1, 2, 3, 4)) <= 2,
                name=f"r13_two_cats_{k}",
            )

        # --- (eq:r14) Only consecutive categories together ----------------
        for k in self.K:
            for r1 in (1, 2, 3, 4):
                for r2 in (1, 2, 3, 4):
                    if r1 < r2 and (r1, r2) not in self.cat_compat:
                        m.addConstr(u[r1, k] + u[r2, k] <= 1, name=f"r14_inc_{r1}_{r2}_{k}")

        # --- "Travel alone" rule for passengers with direct travel > M_r --
        for i in self.alone_passengers:
            for j in self.P:
                if j == i:
                    continue
                for k in self.K:
                    # If both i and j are picked up by the same vehicle, infeasible.
                    sum_xi = gp.quicksum(x[i, h, k] for h in self.V if (i, h, k) in x)
                    sum_xj = gp.quicksum(x[j, h, k] for h in self.V if (j, h, k) in x)
                    m.addConstr(sum_xi + sum_xj <= 1, name=f"alone_{i}_{j}_{k}")

        # --- Symmetry breaking: use vehicles of the same type in order ----
        if self.config.symmetry_breaking:
            for vtype in ("Common", "Large"):
                ks = [k for k in self.K if self.vehicles[k]["type"] == vtype]
                for a, b in zip(ks[:-1], ks[1:], strict=False):
                    m.addConstr(y[b] <= y[a], name=f"sym_{vtype}_{a}_{b}")

        m.update()
        self.model = m
        self.x = x
        self.y = y
        self.B = B
        self.L = L
        self.u = u

        if self.warm_start is not None:
            self._apply_warm_start(self.warm_start)

        return m

    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------
    def _apply_warm_start(self, sol: Solution) -> None:
        """Seed Gurobi with values from a previously computed Solution.

        Maps the routes' vehicle types onto the available pool slots in order
        (Common routes go to Common pool slots, same for Large). Routes that
        don't fit in the pool are silently skipped — Gurobi will still benefit
        from a partial incumbent.
        """
        assert self.x is not None and self.y is not None
        assert self.B is not None and self.L is not None and self.u is not None

        # Index of available pool slots per type
        common_slots = [k for k in self.K if self.vehicles[k]["type"] == "Common"]
        large_slots = [k for k in self.K if self.vehicles[k]["type"] == "Large"]
        common_idx = 0
        large_idx = 0
        skipped = 0

        for route in sol.routes:
            if route.vehicle_type == "Common":
                if common_idx >= len(common_slots):
                    skipped += 1
                    continue
                k = common_slots[common_idx]
                common_idx += 1
            else:
                if large_idx >= len(large_slots):
                    skipped += 1
                    continue
                k = large_slots[large_idx]
                large_idx += 1

            self.y[k].Start = 1.0

            # Set arc and time/load values
            seen_categories: set[int] = set()
            for a, b in zip(route.nodes[:-1], route.nodes[1:], strict=False):
                if (a, b, k) in self.x:
                    self.x[a, b, k].Start = 1.0
            for node, t in route.start_times.items():
                if (node, k) in self.B:
                    # Clamp into the variable's bounds before passing to Gurobi.
                    lb = float(self.B[node, k].LB)
                    ub = float(self.B[node, k].UB)
                    self.B[node, k].Start = max(lb, min(ub, float(t)))
            for node, ld in route.loads.items():
                if (node, k) in self.L:
                    self.L[node, k].Start = float(ld)
            for node in route.nodes:
                if 1 <= node <= self.n:
                    seen_categories.add(self.r[node])
            for r_cat in seen_categories:
                if (r_cat, k) in self.u:
                    self.u[r_cat, k].Start = 1.0

        if skipped > 0:
            warnings.warn(
                f"Warm-start: {skipped} route(s) didn't fit in the MILP fleet pool and were skipped",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Solve / extract
    # ------------------------------------------------------------------
    def solve(self) -> Solution:
        if self.model is None:
            self.build()
        assert self.model is not None  # for type checker

        t0 = time.perf_counter()
        self.model.optimize()
        elapsed = time.perf_counter() - t0

        status = self.model.Status
        has_solution = self.model.SolCount > 0
        if not has_solution:
            return Solution(
                instance_label=self.instance.label,
                n_passengers=self.n,
                routes=[],
                total_cost=float("inf"),
                fixed_cost=0.0,
                variable_cost=0.0,
                is_feasible=False,
                metadata={
                    "solver": "gurobi",
                    "status": int(status),
                    "runtime_s": elapsed,
                    "n_vehicles_pool": len(self.K),
                    "alone_passengers": list(self.alone_passengers),
                },
            )

        return self._extract_solution(elapsed, status)

    def _extract_solution(self, elapsed: float, status: int) -> Solution:
        assert self.model is not None
        assert self.x is not None and self.y is not None
        assert self.B is not None and self.L is not None

        # Trace each active route from the depot.
        routes: list[Route] = []
        fixed_cost_total = 0.0
        variable_cost_total = 0.0

        for k in self.K:
            if self.y[k].X < 0.5:
                continue
            veh = self.vehicles[k]
            fixed_cost_total += veh["fixed_cost"]

            # Walk the route from depot_start to depot_end.
            current = self.depot_start
            visited = [current]
            start_times = {current: float(self.B[current, k].X)}
            loads = {current: int(round(self.L[current, k].X))}
            route_dist = 0.0
            safety = 4 * self.n + 4  # guard against cycles in degenerate solutions
            while current != self.depot_end and safety > 0:
                next_node = None
                for j in self.V:
                    if (current, j, k) in self.x and self.x[current, j, k].X > 0.5:
                        next_node = j
                        break
                if next_node is None:
                    break
                route_dist += float(self.dist[current, next_node])
                current = next_node
                visited.append(current)
                start_times[current] = float(self.B[current, k].X)
                loads[current] = int(round(self.L[current, k].X))
                safety -= 1

            variable_cost_total += self.cost_per_meter * route_dist
            routes.append(
                Route(
                    vehicle_id=k,
                    vehicle_type=veh["type"],
                    nodes=visited,
                    start_times=start_times,
                    loads=loads,
                )
            )

        total_cost = float(self.model.ObjVal)
        feasible = status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED)

        try:
            mip_gap = float(self.model.MIPGap)
        except gp.GurobiError:
            mip_gap = float("nan")

        return Solution(
            instance_label=self.instance.label,
            n_passengers=self.n,
            routes=routes,
            total_cost=total_cost,
            fixed_cost=float(fixed_cost_total),
            variable_cost=float(variable_cost_total),
            is_feasible=feasible,
            metadata={
                "solver": "gurobi",
                "status": int(status),
                "runtime_s": float(elapsed),
                "mip_gap": mip_gap,
                "n_vehicles_pool": len(self.K),
                "n_vehicles_used": len(routes),
                "alone_passengers": list(self.alone_passengers),
                "time_dependent": self.config.time_dependent,
                "fleet_size": {"common": self.fleet_size.common, "large": self.fleet_size.large},
                "relaxed_pickup_passengers": list(self.relaxed_pickup_passengers),
            },
        )
