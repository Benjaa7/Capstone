"""MILP smoke test on a tiny instance.

Builds an 8-passenger sub-instance (2 of each priority where possible) and
verifies that Gurobi finds a feasible, near-optimal solution within a short
time limit. The point is to catch construction bugs in the MILP, not to
benchmark the solver.

Skipped automatically if a valid Gurobi license is not available.
"""

from __future__ import annotations

import pytest

from src.data import Instance
from src.data.case_base import select_case_base


@pytest.fixture(scope="module")
def gurobi_available() -> bool:
    try:
        import gurobipy as gp

        m = gp.Model("license_check")
        m.Params.OutputFlag = 0
        m.addVar()
        m.optimize()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def small_instance() -> Instance:
    full = Instance.from_files(instance="AM")
    return select_case_base(full, n=8, seed=7)


def test_milp_builds_and_solves(gurobi_available: bool, small_instance: Instance) -> None:
    if not gurobi_available:
        pytest.skip("Gurobi license not available")
    from src.milp.td_hdarp import FleetSize, MilpConfig, TDHDARPModel

    model = TDHDARPModel(
        small_instance,
        fleet_size=FleetSize(common=4, large=2),
        config=MilpConfig(
            time_limit_s=180,
            mip_gap=0.05,
            verbose=False,
            symmetry_breaking=True,
        ),
    )
    sol = model.solve()

    # Basic sanity checks
    assert sol.n_passengers == 8
    assert sol.is_feasible, f"Solver returned infeasible: status={sol.metadata.get('status')}"
    assert sol.total_cost > 0
    assert sol.n_vehicles_used >= 1
    assert sol.n_vehicles_used <= 8

    # All passengers must be picked up and delivered exactly once.
    pickup_nodes = list(range(1, 9))
    delivery_nodes = list(range(9, 17))
    visited_pickups: list[int] = []
    visited_deliveries: list[int] = []
    for route in sol.routes:
        for node in route.nodes:
            if node in pickup_nodes:
                visited_pickups.append(node)
            elif node in delivery_nodes:
                visited_deliveries.append(node)
    assert sorted(visited_pickups) == pickup_nodes, f"missing pickups: {visited_pickups}"
    assert sorted(visited_deliveries) == delivery_nodes, f"missing deliveries: {visited_deliveries}"

    # Per-route feasibility: pickup must precede its delivery in the same vehicle.
    for route in sol.routes:
        seen_pickups: set[int] = set()
        for node in route.nodes:
            if node in pickup_nodes:
                seen_pickups.add(node)
            elif node in delivery_nodes:
                paired_pickup = node - small_instance.n_passengers()
                assert paired_pickup in seen_pickups, (
                    f"Delivery {node} appears before pickup {paired_pickup} in route {route.vehicle_id}"
                )

    # Capacity check
    for route in sol.routes:
        max_load = max(route.loads.values()) if route.loads else 0
        cap = (
            small_instance.vehicle_capacity("Common")
            if route.vehicle_type == "Common"
            else small_instance.vehicle_capacity("Large")
        )
        assert max_load <= cap, f"Capacity violation in route {route.vehicle_id}: {max_load} > {cap}"
