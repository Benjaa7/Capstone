"""Tests for the myopic baseline solver."""

from __future__ import annotations

import pytest

from src.baselines.miope import MiopeSolver
from src.data import Instance
from src.data.case_base import select_case_base


@pytest.fixture(scope="module")
def small_instance() -> Instance:
    full = Instance.from_files(instance="AM")
    return select_case_base(full, n=8, seed=7)


def test_miope_returns_feasible_solution(small_instance: Instance) -> None:
    sol = MiopeSolver(small_instance).solve()

    assert sol.is_feasible
    assert sol.n_passengers == 8
    assert sol.n_vehicles_used >= 1
    assert sol.total_cost > 0

    # All pickups and deliveries appear exactly once across routes.
    pickup_nodes = list(range(1, 9))
    delivery_nodes = list(range(9, 17))
    visited_pickups: list[int] = []
    visited_deliveries: list[int] = []
    for r in sol.routes:
        for n in r.nodes:
            if n in pickup_nodes:
                visited_pickups.append(n)
            elif n in delivery_nodes:
                visited_deliveries.append(n)
    assert sorted(visited_pickups) == pickup_nodes
    assert sorted(visited_deliveries) == delivery_nodes


def test_miope_respects_capacity(small_instance: Instance) -> None:
    sol = MiopeSolver(small_instance).solve()
    for r in sol.routes:
        cap = small_instance.vehicle_capacity(r.vehicle_type)
        assert max(r.loads.values()) <= cap


def test_miope_respects_sindical(small_instance: Instance) -> None:
    sol = MiopeSolver(small_instance).solve()
    for r in sol.routes:
        cats = set()
        for node in r.nodes:
            if 1 <= node <= small_instance.n_passengers():
                priority = int(small_instance.passengers.set_index("id").loc[node, "priority"])
                cats.add(priority)
        assert len(cats) <= 2, f"Route {r.vehicle_id} has {len(cats)} categories: {cats}"
        if len(cats) == 2:
            a, b = sorted(cats)
            assert abs(a - b) == 1, f"Non-consecutive categories in route {r.vehicle_id}: {cats}"


def test_miope_pickup_before_delivery(small_instance: Instance) -> None:
    sol = MiopeSolver(small_instance).solve()
    n = small_instance.n_passengers()
    for r in sol.routes:
        seen_pickups: set[int] = set()
        for node in r.nodes:
            if 1 <= node <= n:
                seen_pickups.add(node)
            elif n + 1 <= node <= 2 * n:
                paired = node - n
                assert paired in seen_pickups, (
                    f"Delivery {node} before pickup {paired} in route {r.vehicle_id}"
                )
