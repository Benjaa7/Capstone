"""Tests for the ALNS evaluation function."""

from __future__ import annotations

import pytest

from src.alns.evaluation import (
    Violations,
    WeightTracker,
    evaluate_route,
    evaluate_solution,
    penalised_score,
)
from src.baselines.miope import MiopeSolver
from src.data import Instance
from src.data.case_base import select_case_base


@pytest.fixture(scope="module")
def small_instance() -> Instance:
    full = Instance.from_files(instance="AM")
    return select_case_base(full, n=8, seed=7)


def test_evaluator_marks_miope_solution_feasible(small_instance: Instance) -> None:
    sol = MiopeSolver(small_instance).solve()
    cost, v = evaluate_solution(sol, small_instance)
    assert v.is_clean(), f"Miope solution should be feasible, got: {v.to_dict()}"
    # Cost from evaluator should match the miope's reported cost (within float tolerance).
    assert abs(cost - sol.total_cost) < 1.0


def test_violations_addition() -> None:
    a = Violations(q=1, r=2, d=3, t=4, u=5)
    b = Violations(q=10, r=20, d=30, t=40, u=50)
    c = a + b
    assert c.q == 11 and c.r == 22 and c.d == 33 and c.t == 44 and c.u == 55


def test_violations_clean_threshold() -> None:
    assert Violations().is_clean()
    assert not Violations(q=0.001).is_clean(eps=1e-6)
    assert Violations(q=1e-9).is_clean(eps=1e-6)


def test_weight_tracker_increases_when_violated() -> None:
    w = WeightTracker()
    initial_alpha = w.alpha
    w.update(Violations(q=10))
    assert w.alpha > initial_alpha


def test_weight_tracker_decreases_when_clean() -> None:
    import numpy as np
    w = WeightTracker(rng=np.random.default_rng(0))
    # First raise the weight a bit
    for _ in range(3):
        w.update(Violations(q=10))
    raised = w.alpha
    # Now go clean
    for _ in range(3):
        w.update(Violations())
    assert w.alpha < raised


def test_penalised_score_matches_cost_when_clean() -> None:
    w = WeightTracker()
    score = penalised_score(1000.0, Violations(), w)
    assert score == 1000.0


def test_penalised_score_includes_all_terms() -> None:
    w = WeightTracker(alpha=2.0, beta=3.0, gamma=4.0, epsilon=5.0, phi=6.0)
    v = Violations(q=1, r=1, d=1, t=1, u=1)
    expected = 100.0 + 2 + 3 + 4 + 5 + 6
    assert penalised_score(100.0, v, w) == expected


def test_evaluate_route_assigns_correct_loads(small_instance: Instance) -> None:
    sol = MiopeSolver(small_instance).solve()
    for route in sol.routes:
        ev = evaluate_route(route, small_instance)
        # Loads at every step are in [0, capacity]
        cap = small_instance.vehicle_capacity(route.vehicle_type)
        assert all(0 <= load <= cap for load in ev.loads.values())
        # Pickup increments by 1, delivery decrements by 1; final load (depot end) is 0
        assert ev.loads.get(small_instance.n_passengers() * 2 + 1, 0) == 0
