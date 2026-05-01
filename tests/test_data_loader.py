"""Sanity checks for the data layer.

Validates that the AM instance loads with the expected dimensions, units, and
that the Case Base subsampler returns a stratified subset that respects the
distribution of categories.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import Instance
from src.data.case_base import select_case_base


@pytest.fixture(scope="module")
def am_instance() -> Instance:
    return Instance.from_files(instance="AM")


def test_passengers_count_and_columns(am_instance: Instance) -> None:
    df = am_instance.passengers
    assert len(df) == 620
    expected_cols = {
        "id",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "e_i",
        "sigma_i",
        "priority",
    }
    assert expected_cols.issubset(df.columns)
    # Times are in seconds since midnight: 04:15 -> 4*3600 + 15*60 = 15300 (lower bound).
    assert df["e_i"].min() >= 0
    assert df["sigma_i"].max() <= 24 * 3600


def test_categories_in_range(am_instance: Instance) -> None:
    cats = set(am_instance.passengers["priority"].unique())
    assert cats.issubset({1, 2, 3, 4})


def test_vehicles_two_types(am_instance: Instance) -> None:
    vdf = am_instance.vehicles
    assert set(vdf["type"]) == {"Common", "Large"}
    assert am_instance.vehicle_capacity("Common") == 3
    assert am_instance.vehicle_capacity("Large") == 6
    assert am_instance.vehicle_fixed_cost("Common") == 2200.0
    assert am_instance.vehicle_fixed_cost("Large") == 4500.0


def test_union_constraints_priorities(am_instance: Instance) -> None:
    # Four categories, priority 4 is the strictest at 5 min wait.
    assert set(am_instance.union.keys()) == {1, 2, 3, 4}
    W4, M4 = am_instance.union[4]
    assert W4 == 5 * 60
    assert M4 == 50 * 60


def test_others_units(am_instance: Instance) -> None:
    o = am_instance.others
    assert o["cost_per_meter"] == pytest.approx(800.0 / 1000.0)  # CLP/km -> CLP/m
    assert o["pickup_duration"] == 3 * 60
    assert o["delivery_duration"] == 1 * 60


def test_time_matrices_loaded(am_instance: Instance) -> None:
    # Three matrices for AM (04, 05, 06).
    assert set(am_instance.time_matrices.keys()) == {4, 5, 6}
    for h, df in am_instance.time_matrices.items():
        assert len(df) == 640 * 640
        assert {"orig_lat", "orig_lon", "dest_lat", "dest_lon", "time", "distance"}.issubset(df.columns)


def test_tau_lookup_static_and_dynamic(am_instance: Instance) -> None:
    # Pick the first row of the 04:00 matrix, look it up in static and dynamic mode.
    first = am_instance.time_matrices[4].iloc[0]
    orig = (first["orig_lat"], first["orig_lon"])
    dest = (first["dest_lat"], first["dest_lon"])
    t_static = am_instance.tau(orig, dest)
    t_dynamic = am_instance.tau(orig, dest, t=4 * 3600)
    assert t_dynamic == pytest.approx(first["time"])
    assert t_static > 0
    # Static = average of the same (orig, dest) pair across the three hourly matrices.
    expected_times = []
    for h, mat in am_instance.time_matrices.items():
        match = mat[
            (mat["orig_lat"] == orig[0])
            & (mat["orig_lon"] == orig[1])
            & (mat["dest_lat"] == dest[0])
            & (mat["dest_lon"] == dest[1])
        ]
        assert len(match) == 1, f"Pair not unique in hour {h}"
        expected_times.append(float(match["time"].iat[0]))
    assert t_static == pytest.approx(sum(expected_times) / len(expected_times))


def test_case_base_size_and_stratification(am_instance: Instance) -> None:
    cb = select_case_base(am_instance, n=30, seed=42)
    assert cb.n_passengers() == 30
    counts = cb.category_counts()
    full_counts = am_instance.category_counts()
    # All four categories represented unless one is too rare; check totals add up.
    assert counts.sum() == 30
    # Proportions roughly match the original (within 10 percentage points each).
    full_prop = full_counts / full_counts.sum()
    cb_prop = counts / counts.sum()
    diff = (full_prop - cb_prop).abs()
    assert diff.max() < 0.1


def test_case_base_reproducible(am_instance: Instance) -> None:
    cb1 = select_case_base(am_instance, n=30, seed=42)
    cb2 = select_case_base(am_instance, n=30, seed=42)
    pd.testing.assert_frame_equal(cb1.passengers, cb2.passengers)
