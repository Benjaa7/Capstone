"""High-level container for a TD-HDARP instance.

The :class:`Instance` class wraps the four pieces returned by the loaders
(passengers, vehicles, union constraints, others) plus the hourly travel-time
matrices, and exposes a few convenience accessors used by the MILP and ALNS
implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import (
    DEFAULT_DATA_DIR,
    INSTANCE_FILES,
    load_others,
    load_passengers,
    load_time_matrices,
    load_union_constraints,
    load_vehicles,
)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def _round_coord(x: float, digits: int = 6) -> float:
    """Snap coordinates to the precision used in the source CSVs."""
    return round(float(x), digits)


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------
@dataclass
class Instance:
    """A complete operational instance (AM or PM).

    The travel-time and distance matrices are stored as nested dicts keyed by
    rounded ``(lat, lon)`` tuples for O(1) lookup. The same ``Instance`` can be
    used in time-dependent (``tau(i, j, t)``) or static mode (``tau(i, j)``)
    depending on whether a departure time ``t`` is supplied.
    """

    label: str
    passengers: pd.DataFrame
    vehicles: pd.DataFrame
    union: dict[int, tuple[float, float]]
    others: dict[str, float | str]
    time_matrices: dict[int, pd.DataFrame] = field(repr=False)

    # Internal lookups built lazily on first access.
    _time_lookup: dict[int, dict[tuple, tuple[float, float]]] = field(default_factory=dict, repr=False)
    _avg_lookup: dict[tuple, tuple[float, float]] = field(default_factory=dict, repr=False)
    _pax_cache: dict[int, dict] = field(default_factory=dict, repr=False)
    _eff_pickup_cache: dict[int, tuple[float, float]] = field(default_factory=dict, repr=False)
    _vehicle_cache: dict[str, dict] = field(default_factory=dict, repr=False)
    _solo_cache: set[int] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        instance: str = "AM",
        data_dir: Path | None = None,
        hours: tuple[int, ...] | None = None,
    ) -> "Instance":
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        if instance not in INSTANCE_FILES:
            raise ValueError(f"Unknown instance: {instance!r}")
        xlsx_path = data_dir / INSTANCE_FILES[instance]["xlsx"]

        passengers = load_passengers(xlsx_path)
        vehicles = load_vehicles(xlsx_path)
        union = load_union_constraints(xlsx_path)
        others = load_others(xlsx_path)
        time_matrices = load_time_matrices(data_dir, instance=instance, hours=hours)

        return cls(
            label=instance,
            passengers=passengers,
            vehicles=vehicles,
            union=union,
            others=others,
            time_matrices=time_matrices,
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def _build_lookup_for_hour(self, hour: int) -> dict[tuple, tuple[float, float]]:
        df = self.time_matrices[hour]
        out: dict[tuple, tuple[float, float]] = {}
        for row in df.itertuples(index=False):
            key = (
                _round_coord(row.orig_lat),
                _round_coord(row.orig_lon),
                _round_coord(row.dest_lat),
                _round_coord(row.dest_lon),
            )
            out[key] = (float(row.time), float(row.distance))
        return out

    def _ensure_hour_lookup(self, hour: int) -> dict[tuple, tuple[float, float]]:
        if hour not in self._time_lookup:
            self._time_lookup[hour] = self._build_lookup_for_hour(hour)
        return self._time_lookup[hour]

    def _ensure_avg_lookup(self) -> dict[tuple, tuple[float, float]]:
        if self._avg_lookup:
            return self._avg_lookup

        # Compute the per-pair average across all hourly matrices.
        accum: dict[tuple, list[tuple[float, float]]] = {}
        for hour in self.time_matrices:
            for key, (t, d) in self._ensure_hour_lookup(hour).items():
                accum.setdefault(key, []).append((t, d))
        self._avg_lookup = {
            key: (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])))
            for key, vals in accum.items()
        }
        return self._avg_lookup

    @staticmethod
    def _hour_from_seconds(t_seconds: float, available: list[int]) -> int:
        """Return the hour from ``available`` whose value is closest (mod 24) to ``t_seconds``."""
        h = int(t_seconds // 3600) % 24
        if h in available:
            return h
        # Fall back to the nearest available hour with circular distance.
        return min(available, key=lambda x: min((x - h) % 24, (h - x) % 24))

    def tau(
        self,
        orig: tuple[float, float],
        dest: tuple[float, float],
        t: float | None = None,
    ) -> float:
        """Return travel time (s) from ``orig`` to ``dest`` for a departure at ``t``.

        If ``t`` is None, the cross-hour mean is returned (static mode).
        """
        return self._lookup(orig, dest, t)[0]

    def distance(
        self,
        orig: tuple[float, float],
        dest: tuple[float, float],
        t: float | None = None,
    ) -> float:
        """Return travel distance (m) from ``orig`` to ``dest``."""
        return self._lookup(orig, dest, t)[1]

    def _lookup(
        self,
        orig: tuple[float, float],
        dest: tuple[float, float],
        t: float | None,
    ) -> tuple[float, float]:
        key = (
            _round_coord(orig[0]),
            _round_coord(orig[1]),
            _round_coord(dest[0]),
            _round_coord(dest[1]),
        )
        if t is None:
            data = self._ensure_avg_lookup()
        else:
            hour = self._hour_from_seconds(t, sorted(self.time_matrices.keys()))
            data = self._ensure_hour_lookup(hour)
        if key not in data:
            raise KeyError(f"Pair not found in matrix: {key}")
        return data[key]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.passengers)

    def n_passengers(self) -> int:
        return len(self.passengers)

    def category_counts(self) -> pd.Series:
        return self.passengers["priority"].value_counts().sort_index()

    def _ensure_vehicle_cache(self) -> None:
        if self._vehicle_cache:
            return
        for _, row in self.vehicles.iterrows():
            self._vehicle_cache[str(row["type"])] = {
                "capacity": int(row["capacity"]),
                "fixed_cost": float(row["fixed_cost"]),
                "stock": int(row["stock"]),
            }

    def vehicle_capacity(self, vehicle_type: str) -> int:
        self._ensure_vehicle_cache()
        return self._vehicle_cache[vehicle_type]["capacity"]

    def vehicle_fixed_cost(self, vehicle_type: str) -> float:
        self._ensure_vehicle_cache()
        return self._vehicle_cache[vehicle_type]["fixed_cost"]

    def vehicle_stock(self, vehicle_type: str) -> int:
        self._ensure_vehicle_cache()
        return self._vehicle_cache[vehicle_type]["stock"]

    # ------------------------------------------------------------------
    # Fast passenger lookup (used heavily by ALNS evaluation loops)
    # ------------------------------------------------------------------
    def pax_dict(self, pid: int) -> dict:
        """Return a plain ``dict`` view of passenger ``pid``'s row, cached.

        This is much faster than ``passengers.set_index('id').loc[pid]``
        when called repeatedly inside hot inner loops.
        """
        if not self._pax_cache:
            for _, row in self.passengers.iterrows():
                self._pax_cache[int(row["id"])] = {
                    "id": int(row["id"]),
                    "pickup_lat": float(row["pickup_lat"]),
                    "pickup_lon": float(row["pickup_lon"]),
                    "delivery_lat": float(row["delivery_lat"]),
                    "delivery_lon": float(row["delivery_lon"]),
                    "e_i": float(row["e_i"]),
                    "sigma_i": float(row["sigma_i"]),
                    "priority": int(row["priority"]),
                }
        return self._pax_cache[int(pid)]

    # ------------------------------------------------------------------
    # Effective time windows (with pickup_from relaxation)
    # ------------------------------------------------------------------
    def effective_pickup_window(self, pid: int) -> tuple[float, float]:
        """Return ``(e_i_eff, l_i)`` for passenger ``pid``.

        ``e_i_eff = min(pickup_from, sigma - W_r - tau_direct - s_p - s_d)``
        so that direct travel from the relaxed earliest pickup arrives within
        the ``W_r`` advance window. ``l_i`` is the unmodified ``sigma_i``.
        Cached on first access — this is called heavily by the ALNS.
        """
        pid = int(pid)
        if pid in self._eff_pickup_cache:
            return self._eff_pickup_cache[pid]
        info = self.pax_dict(pid)
        sigma = info["sigma_i"]
        e_raw = info["e_i"]
        priority = info["priority"]
        W_r, _M_r = self.union[priority]
        co_p = (info["pickup_lat"], info["pickup_lon"])
        co_d = (info["delivery_lat"], info["delivery_lon"])
        tau_direct = self.tau(co_p, co_d)
        s_p = float(self.others["pickup_duration"])
        s_d = float(self.others["delivery_duration"])
        feasibility = max(0.0, sigma - W_r - tau_direct - s_p - s_d)
        e_eff = min(e_raw, feasibility) if e_raw > feasibility else e_raw
        result = (e_eff, sigma)
        self._eff_pickup_cache[pid] = result
        return result

    def is_pickup_relaxed(self, pid: int) -> bool:
        """True if ``effective_pickup_window`` returns a value below ``pickup_from``."""
        e_eff, _ = self.effective_pickup_window(pid)
        e_raw = self.pax_dict(int(pid))["e_i"]
        return e_eff < e_raw - 1.0

    @property
    def solo_passenger_ids(self) -> set[int]:
        """Set of passenger ids whose direct travel time exceeds their M_r limit.

        These passengers must travel alone (load ≤ 1 between their pickup and
        delivery) per the MILP solo-trip clause. Computed once and cached.
        """
        if self._solo_cache is not None:
            return self._solo_cache
        solo: set[int] = set()
        for _, row in self.passengers.iterrows():
            pid = int(row["id"])
            info = self.pax_dict(pid)
            _, M_r = self.union[info["priority"]]
            co_p = (info["pickup_lat"], info["pickup_lon"])
            co_d = (info["delivery_lat"], info["delivery_lon"])
            if self.tau(co_p, co_d) > M_r:
                solo.add(pid)
        self._solo_cache = solo
        return self._solo_cache
