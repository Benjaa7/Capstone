"""Solution data structures shared by the MILP and ALNS solvers.

A :class:`Solution` represents a complete assignment of passengers to
vehicles, with the corresponding routes, service start times and loads. Both
the MILP solver (:mod:`src.milp.td_hdarp`) and the ALNS engine
(:mod:`src.alns.engine`) produce instances of this class so KPIs and
visualizations can be computed uniformly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Route:
    """A single vehicle route in a TD-HDARP solution.

    ``nodes`` uses the MILP node indexing convention:
      0 = start depot, 1..n = pickups, n+1..2n = deliveries, 2n+1 = end depot.
    """

    vehicle_id: int
    vehicle_type: str  # "Common" or "Large"
    nodes: list[int]
    start_times: dict[int, float] = field(default_factory=dict)  # node -> B_ik (s)
    loads: dict[int, int] = field(default_factory=dict)  # node -> L_ik

    @property
    def n_pickups(self) -> int:
        return sum(1 for n in self.nodes if n != 0 and n in self.start_times and self.loads.get(n, 0) > 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "vehicle_type": self.vehicle_type,
            "nodes": list(self.nodes),
            "start_times": {str(k): v for k, v in self.start_times.items()},
            "loads": {str(k): v for k, v in self.loads.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Route":
        return cls(
            vehicle_id=int(d["vehicle_id"]),
            vehicle_type=str(d["vehicle_type"]),
            nodes=[int(x) for x in d["nodes"]],
            start_times={int(k): float(v) for k, v in d.get("start_times", {}).items()},
            loads={int(k): int(v) for k, v in d.get("loads", {}).items()},
        )


@dataclass
class Solution:
    """Container for a complete solution, plus solver metadata."""

    instance_label: str
    n_passengers: int
    routes: list[Route]
    total_cost: float
    fixed_cost: float
    variable_cost: float
    is_feasible: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_vehicles_used(self) -> int:
        return len(self.routes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_label": self.instance_label,
            "n_passengers": self.n_passengers,
            "routes": [r.to_dict() for r in self.routes],
            "total_cost": float(self.total_cost),
            "fixed_cost": float(self.fixed_cost),
            "variable_cost": float(self.variable_cost),
            "is_feasible": bool(self.is_feasible),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Solution":
        return cls(
            instance_label=str(d["instance_label"]),
            n_passengers=int(d["n_passengers"]),
            routes=[Route.from_dict(r) for r in d["routes"]],
            total_cost=float(d["total_cost"]),
            fixed_cost=float(d["fixed_cost"]),
            variable_cost=float(d["variable_cost"]),
            is_feasible=bool(d["is_feasible"]),
            metadata=dict(d.get("metadata", {})),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Solution":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
