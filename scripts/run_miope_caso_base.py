"""Run the myopic baseline on the Caso Base."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.miope import MiopeSolver  # noqa: E402
from src.data import Instance  # noqa: E402
from src.data.case_base import select_case_base  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="AM", choices=["AM", "PM"])
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/miope_caso_base.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    full = Instance.from_files(instance=args.instance)
    cb = select_case_base(full, n=args.n, seed=args.seed)
    print(f"Caso Base: n={cb.n_passengers()}  cats={cb.category_counts().to_dict()}")

    sol = MiopeSolver(cb).solve()
    print("=" * 60)
    print(f"Cost:           {sol.total_cost:>12.0f} CLP")
    print(f"  Fixed:        {sol.fixed_cost:>12.0f} CLP")
    print(f"  Variable:     {sol.variable_cost:>12.0f} CLP")
    print(f"Vehicles used:  {sol.n_vehicles_used}")
    by_type: dict[str, int] = {}
    for r in sol.routes:
        by_type[r.vehicle_type] = by_type.get(r.vehicle_type, 0) + 1
    print(f"  by type:      {by_type}")
    print("=" * 60)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sol.save(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
