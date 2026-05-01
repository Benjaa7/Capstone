"""Run the MILP solver on the 30-passenger Caso Base.

Persists the resulting :class:`Solution` to ``results/milp_caso_base.json``,
which the ALNS engine will use as oracle for validation.

Usage:
    python scripts/run_milp_caso_base.py [--n 30] [--seed 42] [--time-limit 43200]
        [--mip-gap 0.01] [--common 18] [--large 6] [--time-dependent]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Make ``src/`` importable when this script is run directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.miope import MiopeSolver  # noqa: E402
from src.data import Instance  # noqa: E402
from src.data.case_base import select_case_base  # noqa: E402
from src.milp.td_hdarp import FleetSize, MilpConfig, TDHDARPModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the MILP on the Caso Base")
    p.add_argument("--instance", default="AM", choices=["AM", "PM"])
    p.add_argument("--n", type=int, default=30, help="Caso Base size (passengers)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-limit", type=int, default=43_200, help="Solver time limit (s)")
    p.add_argument("--mip-gap", type=float, default=0.01)
    p.add_argument("--threads", type=int, default=0, help="0 = use all cores")
    p.add_argument("--common", type=int, default=18, help="Common vehicles in fleet pool")
    p.add_argument("--large", type=int, default=6, help="Large vehicles in fleet pool")
    p.add_argument("--time-dependent", action="store_true")
    p.add_argument("--no-warm-start", action="store_true", help="Skip miope warm-start")
    p.add_argument("--quiet", action="store_true", help="Suppress Gurobi log")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/milp_caso_base.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("default")  # show user warnings (e.g. relaxed e_i)

    print(f"Loading {args.instance} instance...")
    full = Instance.from_files(instance=args.instance)
    print(f"  full instance: {full.n_passengers()} passengers")

    print(f"Sampling Caso Base (n={args.n}, seed={args.seed})...")
    cb = select_case_base(full, n=args.n, seed=args.seed)
    counts = cb.category_counts()
    print(f"  category counts: {counts.to_dict()}")

    warm_sol = None
    if not args.no_warm_start:
        print("Computing miope warm-start...")
        warm_sol = MiopeSolver(cb).solve()
        print(
            f"  miope cost: {warm_sol.total_cost:.0f} CLP, "
            f"vehicles used: {warm_sol.n_vehicles_used}"
        )

    print("Building MILP...")
    model = TDHDARPModel(
        cb,
        fleet_size=FleetSize(common=args.common, large=args.large),
        config=MilpConfig(
            time_dependent=args.time_dependent,
            time_limit_s=args.time_limit,
            mip_gap=args.mip_gap,
            threads=args.threads,
            verbose=not args.quiet,
        ),
        warm_start=warm_sol,
    )
    if model.relaxed_pickup_passengers:
        print(
            f"  WARNING: relaxed pickup_from for {len(model.relaxed_pickup_passengers)} "
            f"passengers: {model.relaxed_pickup_passengers}"
        )
    if model.alone_passengers:
        print(f"  alone-required passengers: {model.alone_passengers}")

    print("Solving...")
    sol = model.solve()

    print("=" * 60)
    print(
        f"Status: {sol.metadata['status']}  Cost: {sol.total_cost:.0f} CLP  "
        f"Vehicles used: {sol.n_vehicles_used}/{len(model.K)}"
    )
    print(f"  Fixed cost:    {sol.fixed_cost:>12.0f} CLP")
    print(f"  Variable cost: {sol.variable_cost:>12.0f} CLP")
    print(f"  MIP gap:       {sol.metadata.get('mip_gap', float('nan')):>12.4f}")
    print(f"  Runtime:       {sol.metadata['runtime_s']:>12.1f} s")
    print(f"  Feasible:      {sol.is_feasible}")
    print("=" * 60)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sol.save(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
