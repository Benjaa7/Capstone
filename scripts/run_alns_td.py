"""Run ALNS on the full AM instance with Time-Dependent (XGBoost) travel times.

Compares cost vs the static run stored in results/alns_caso_base.json.
Usage:
    python scripts/run_alns_td.py [--n N] [--seed S] [--time-limit T]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.alns.engine import ALNSConfig, run_alns  # noqa: E402
from src.baselines.miope import MiopeSolver  # noqa: E402
from src.data import Instance  # noqa: E402
from src.data.case_base import select_case_base  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=620, help="Number of passengers (0=full)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-limit", type=int, default=3600)
    p.add_argument("--no-td", action="store_true", help="Skip XGBoost (static baseline)")
    args = p.parse_args()

    print("Loading AM instance...")
    full = Instance.from_files(instance="AM")
    if args.n < len(full.passengers):
        inst = select_case_base(full, n=args.n, seed=args.seed)
    else:
        inst = full
    print(f"  passengers: {inst.n_passengers()}  cats: {inst.category_counts().to_dict()}")

    # 1. Warm static caches
    print("Warming caches...")
    inst._ensure_avg_lookup()
    for _, row in inst.passengers.iterrows():
        inst.effective_pickup_window(int(row["id"]))
    inst._ensure_vehicle_cache()

    # 2. Load XGBoost TD tables
    if not args.no_td:
        print("Pre-computing XGBoost TD lookup tables...")
        inst.use_xgboost_batch(verbose=True)
        mode = "TD (XGBoost)"
    else:
        mode = "Static (matrix avg)"

    print(f"\nMode: {mode}")
    print("=" * 60)

    # 3. Miope warm-start
    print("Computing miope warm-start...")
    miope_sol = MiopeSolver(inst).solve()
    print(f"  miope cost: {miope_sol.total_cost:,.0f} CLP  vehicles: {miope_sol.n_vehicles_used}")

    # 4. ALNS
    print("Running ALNS...")
    result = run_alns(
        inst,
        ALNSConfig(
            max_iter=100_000,
            t_tot=float(args.time_limit),
            t_last=500,
            seed=args.seed,
            verbose=True,
            log_every=100,
            initial=miope_sol,
        ),
    )

    # 5. Results
    improvement = (miope_sol.total_cost - result.best_score) / miope_sol.total_cost * 100
    print()
    print("=" * 60)
    print(f"Mode:       {mode}")
    print(f"MIOPE:      {miope_sol.total_cost:>12,.0f} CLP  ({miope_sol.n_vehicles_used} vehicles)")
    print(f"ALNS best:  {result.best_score:>12,.0f} CLP  ({result.best.n_vehicles_used} vehicles)")
    print(f"Improvement: {improvement:+.2f}%")
    print(f"Iterations:  {result.iterations}")
    print(f"Runtime:     {result.runtime_s:.1f}s")
    print("=" * 60)

    # 6. Save
    tag = "td" if not args.no_td else "static"
    out_path = Path("results") / f"alns_{tag}_n{inst.n_passengers()}.json"
    out_path.parent.mkdir(exist_ok=True)
    result.best.save(out_path)
    meta_path = out_path.with_suffix(".meta.json")
    with meta_path.open("w") as f:
        json.dump({
            "mode": mode,
            "n_passengers": inst.n_passengers(),
            "miope_cost": miope_sol.total_cost,
            "alns_cost": result.best_score,
            "improvement_pct": improvement,
            "iterations": result.iterations,
            "runtime_s": result.runtime_s,
            "vehicles_miope": miope_sol.n_vehicles_used,
            "vehicles_alns": result.best.n_vehicles_used,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
