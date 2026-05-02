"""Run ALNS on the full AM instance (Caso Base: fixed hourly matrices, no XGBoost)."""
from src.alns.engine import run_alns, ALNSConfig
from src.data import Instance
from src.baselines.miope import MiopeSolver
import warnings
warnings.filterwarnings("ignore")

inst = Instance.from_files("AM")
print(f"Running full AM Caso Base: n={inst.n_passengers()} passengers")

miope = MiopeSolver(inst).solve()
print(f"MIOPE: {miope.total_cost:.0f} CLP ({miope.n_vehicles_used} vehicles)")

result = run_alns(inst, ALNSConfig(
    max_iter=100_000,
    t_tot=3600,      # 1 hour as specified
    t_last=500,
    k_min=8,
    k_max=20,
    seed=42,
    verbose=True,
    log_every=200,
    initial=miope,
))

print(f"ALNS:  {result.best_score:.0f} CLP, iters={result.iterations}, "
      f"runtime={result.runtime_s:.1f}s, vehicles={result.best.n_vehicles_used}")

if result.best_score < float("inf"):
    pct = (miope.total_cost - result.best_score) / miope.total_cost * 100
    print(f"  improvement over MIOPE: {pct:+.2f}%")
else:
    print("  No feasible solution found within time limit")

print(f"  destroy_weights: {result.metadata['destroy_weights']}")
print(f"  repair_weights:  {result.metadata['repair_weights']}")
