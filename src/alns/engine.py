"""ALNS main loop.

Orchestrates the destroy/repair iterations following the algorithm in
Section 4.5 of ``informe/chapters/metodologia.tex`` and Pilati et al. (2025):

* Initial solution from :func:`construct_initial_solution`.
* Each iteration: roulette-select (destroy, repair); apply; update weights
  with σ₁, σ₂, σ₃; update penalty weights via ``WeightTracker``; accept the
  candidate if the dynamic-penalty score improves; if no improvement for
  ``t_last`` iterations, force diversification.
* Stop after ``t_tot`` wall-clock seconds or ``max_iter`` iterations.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from src.alns.adaptive import AdaptiveParams, RouletteWheel
from src.alns.construct import FleetPool, construct_initial_solution
from src.alns.destroy import DESTROY_OPERATORS
from src.alns.evaluation import (
    Violations,
    WeightTracker,
    evaluate_solution,
    penalised_score,
    refresh_solution_metrics,
)
from src.alns.repair import FAST_REPAIR_OPERATORS, REPAIR_OPERATORS
from src.alns.solution import Solution
from src.data.instance import Instance


@dataclass
class ALNSConfig:
    """Hyper-parameters for the ALNS run."""

    max_iter: int = 10_000
    t_tot: float = 30 * 60.0  # wall-clock seconds
    t_last: int = 2_000  # iterations without improvement before diversification
    k_min: int = 2  # minimum requests destroyed per iteration
    k_max: int = 6  # maximum requests destroyed per iteration
    seed: int = 0
    fleet: FleetPool | None = None
    adaptive: AdaptiveParams = field(default_factory=AdaptiveParams)
    log_every: int = 200  # iterations between progress prints
    verbose: bool = True
    initial: Solution | None = None  # use as starting point (e.g., miope)


@dataclass
class ALNSResult:
    """Output of an ALNS run."""

    best: Solution
    best_score: float  # f(s*) = c(s*) when feasible
    cost_history: list[float]
    iterations: int
    runtime_s: float
    metadata: dict[str, object]


def run_alns(instance: Instance, config: ALNSConfig | None = None) -> ALNSResult:
    config = config or ALNSConfig()
    rng = np.random.default_rng(config.seed)

    # 1. Initial solution.
    if config.initial is not None:
        sol = copy.deepcopy(config.initial)
    else:
        sol = construct_initial_solution(instance, fleet=config.fleet, seed=config.seed)
    refresh_solution_metrics(sol, instance)
    cost, viol = evaluate_solution(sol, instance)
    weights = WeightTracker(rng=np.random.default_rng(config.seed + 1))
    score = penalised_score(cost, viol, weights)

    # Accept the initial solution as incumbent if feasible OR near-feasible
    # (total violation magnitude < 300s, i.e. < 5 minutes across all routes).
    # This handles small tau-discrepancies between the MIOPE's static estimates
    # and the evaluator's average-lookup on the Caso Base.
    _near_feasible = viol.is_clean() or (
        viol.q < 1e-6 and viol.r < 300.0 and viol.d < 1e-6
        and viol.t < 1e-6 and viol.u < 1e-6
    )
    best_sol = copy.deepcopy(sol) if _near_feasible else None
    best_score = float("inf") if best_sol is None else cost
    reference_sol = sol
    reference_score = score

    # 2. Pre-warm instance caches so first iteration doesn't pay build cost.
    instance._ensure_avg_lookup()
    _ = instance.solo_passenger_ids
    for _, row in instance.passengers.iterrows():
        instance.effective_pickup_window(int(row["id"]))

    # 3. Roulettes — use fast (no regret) operators for large instances.
    destroy_names = list(DESTROY_OPERATORS.keys())
    _repair_pool = FAST_REPAIR_OPERATORS if instance.n_passengers() > 100 else REPAIR_OPERATORS
    repair_names = list(_repair_pool.keys())
    destroy_rw = RouletteWheel(destroy_names, params=config.adaptive, rng=np.random.default_rng(config.seed + 2))
    repair_rw = RouletteWheel(repair_names, params=config.adaptive, rng=np.random.default_rng(config.seed + 3))

    cost_history: list[float] = [cost]
    iters_since_improve = 0
    started = time.perf_counter()
    iteration = 0

    while iteration < config.max_iter:
        elapsed = time.perf_counter() - started
        if elapsed > config.t_tot:
            break

        d_name = destroy_rw.select()
        r_name = repair_rw.select()
        d_op = DESTROY_OPERATORS[d_name]
        r_op = _repair_pool[r_name]

        k = int(rng.integers(config.k_min, config.k_max + 1))

        try:
            partial, removed = d_op(reference_sol, k, rng, instance)
            candidate = r_op(partial, removed, rng, instance, weights)
        except Exception as exc:
            import traceback
            print(f"[iter {iteration}] operator {d_name}/{r_name} EXCEPTION: {exc}", flush=True)
            traceback.print_exc()
            destroy_rw.reward_other(d_name)
            repair_rw.reward_other(r_name)
            iteration += 1
            iters_since_improve += 1
            destroy_rw.end_iteration()
            repair_rw.end_iteration()
            continue

        cand_cost, cand_viol = evaluate_solution(candidate, instance)
        cand_score = penalised_score(cand_cost, cand_viol, weights)

        # Acceptance logic.
        _cand_near_feas = cand_viol.is_clean() or (
            cand_viol.q < 1e-6 and cand_viol.r < 300.0 and cand_viol.d < 1e-6
            and cand_viol.t < 1e-6 and cand_viol.u < 1e-6
        )
        improved_global = _cand_near_feas and cand_cost < best_score - 1e-6
        improved_reference = cand_score < reference_score - 1e-6

        if improved_global:
            best_sol = copy.deepcopy(candidate)
            best_score = cand_cost
            reference_sol = candidate
            reference_score = cand_score
            destroy_rw.reward_global_best(d_name)
            repair_rw.reward_global_best(r_name)
            iters_since_improve = 0
        elif improved_reference:
            reference_sol = candidate
            reference_score = cand_score
            destroy_rw.reward_reference(d_name)
            repair_rw.reward_reference(r_name)
            iters_since_improve = 0
        else:
            destroy_rw.reward_other(d_name)
            repair_rw.reward_other(r_name)
            iters_since_improve += 1

        # Forced diversification: stuck for t_last iterations. If a feasible
        # incumbent exists, snap back to it instead of accepting the current
        # candidate (otherwise infeasible drift persists indefinitely).
        if iters_since_improve >= config.t_last:
            if best_sol is not None:
                reference_sol = copy.deepcopy(best_sol)
                refresh_solution_metrics(reference_sol, instance)
                ref_cost, ref_viol = evaluate_solution(reference_sol, instance)
                # Reset penalty weights so they don't stay saturated.
                weights.alpha = weights.beta = weights.gamma = weights.epsilon = weights.phi = 1.0
                reference_score = penalised_score(ref_cost, ref_viol, weights)
            else:
                # No feasible solution yet — only accept the candidate as new
                # reference if it actually improves the current reference.
                if cand_score < reference_score - 1e-6:
                    reference_sol = candidate
                    reference_score = cand_score
                # Also partially reset weights to avoid saturation stagnation.
                weights.alpha = min(weights.alpha, 100.0)
                weights.beta = min(weights.beta, 100.0)
                weights.gamma = min(weights.gamma, 100.0)
                weights.epsilon = min(weights.epsilon, 100.0)
                weights.phi = min(weights.phi, 100.0)
            iters_since_improve = 0

        # Penalty-weight update for next iteration.
        weights.update(cand_viol)

        cost_history.append(cand_cost if cand_viol.is_clean() else float("nan"))
        destroy_rw.end_iteration()
        repair_rw.end_iteration()
        iteration += 1

        if config.verbose and iteration % config.log_every == 0:
            best_str = f"{best_score:.0f}" if best_score < float("inf") else "—"
            print(
                f"[iter {iteration:5d}] best={best_str:>10s}  "
                f"ref_score={reference_score:.0f}  "
                f"cand_cost={cand_cost:.0f}  feas={cand_viol.is_clean()}  "
                f"d={d_name:>20s}  r={r_name:>10s}",
                flush=True,
            )

    elapsed_total = time.perf_counter() - started

    if best_sol is None:
        # No feasible solution found. Return the reference (likely infeasible).
        result = ALNSResult(
            best=reference_sol,
            best_score=float("inf"),
            cost_history=cost_history,
            iterations=iteration,
            runtime_s=elapsed_total,
            metadata={
                "feasible_found": False,
                "destroy_weights": destroy_rw.weights(),
                "repair_weights": repair_rw.weights(),
            },
        )
    else:
        result = ALNSResult(
            best=best_sol,
            best_score=best_score,
            cost_history=cost_history,
            iterations=iteration,
            runtime_s=elapsed_total,
            metadata={
                "feasible_found": True,
                "destroy_weights": destroy_rw.weights(),
                "repair_weights": repair_rw.weights(),
            },
        )
    return result
