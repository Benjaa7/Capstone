"""Adaptive operator selection for the ALNS.

Implements the roulette-wheel mechanism with periodic weight updates as
described in Section 4.5 of ``informe/chapters/metodologia.tex``:

    w_{i, j+1} = w_{i, j} (1 - ρ) + ρ · π_i / θ_i

with scores σ₁ = 10 (new global best), σ₂ = 2 (new reference), σ₃ = 0
(otherwise) and reaction factor ρ = 0.1 (Pilati 2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdaptiveParams:
    """Hyper-parameters for the adaptive operator selection."""

    sigma1: float = 10.0  # new global best
    sigma2: float = 2.0  # new reference (improving over s*)
    sigma3: float = 0.0  # otherwise
    rho: float = 0.1  # reaction factor
    segment_length: int = 100  # iterations between weight updates
    init_weight: float = 1.0
    min_weight: float = 0.05  # avoid weights collapsing to 0


@dataclass
class _OperatorStats:
    weight: float = 1.0
    score: float = 0.0
    uses: int = 0


class RouletteWheel:
    """Track per-operator weights and select with probability ∝ weight."""

    def __init__(
        self,
        names: list[str],
        params: AdaptiveParams | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.params = params or AdaptiveParams()
        self.rng = rng or np.random.default_rng(0)
        self.stats: dict[str, _OperatorStats] = {
            name: _OperatorStats(weight=self.params.init_weight) for name in names
        }
        self._iter_in_segment = 0

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def select(self) -> str:
        names = list(self.stats.keys())
        weights = np.array([self.stats[n].weight for n in names])
        probs = weights / weights.sum()
        idx = int(self.rng.choice(len(names), p=probs))
        chosen = names[idx]
        self.stats[chosen].uses += 1
        return chosen

    # ------------------------------------------------------------------
    # Score / update
    # ------------------------------------------------------------------
    def reward(self, name: str, sigma: float) -> None:
        self.stats[name].score += sigma

    def reward_global_best(self, name: str) -> None:
        self.reward(name, self.params.sigma1)

    def reward_reference(self, name: str) -> None:
        self.reward(name, self.params.sigma2)

    def reward_other(self, name: str) -> None:
        self.reward(name, self.params.sigma3)

    def end_iteration(self) -> None:
        self._iter_in_segment += 1
        if self._iter_in_segment >= self.params.segment_length:
            self._update_weights()
            self._iter_in_segment = 0

    def _update_weights(self) -> None:
        rho = self.params.rho
        for name, st in self.stats.items():
            if st.uses == 0:
                # No usage in this segment — keep current weight, just lightly damp.
                avg_score = 0.0
            else:
                avg_score = st.score / st.uses
            st.weight = max(
                self.params.min_weight,
                st.weight * (1 - rho) + rho * avg_score,
            )
            st.score = 0.0
            st.uses = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def weights(self) -> dict[str, float]:
        return {name: st.weight for name, st in self.stats.items()}
