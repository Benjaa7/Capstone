"""Travel-time predictor using the trained XGBoost model.

The predictor wraps the saved artefact from :mod:`src.predictive.train` and
exposes a simple ``predict(orig, dest, departure_seconds)`` method that
returns a travel-time estimate in seconds, ready to replace the static
hourly-matrix lookup in :class:`src.data.instance.Instance`.

The provider's ``predicted_time`` (needed as a feature) is obtained by
querying the closest hourly matrix that the Instance already holds, so no
extra data files are needed at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import xgboost as xgb

from src.predictive.features import build_inference_row

if TYPE_CHECKING:
    from src.data.instance import Instance

DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2] / "results" / "xgb_tau.json"
)


class TravelTimePredictor:
    """XGBoost-based τ(orig, dest, t) estimator.

    Load once (heavyweight), then call ``predict`` cheaply. Uses the
    instance's hourly matrix lookup to supply ``predicted_time`` as a feature
    so the model can correct systematic provider biases.
    """

    def __init__(self, model_path: Path | str = DEFAULT_MODEL) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at {model_path}. "
                "Run `python -m src.predictive.train` first."
            )
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(model_path))

    def predict(
        self,
        orig: tuple[float, float],
        dest: tuple[float, float],
        departure_seconds: float,
        instance: "Instance",
        reference_date: str = "2026-03-31",
    ) -> float:
        """Return predicted travel time (seconds).

        Parameters
        ----------
        orig, dest:
            ``(lat, lon)`` tuples in the same coordinate system as the
            time matrices.
        departure_seconds:
            Departure time in seconds since midnight.
        instance:
            Used to look up the provider's predicted time from the nearest
            hourly matrix (needed as a feature).
        reference_date:
            Operating date string ``YYYY-MM-DD``; used for day-of-week
            encoding.
        """
        # Obtain the provider's estimate directly from the matrix (no recursion).
        # _lookup bypasses the TD predictor check in tau().
        try:
            provider_time = instance._lookup(orig, dest, departure_seconds)[0]
        except KeyError:
            try:
                provider_time = instance._lookup(orig, dest, None)[0]
            except KeyError:
                provider_time = 600.0  # 10-minute fallback

        X = build_inference_row(
            orig_lat=orig[0],
            orig_lon=orig[1],
            dest_lat=dest[0],
            dest_lon=dest[1],
            departure_seconds=departure_seconds,
            predicted_time_s=provider_time,
            reference_date=reference_date,
        )
        import numpy as np
        log_ratio = float(self._model.predict(X)[0])
        # Model predicts log(real_time / provider_time); convert back to seconds.
        pred = provider_time * float(np.exp(log_ratio))
        return max(pred, 30.0)  # clamp: no trip shorter than 30s

    @classmethod
    def load_default(cls) -> "TravelTimePredictor":
        return cls(DEFAULT_MODEL)
