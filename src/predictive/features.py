"""Feature engineering for the XGBoost travel-time predictor.

Builds a feature matrix from raw trip records (real_times_sample.csv) or
from on-the-fly (origin, destination, departure_time) triples used at
inference during the ALNS run.

Features
--------
* ``haversine_km``      — great-circle distance in km (proxy for route length).
* ``delta_lat``         — signed latitude difference (captures rough direction).
* ``delta_lon``         — signed longitude difference.
* ``orig_lat/lon``      — absolute pickup location.
* ``dest_lat/lon``      — absolute delivery location.
* ``hour_sin/cos``      — cyclic hour-of-day encoding.
* ``dow_sin/cos``       — cyclic day-of-week encoding.
* ``predicted_time``    — provider's forecast (strong baseline signal).

The cyclic encodings preserve continuity at midnight/Sunday without needing
the model to learn arbitrary linear slopes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "haversine_km",
    "delta_lat",
    "delta_lon",
    "orig_lat",
    "orig_lon",
    "dest_lat",
    "dest_lon",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "predicted_time",
]

TARGET_COL = "real_time"
# We train on log(real_time / predicted_time) so predictions are a multiplicative
# correction to the provider's estimate. This makes the model robust to out-of-
# distribution trip lengths (the AM routing trips are ~18 km avg, 4x longer than
# the training data at ~4 km avg). After fitting, predict = provider * exp(output).
LOG_RATIO_TARGET = "log_ratio"


# ---------------------------------------------------------------------------
# Haversine helper
# ---------------------------------------------------------------------------
def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised great-circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Build features from a DataFrame
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature DataFrame aligned with ``FEATURE_COLS``.

    Expects input columns:
      ``orig_latitude, orig_longitude, dest_latitude, dest_longitude,
      departure_at (datetime), predicted_time``.
    The ``real_time`` column is NOT included here.
    """
    out = pd.DataFrame(index=df.index)

    # Geometry
    out["haversine_km"] = haversine_km(
        df["orig_latitude"].values,
        df["orig_longitude"].values,
        df["dest_latitude"].values,
        df["dest_longitude"].values,
    )
    out["delta_lat"] = df["dest_latitude"].values - df["orig_latitude"].values
    out["delta_lon"] = df["dest_longitude"].values - df["orig_longitude"].values
    out["orig_lat"] = df["orig_latitude"].values
    out["orig_lon"] = df["orig_longitude"].values
    out["dest_lat"] = df["dest_latitude"].values
    out["dest_lon"] = df["dest_longitude"].values

    # Time
    dt = pd.to_datetime(df["departure_at"])
    hour = dt.dt.hour + dt.dt.minute / 60.0
    dow = dt.dt.dayofweek.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Provider prediction
    out["predicted_time"] = df["predicted_time"].values

    return out[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Build features for a single inference call
# ---------------------------------------------------------------------------
def build_inference_row(
    orig_lat: float,
    orig_lon: float,
    dest_lat: float,
    dest_lon: float,
    departure_seconds: float,
    predicted_time_s: float,
    reference_date: str = "2026-03-31",
) -> pd.DataFrame:
    """Construct a single-row feature DataFrame for on-the-fly prediction.

    ``departure_seconds`` is the departure time in seconds since midnight
    (matching the convention used everywhere in the ALNS).
    ``predicted_time_s`` is the provider's travel-time estimate (seconds)
    obtained from the closest hourly matrix lookup.
    ``reference_date`` is the operating date (used only for day-of-week).
    """
    departure_s = float(departure_seconds)
    hour = (departure_s / 3600.0) % 24.0
    from datetime import datetime
    base = datetime.strptime(reference_date, "%Y-%m-%d")
    dow = float(base.weekday())

    dlat = dest_lat - orig_lat
    dlon = dest_lon - orig_lon
    row = {
        "haversine_km": float(haversine_km(
            np.array([orig_lat]), np.array([orig_lon]),
            np.array([dest_lat]), np.array([dest_lon])
        )[0]),
        "delta_lat": dlat,
        "delta_lon": dlon,
        "orig_lat": orig_lat,
        "orig_lon": orig_lon,
        "dest_lat": dest_lat,
        "dest_lon": dest_lon,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7.0)),
        "predicted_time": predicted_time_s,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Load and clean the raw sample file
# ---------------------------------------------------------------------------
def load_clean_sample(
    csv_path: Path | str,
    p_low: float = 0.05,
    p_high: float = 0.95,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load ``real_times_sample.csv``, filter outliers, and return (X, y).

    Removes the bottom ``p_low`` and top ``p_high`` quantiles of ``real_time``
    to reduce the influence of extreme congestion and GPS noise.
    """
    df = pd.read_csv(csv_path)
    df["departure_at"] = pd.to_datetime(df["departure_at"])

    lo, hi = df[TARGET_COL].quantile(p_low), df[TARGET_COL].quantile(p_high)
    mask = (df[TARGET_COL] >= lo) & (df[TARGET_COL] <= hi)
    df = df[mask].reset_index(drop=True)

    X = build_features(df)
    # Target: log(real_time / predicted_time) — multiplicative correction.
    # Robust for out-of-distribution trip lengths: model learns ≈0 for typical
    # trips and corrects systematic bias by hour/location.
    ratio = df[TARGET_COL].values / df["predicted_time"].values
    y = pd.Series(np.log(ratio.clip(0.01, 10.0)), name=LOG_RATIO_TARGET)
    return X, y.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------
def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split keeping temporal ordering intact (no shuffle)."""
    n = len(X)
    split = int(n * (1 - test_frac))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
