"""Train the XGBoost travel-time correction model.

The target is ``real_time`` (seconds). Features are built by
:mod:`src.predictive.features`. A temporal 80/20 split avoids data leakage.

Usage (from repo root)::

    python -m src.predictive.train          # default paths
    python -m src.predictive.train --out models/xgb_tau.json

The saved artefact is a JSON file loadable by
:class:`src.predictive.predict.TravelTimePredictor`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from src.predictive.features import (
    FEATURE_COLS,
    TARGET_COL,
    load_clean_sample,
    temporal_train_test_split,
)

DEFAULT_DATA = (
    Path(__file__).resolve().parents[2]
    / "DATOS P5 - Ruteo de profesionales de la salud"
    / "real_times_sample.csv"
)
DEFAULT_MODEL_OUT = Path(__file__).resolve().parents[2] / "results" / "xgb_tau.json"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    csv_path: Path = DEFAULT_DATA,
    model_out: Path = DEFAULT_MODEL_OUT,
    grid_search: bool = True,
    verbose: bool = True,
) -> dict:
    """Train and save the model; return an evaluation metrics dict."""
    # 1. Load
    X, y = load_clean_sample(csv_path)
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)
    if verbose:
        print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
        print(f"Features: {FEATURE_COLS}")

    # 2. Fit (with optional grid search for n_estimators / max_depth)
    base_params: dict = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "verbosity": 0,
        "n_jobs": -1,
    }
    if grid_search:
        param_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.10],
            "subsample": [0.8, 1.0],
        }
        estimator = xgb.XGBRegressor(**base_params)
        gs = GridSearchCV(
            estimator,
            param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=int(verbose),
            refit=True,
        )
        gs.fit(X_train, y_train)
        best_model: xgb.XGBRegressor = gs.best_estimator_
        if verbose:
            print(f"Best params: {gs.best_params_}")
    else:
        # Fast default (no grid search)
        best_model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            **base_params,
        )
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

    # 3. Evaluate on hold-out (target is log-ratio, convert back to seconds for UX)
    log_ratio_pred = best_model.predict(X_test)
    log_ratio_true = y_test.values
    # Convert to actual predicted times using provider times from test features
    provider_test = X_test["predicted_time"].values
    time_pred = provider_test * np.exp(log_ratio_pred)
    time_true = provider_test * np.exp(log_ratio_true)
    mae = mean_absolute_error(time_true, time_pred)
    rmse = float(np.sqrt(mean_squared_error(time_true, time_pred)))
    r2 = r2_score(log_ratio_true, log_ratio_pred)
    if verbose:
        print(f"\n=== Hold-out metrics (log-ratio model) ===")
        print(f"  MAE on real_time:  {mae:.1f} s ({mae/60:.2f} min)")
        print(f"  RMSE on real_time: {rmse:.1f} s ({rmse/60:.2f} min)")
        print(f"  R2 (log_ratio):    {r2:.4f}")

    # 4. Save model + metadata
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    best_model.save_model(str(model_out))
    meta = {
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "mae_s": float(mae),
        "rmse_s": float(rmse),
        "r2": float(r2),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_file": str(model_out),
    }
    # Persist any best params
    try:
        meta["best_params"] = gs.best_params_  # type: ignore[name-defined]
    except NameError:
        pass
    meta_path = model_out.with_suffix(".meta.json")
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"\nModel saved: {model_out}")
        print(f"Meta  saved: {meta_path}")
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--no-grid", action="store_true")
    args = p.parse_args()
    train(csv_path=args.data, model_out=args.out, grid_search=not args.no_grid)
