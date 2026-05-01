"""Data loaders for the TD-HDARP instance files.

Reads the raw operational data from the project folder
``DATOS P5 - Ruteo de profesionales de la salud`` and returns clean
``pandas.DataFrame`` / ``dict`` objects in SI-friendly units (seconds, meters, CLP).
"""

from __future__ import annotations

from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default location of the raw data folder, relative to the repo root.
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "DATOS P5 - Ruteo de profesionales de la salud"

# Mapping between the instance label and the prefixes used in the CSV filenames.
INSTANCE_FILES = {
    "AM": {
        "xlsx": "AM_large.xlsx",
        "time_csv_template": "times_AM_large_{hour:02d}.csv",
        "default_hours": (4, 5, 6),
    },
    "PM": {
        "xlsx": "PM_small.xlsx",
        "time_csv_template": "times_PM_small_{hour:02d}.csv",
        "default_hours": (22, 23, 0, 1),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_seconds(value) -> float:
    """Convert ``HH:MM:SS`` strings or ``datetime.time`` to seconds since midnight."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, dt_time):
        return value.hour * 3600 + value.minute * 60 + value.second
    if isinstance(value, str):
        h, m, s = value.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    raise TypeError(f"Cannot interpret time value: {value!r}")


def _parse_coord_pair(text: str) -> tuple[float, float]:
    """Parse a ``"lat,lon"`` string into ``(lat, lon)``."""
    lat_str, lon_str = text.split(",")
    return float(lat_str), float(lon_str)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------
def load_passengers(xlsx_path: Path) -> pd.DataFrame:
    """Load the *Passengers* sheet of an instance file.

    Returned columns (all in SI units):
      id, pickup_lat, pickup_lon, delivery_lat, delivery_lon,
      e_i (seconds since midnight), sigma_i (seconds since midnight), priority.
    """
    df = pd.read_excel(xlsx_path, sheet_name="Passengers")
    df = df.dropna(axis=1, how="all")  # drop the trailing empty column

    rename_map = {
        "Number": "id",
        "Pickup Lat": "pickup_lat",
        "Pickup Lon": "pickup_lon",
        "Delivery Lat": "delivery_lat",
        "Delivery Lon": "delivery_lon",
        "Pickup From": "e_i",
        "Turn Start": "sigma_i",
        "Turn End": "sigma_i",  # PM instance uses Turn End
        "Priority": "priority",
    }
    df = df.rename(columns=rename_map)

    df["e_i"] = df["e_i"].apply(_to_seconds)
    df["sigma_i"] = df["sigma_i"].apply(_to_seconds)
    df["priority"] = df["priority"].astype(int)
    df["id"] = df["id"].astype(int)
    return df.reset_index(drop=True)


def load_vehicles(xlsx_path: Path) -> pd.DataFrame:
    """Load the *Vehicles* sheet.

    Columns: type, stock, capacity, fixed_cost (CLP).
    """
    df = pd.read_excel(xlsx_path, sheet_name="Vehicles")
    df = df.rename(
        columns={
            "Type": "type",
            "Stock": "stock",
            "Size": "capacity",
            "Cost [CLP]": "fixed_cost",
        }
    )
    return df.reset_index(drop=True)


def load_union_constraints(xlsx_path: Path) -> dict[int, tuple[float, float]]:
    """Load the *Union constraints* sheet.

    Returns ``{priority: (W_r, M_r)}`` with both values in seconds.
    """
    df = pd.read_excel(xlsx_path, sheet_name="Union constraints")
    df = df.rename(
        columns={
            "Priority": "priority",
            "Max waiting time [min]": "W_min",
            "Max time in vehicle [min]": "M_min",
        }
    )
    result: dict[int, tuple[float, float]] = {}
    for _, row in df.iterrows():
        result[int(row["priority"])] = (float(row["W_min"]) * 60.0, float(row["M_min"]) * 60.0)
    return result


def load_others(xlsx_path: Path) -> dict[str, float | str]:
    """Load the *Others* sheet.

    Returns a dict with:
      date, cost_per_meter (CLP/m), pickup_duration (s), delivery_duration (s).
    """
    df = pd.read_excel(xlsx_path, sheet_name="Others", header=None)
    raw = dict(zip(df[0].astype(str), df[1], strict=False))

    return {
        "date": str(raw.get("Date", "")),
        "cost_per_meter": float(raw["Cost per distance [CLP/km]"]) / 1000.0,
        "pickup_duration": float(raw["Pickup duration [min]"]) * 60.0,
        "delivery_duration": float(raw["Delivery duration [min]"]) * 60.0,
    }


def load_time_matrix(csv_path: Path) -> pd.DataFrame:
    """Load a single ``times_*_HH.csv`` file.

    Returns a DataFrame with columns:
      orig_lat, orig_lon, dest_lat, dest_lon, time (s), distance (m).
    """
    df = pd.read_csv(csv_path, sep=";")
    orig = df["origin"].apply(_parse_coord_pair)
    dest = df["destination"].apply(_parse_coord_pair)
    out = pd.DataFrame(
        {
            "orig_lat": [c[0] for c in orig],
            "orig_lon": [c[1] for c in orig],
            "dest_lat": [c[0] for c in dest],
            "dest_lon": [c[1] for c in dest],
            "time": df["time"].astype(float),
            "distance": df["distance"].astype(float),
        }
    )
    return out


def load_time_matrices(
    folder: Path,
    instance: str = "AM",
    hours: tuple[int, ...] | None = None,
) -> dict[int, pd.DataFrame]:
    """Load all hourly travel-time matrices for an instance.

    Returns ``{hour: DataFrame}`` with the columns described in
    :func:`load_time_matrix`.
    """
    if instance not in INSTANCE_FILES:
        raise ValueError(f"Unknown instance: {instance!r}")
    cfg = INSTANCE_FILES[instance]
    if hours is None:
        hours = cfg["default_hours"]

    out: dict[int, pd.DataFrame] = {}
    for h in hours:
        path = folder / cfg["time_csv_template"].format(hour=h)
        if not path.exists():
            raise FileNotFoundError(path)
        out[h] = load_time_matrix(path)
    return out


def load_real_times_sample(csv_path: Path) -> pd.DataFrame:
    """Load the ``real_times_sample.csv`` file used to train the predictive model."""
    df = pd.read_csv(csv_path)
    df["departure_at"] = pd.to_datetime(df["departure_at"])
    return df
