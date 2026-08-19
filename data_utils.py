"""
data_utils.py — shared utilities for the electricity demand forecasting thesis.

Functions cover: ESIOS download, Open-Meteo download, Spanish holidays,
hourly feature engineering, chronological split, metrics, and plotting.
"""

import os
import pickle
import hashlib
import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CITIES = {
    "MAD": {"name": "Madrid",    "lat": 40.4084, "lon": -3.6786},
    "BCN": {"name": "Barcelona", "lat": 41.3842, "lon":  2.1763},
    "VLC": {"name": "Valencia",  "lat": 39.4753, "lon": -0.3756},
    "SEV": {"name": "Sevilla",   "lat": 37.3862, "lon": -5.9925},
    "ZGZ": {"name": "Zaragoza",  "lat": 41.6564, "lon": -0.8792},
    "MLG": {"name": "Málaga",    "lat": 36.7203, "lon": -4.4199},
}

WEATHER_VARS = [
    "temperature_2m",
    "cloud_cover",
    "wind_speed_10m",
    "surface_pressure",
]

ESIOS_REAL_DEMAND   = 1293   # Demanda real del sistema eléctrico
ESIOS_REE_FORECAST  = 544    # Previsión de demanda de REE

# ---------------------------------------------------------------------------
# API key & cache helpers
# ---------------------------------------------------------------------------

def read_api_key(filepath: str = "esios_api_key.txt") -> str:
    with open(filepath, "r") as f:
        return f.read().strip()


def _cache_path(prefix: str, key: str, cache_dir: str = "cache") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(cache_dir, f"{prefix}_{h}.pkl")


def _load_cache(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(data, path: str) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: could not write cache {path}: {e}")


# ---------------------------------------------------------------------------
# ESIOS download
# ---------------------------------------------------------------------------

def download_esios(
    indicator_id: int,
    start_date: str,
    end_date: str,
    api_key_file: str = "esios_api_key.txt",
    cache_dir: str = "cache",
    timeout: int = 300,
) -> pd.DataFrame:
    """
    Download an ESIOS indicator and return a UTC-indexed hourly DataFrame.

    Parameters
    ----------
    indicator_id : int
        ESIOS indicator (e.g. 1293 for real demand).
    start_date : str
        ISO date/datetime, e.g. '2020-01-01' or '2020-01-01T00:00:00'.
    end_date : str
        ISO date/datetime, e.g. '2025-07-31' or '2025-07-31T23:59:59'.
    api_key_file : str
        Path to the plain-text file that contains the ESIOS API key.
    cache_dir : str
        Directory for pickle cache files.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Columns: ['datetime' (UTC, hourly), 'value' (MW)].
    """
    cache_key = f"{indicator_id}_{start_date}_{end_date}"
    path = _cache_path("esios", cache_key, cache_dir)

    cached = _load_cache(path)
    if cached is not None:
        print(f"[ESIOS] Loaded indicator {indicator_id} from cache.")
        return cached

    print(f"[ESIOS] Downloading indicator {indicator_id} ({start_date} → {end_date})…")

    # Normalise date strings to ESIOS format (YYYY-MM-DDTHH:MM:SS)
    if "T" not in start_date:
        start_date = start_date + "T00:00:00"
    if "T" not in end_date:
        end_date = end_date + "T23:59:59"

    api_key = read_api_key(api_key_file)
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    headers = {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    resp = requests.get(
        url,
        headers=headers,
        params={"start_date": start_date, "end_date": end_date},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data["indicator"]["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[["datetime", "value"]].sort_values("datetime").reset_index(drop=True)

    # Resample from 5-min to 1-hour averages
    df = df.set_index("datetime").resample("1h").mean().reset_index()
    df["value"] = df["value"].interpolate(method="time")

    _save_cache(df, path)
    print(f"[ESIOS] Downloaded {len(df)} hourly rows → cached.")
    return df


# ---------------------------------------------------------------------------
# Open-Meteo weather download
# ---------------------------------------------------------------------------

def download_openmeteo(
    start_date: str,
    end_date: str,
    cities: Optional[Dict] = None,
    variables: Optional[List[str]] = None,
    cache_dir: str = "cache",
) -> pd.DataFrame:
    """
    Download hourly weather data from Open-Meteo for a set of Spanish cities.

    Returns a single UTC-indexed DataFrame with columns:
        datetime, MAD_temperature_2m, MAD_cloud_cover, …, MLG_surface_pressure

    Parameters
    ----------
    start_date : str  e.g. '2020-01-01'
    end_date   : str  e.g. '2025-07-31'
    cities     : dict  keys are city codes, values have 'lat'/'lon'.
                 Defaults to the 6 predefined Spanish cities.
    variables  : list of Open-Meteo variable names.
                 Defaults to WEATHER_VARS.
    cache_dir  : str
    """
    if cities is None:
        cities = CITIES
    if variables is None:
        variables = WEATHER_VARS

    # Strip time part if present
    start_d = start_date.split("T")[0]
    end_d   = end_date.split("T")[0]

    cache_key = f"weather_{start_d}_{end_d}_{'_'.join(cities.keys())}"
    path = _cache_path("openmeteo", cache_key, cache_dir)

    cached = _load_cache(path)
    if cached is not None:
        print("[Weather] Loaded from cache.")
        return cached

    base_url = "https://archive-api.open-meteo.com/v1/archive"
    merged = None

    for code, info in cities.items():
        print(f"[Weather] Downloading {info['name']} ({start_d} → {end_d})…")
        params = {
            "latitude":   info["lat"],
            "longitude":  info["lon"],
            "start_date": start_d,
            "end_date":   end_d,
            "hourly":     ",".join(variables),
            "timezone":   "UTC",
        }
        resp = requests.get(base_url, params=params, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["hourly"]

        city_df = pd.DataFrame({"datetime": pd.to_datetime(raw["time"], utc=True)})
        for var in variables:
            city_df[f"{code}_{var}"] = raw.get(var)

        merged = city_df if merged is None else merged.merge(city_df, on="datetime", how="outer")

    merged = merged.sort_values("datetime").reset_index(drop=True)
    # Linear interpolation for any NaN gaps
    num_cols = [c for c in merged.columns if c != "datetime"]
    merged[num_cols] = merged[num_cols].interpolate(method="linear")

    _save_cache(merged, path)
    print(f"[Weather] Downloaded {len(merged)} hourly rows → cached.")
    return merged


# ---------------------------------------------------------------------------
# Spanish holidays
# ---------------------------------------------------------------------------

def _easter(year: int) -> datetime.date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = ((h + l - 7 * m + 114) % 31) + 1
    return datetime.date(year, month, day)


def compute_spanish_holidays(start_year: int, end_year: int) -> Dict[str, str]:
    """
    Return a dict {date_str → holiday_name} for Spanish national + major local
    holidays from start_year through end_year (inclusive).
    """
    holidays: Dict[str, str] = {}

    for year in range(start_year, end_year + 1):
        easter = _easter(year)
        good_friday   = easter - datetime.timedelta(days=2)
        holy_thursday = easter - datetime.timedelta(days=3)

        national = {
            f"{year}-01-01": "Año Nuevo",
            f"{year}-01-06": "Reyes Magos",
            good_friday.strftime("%Y-%m-%d"):   "Viernes Santo",
            f"{year}-05-01": "Día del Trabajo",
            f"{year}-08-15": "Asunción de la Virgen",
            f"{year}-10-12": "Fiesta Nacional de España",
            f"{year}-11-01": "Todos los Santos",
            f"{year}-12-06": "Día de la Constitución",
            f"{year}-12-08": "Inmaculada Concepción",
            f"{year}-12-25": "Navidad",
        }
        local = {
            # Madrid
            f"{year}-05-15": "San Isidro (MAD)",
            f"{year}-11-09": "Almudena (MAD)",
            # Barcelona
            f"{year}-09-24": "La Mercè (BCN)",
            f"{year}-04-23": "Sant Jordi (BCN)",
            # Valencia
            f"{year}-03-19": "Fallas - Sant Josep (VLC)",
            f"{year}-10-09": "Día Comunitat Valenciana (VLC)",
            # Sevilla
            holy_thursday.strftime("%Y-%m-%d"): "Jueves Santo (SEV)",
            # Zaragoza
            f"{year}-10-12": "Virgen del Pilar (ZGZ)",  # same as national
            # Málaga
            f"{year}-08-19": "Reconquista (MLG)",
        }
        holidays.update(national)
        holidays.update(local)

    return holidays


# ---------------------------------------------------------------------------
# Feature engineering (hourly)
# ---------------------------------------------------------------------------

def build_hourly_features(
    df_demand: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_holidays: Optional[Dict[str, str]] = None,
    hdd_base: float = 15.0,
    cdd_base: float = 22.0,
) -> pd.DataFrame:
    """
    Merge demand + weather and engineer all features required by the thesis.

    Parameters
    ----------
    df_demand  : DataFrame with columns ['datetime' (UTC), 'value' (MW)].
    df_weather : DataFrame with columns ['datetime' (UTC), city_var…].
    df_holidays: dict {date_str → name} from compute_spanish_holidays().
                 If None, holidays are not flagged.
    hdd_base   : Heating degree-day base temperature (°C).
    cdd_base   : Cooling degree-day base temperature (°C).

    Returns
    -------
    pd.DataFrame with datetime index and all feature columns.
    The target column is 'demand_mw'.
    """
    # -- Merge on UTC datetime --------------------------------------------------
    df = df_demand.rename(columns={"value": "demand_mw"}).merge(
        df_weather, on="datetime", how="inner"
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    dt = pd.to_datetime(df["datetime"])
    hour = dt.dt.hour
    dow  = dt.dt.dayofweek   # 0=Monday
    doy  = dt.dt.dayofyear
    month = dt.dt.month

    # -- Cyclical time encodings -----------------------------------------------
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * dow  /  7)
    df["dow_cos"]  = np.cos(2 * np.pi * dow  /  7)
    df["doy_sin"]  = np.sin(2 * np.pi * doy  / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * doy  / 365)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # -- Calendar flags --------------------------------------------------------
    df["is_weekday"]  = (dow < 5).astype(int)
    df["is_saturday"] = (dow == 5).astype(int)
    df["is_sunday"]   = (dow == 6).astype(int)

    if df_holidays:
        date_str = dt.dt.strftime("%Y-%m-%d")
        df["is_national_holiday"] = date_str.isin(
            [d for d, n in df_holidays.items() if "Virgen" not in n
             and "MAD" not in n and "BCN" not in n
             and "VLC" not in n and "SEV" not in n
             and "ZGZ" not in n and "MLG" not in n]
        ).astype(int)
        df["is_any_holiday"] = date_str.isin(df_holidays.keys()).astype(int)
    else:
        df["is_national_holiday"] = 0
        df["is_any_holiday"] = 0

    # -- Lag features ----------------------------------------------------------
    df["demand_lag_1h"]   = df["demand_mw"].shift(1)
    df["demand_lag_24h"]  = df["demand_mw"].shift(24)
    df["demand_lag_168h"] = df["demand_mw"].shift(168)

    # -- Rolling statistics (on past values only — use shift to avoid leakage) -
    df["demand_roll24_mean"]  = df["demand_mw"].shift(1).rolling(24,  min_periods=1).mean()
    df["demand_roll24_std"]   = df["demand_mw"].shift(1).rolling(24,  min_periods=1).std()
    df["demand_roll168_mean"] = df["demand_mw"].shift(1).rolling(168, min_periods=1).mean()
    df["demand_roll168_std"]  = df["demand_mw"].shift(1).rolling(168, min_periods=1).std()

    # -- HDD / CDD (using Madrid temperature as representative) ----------------
    if "MAD_temperature_2m" in df.columns:
        t = df["MAD_temperature_2m"]
        df["hdd"] = np.maximum(hdd_base - t, 0)
        df["cdd"] = np.maximum(t - cdd_base, 0)

    # Drop initial rows that have NaN lags (up to 168 hours)
    df = df.dropna(subset=["demand_lag_168h"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float   = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / validation / test sets chronologically (no shuffle).

    Parameters
    ----------
    df         : DataFrame sorted by time.
    train_frac : Fraction for training (default 0.60).
    val_frac   : Fraction for validation (default 0.20).
                 Remainder goes to test.

    Returns
    -------
    (df_train, df_val, df_test)
    """
    n = len(df)
    i_train = int(n * train_frac)
    i_val   = int(n * (train_frac + val_frac))

    df_train = df.iloc[:i_train].reset_index(drop=True)
    df_val   = df.iloc[i_train:i_val].reset_index(drop=True)
    df_test  = df.iloc[i_val:].reset_index(drop=True)

    print(
        f"Split: train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,} rows  "
        f"({100*train_frac:.0f}/{100*val_frac:.0f}/{100*(1-train_frac-val_frac):.0f} %)"
    )
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> Dict[str, float]:
    """
    Compute MAPE, RMSE and MAE.

    Parameters
    ----------
    y_true : array-like of actual values (MW).
    y_pred : array-like of predicted values (MW).
    label  : optional prefix for printed output.

    Returns
    -------
    dict with keys 'mape', 'rmse', 'mae'.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    tag = f"[{label}] " if label else ""
    print(f"{tag}MAPE={mape:.3f}%  RMSE={rmse:.1f} MW  MAE={mae:.1f} MW")
    return {"mape": mape, "rmse": rmse, "mae": mae}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Demand forecast vs actual",
    dates: Optional[pd.DatetimeIndex] = None,
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    Line plot of actual vs predicted demand.

    Parameters
    ----------
    y_true     : actual values.
    y_pred     : predicted values.
    title      : figure title.
    dates      : optional DatetimeIndex for x-axis labels.
    figsize    : figure size.
    save_path  : if given, save figure to this path (PNG/PDF).
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = dates if dates is not None else np.arange(len(y_true))
    ax.plot(x, y_true, label="Actual",    color="#1f77b4", linewidth=1.2)
    ax.plot(x, y_pred, label="Predicted", color="#ff7f0e", linewidth=1.2, linestyle="--")

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

    ax.set_title(title)
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    plt.show()


def plot_training_history(
    history,
    title: str = "Training history",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot Keras training / validation loss curves.

    Parameters
    ----------
    history   : Keras History object (or dict with 'loss' / 'val_loss').
    title     : figure title.
    figsize   : figure size.
    save_path : if given, save figure to this path.
    """
    hist = history.history if hasattr(history, "history") else history

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(hist["loss"]) + 1)

    axes[0].plot(epochs, hist["loss"],     label="Train loss")
    axes[0].plot(epochs, hist["val_loss"], label="Val loss")
    axes[0].set_title(f"{title} — Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "mae" in hist:
        axes[1].plot(epochs, hist["mae"],     label="Train MAE")
        axes[1].plot(epochs, hist["val_mae"], label="Val MAE")
        axes[1].set_title(f"{title} — MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    plt.show()


def plot_comparison_table(
    results: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (10, 3),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Render a comparison table of metrics across models and return it as a DataFrame.

    Parameters
    ----------
    results : {model_name → {'mape': …, 'rmse': …, 'mae': …, 'train_s': …}}
    """
    rows = []
    for model, m in results.items():
        rows.append({
            "Model": model,
            "MAPE (%)": f"{m['mape']:.3f}",
            "RMSE (MW)": f"{m['rmse']:.1f}",
            "MAE (MW)":  f"{m['mae']:.1f}",
            "Train time (s)": f"{m.get('train_s', 0):.1f}" if m.get("train_s") else "—",
        })
    df = pd.DataFrame(rows).set_index("Model")
    print(df.to_string())

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    plt.show()
    return df


# ---------------------------------------------------------------------------
# LSTM sequence builder
# ---------------------------------------------------------------------------

def make_sequences(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    lookback: int = 168,
    horizon: int = 24,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window (X, y) arrays for sequence-to-sequence forecasting.

    Parameters
    ----------
    feature_array : shape (T, n_features) — already normalised.
    target_array  : shape (T,) — already normalised.
    lookback      : input window length in hours (default 168 = 1 week).
    horizon       : forecast horizon in hours (default 24).
    stride        : step between consecutive windows (1 for train, 24 for test).

    Returns
    -------
    X : (n_samples, lookback, n_features)
    y : (n_samples, horizon)
    """
    X, y = [], []
    max_start = len(feature_array) - lookback - horizon + 1
    for i in range(0, max_start, stride):
        X.append(feature_array[i : i + lookback])
        y.append(target_array[i + lookback : i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
