"""Example dataset loaders and demo helpers for the notebooks."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .plotting import format_date_axis  # noqa: F401 — re-export for bitcoin notebook
from warp_regression.covariates.sine import sine_from_fit  # noqa: F401

# examples/utils -> src/data
DATA_ROOT = Path(__file__).resolve().parents[2] / "src" / "data"
LYNX_CSV = DATA_ROOT / "lynx.csv"
BITCOIN_CSV = DATA_ROOT / "bitcoin_daily.csv"

# ---------------------------------------------------------------------------
# CSV series (lynx)
# ---------------------------------------------------------------------------


def _load_series_csv(
    csv_path: Path | str,
    transform: str,
    *,
    year_dtype: Any = np.int32,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    years = df["time"].to_numpy(dtype=year_dtype)
    counts = df["value"].to_numpy(dtype=np.float64)
    n = len(years)
    t = np.linspace(0.0, 1.0, n)
    if transform == "log1p":
        y_log = np.log1p(counts)
    elif transform == "log":
        y_log = np.log(np.maximum(counts, 1e-8))
    elif transform == "identity":
        y_log = counts.copy()
    else:
        raise ValueError(transform)
    return {"years": years, "counts": counts, "t": t, "y_log": y_log, "n": n, "transform": transform}


def prepare_lynx_log(csv_path: Path | str = LYNX_CSV, transform: str = "log1p") -> Dict[str, Any]:
    return _load_series_csv(csv_path, transform)


# ---------------------------------------------------------------------------
# Bitcoin daily series
# ---------------------------------------------------------------------------

DATA_PATH = BITCOIN_CSV

# Data / calendar defaults (training knobs live in examples/models/*.yaml + notebooks)
N_CYCLE_PEAKS = 5
CYCLE_PEAK_YEARS = (2011, 2013, 2017, 2021, 2024)
CYCLE_PEAK_HALF_WINDOW_DAYS = 540
TEST_DAYS = 365


def fetch_bitcoin_daily(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"])
        if len(df) > 100:
            return df

    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=json&sampled=false"
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    rows = [
        {
            "date": datetime.fromtimestamp(pt["x"], tz=timezone.utc).date().isoformat(),
            "price_usd": float(pt["y"]),
        }
        for pt in payload["values"]
        if pt.get("y") and float(pt["y"]) > 0.0
    ]
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def log_price(price: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(price, 1e-12))


def price_from_log(y: np.ndarray) -> np.ndarray:
    return np.exp(y)


def normalized_time(n: int) -> np.ndarray:
    """Unit-interval calendar time for the sine covariate: (i+½)/n, i=0…n−1."""
    return (np.arange(n, dtype=np.float64) + 0.5) / n


def split_bitcoin_holdout(
    n: int,
    test_days: int = TEST_DAYS,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, Any]:
    """Deprecated alias — use ``split_holdout(n, n_test=...)``."""
    from warp_regression.utilities.splits import split_holdout

    out = split_holdout(n, n_test=int(test_days), dates=dates, min_train=30)
    out["test_days"] = out["n_test"]
    return out


def day_index(n: int, start: int = 1) -> np.ndarray:
    """Day counter for the log trend: start, start+1, …, start+n−1."""
    return np.arange(start, start + n, dtype=np.float64)


def log_day_index(n: int, start: int = 1) -> np.ndarray:
    """Natural log of day index; start=1 keeps log(t) finite."""
    return np.log(day_index(n, start))


def log_shifted_day_index(t_idx: np.ndarray, z: float) -> np.ndarray:
    """log(t_idx − z) with t_idx ≥ 1 and z < 1 so the argument stays positive."""
    return np.log(t_idx - z)


def _peak_center(year: int) -> pd.Timestamp:
    if year == 2011:
        return pd.Timestamp("2011-06-01")
    if year == 2013:
        return pd.Timestamp("2013-12-01")
    return pd.Timestamp(f"{year}-06-15")


def detect_btc_cycle_peaks(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    cycle_years: Tuple[int, ...] = CYCLE_PEAK_YEARS,
    half_window_days: int = CYCLE_PEAK_HALF_WINDOW_DAYS,
) -> np.ndarray:
    peak_idx: List[int] = []
    for year in cycle_years:
        center = _peak_center(year)
        lo = center - pd.Timedelta(days=half_window_days)
        hi = center + pd.Timedelta(days=half_window_days)
        mask = (dates >= lo) & (dates <= hi)
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        peak_idx.append(int(idx[np.argmax(y[idx])]))
    if len(peak_idx) < 2:
        return np.linspace(0, len(y) - 1, N_CYCLE_PEAKS, dtype=int)
    return np.array(sorted(set(peak_idx)), dtype=int)


def omega_from_peak_times(t: np.ndarray, peak_idx: np.ndarray) -> float:
    t_peaks = t[peak_idx]
    span = float(t_peaks[-1] - t_peaks[0])
    if span < 1e-6:
        return 1.0
    return float(len(peak_idx) - 1) / span


def fit_log_trend(
    y: np.ndarray, t_idx: np.ndarray, z_grid: Optional[np.ndarray] = None
) -> Tuple[float, float, float, np.ndarray, Dict[str, Any]]:
    """Fit C + B·log(u)/(1+γ·log(u)), u=t_idx−z; γ≥0 nests plain log."""
    from warp_regression.prefit import analyze_log_trend

    out = analyze_log_trend(y, t_idx, z_grid=z_grid)
    pin = dict(out.pin)
    pin["gamma"] = float(out.gamma)
    return out.B, out.C, out.z, out.trend, pin


def fit_sine_peak_presize(
    y_residual: np.ndarray,
    y_prefit: np.ndarray,
    t: np.ndarray,
    dates: pd.DatetimeIndex,
    n_phase: int = 360,
) -> Dict[str, Any]:
    """Presize sine to N_CYCLE_PEAKS full cycles; endpoint-pin φ to macro peaks."""
    from warp_regression.prefit import prefit

    peak_idx = detect_btc_cycle_peaks(y_prefit, dates)
    pf = prefit(
        y_residual,
        t,
        n_sines=1,
        peak_idx=peak_idx,
        omega=float(N_CYCLE_PEAKS),
        n_phase=n_phase,
        envelope_init=True,
    )
    sine_fit = dict(pf.sine_fit)
    sine_fit["n_cycles"] = N_CYCLE_PEAKS
    sine_fit["omega_from_peaks"] = omega_from_peak_times(t, peak_idx)
    sine_fit["period_series_units"] = 1.0 / float(N_CYCLE_PEAKS)
    sine_fit["peak_dates"] = [str(dates[i].date()) for i in peak_idx]
    last_idx = len(t) - 1
    a_floor = float(sine_fit.get("a_floor_init", sine_fit.get("a_inf_init", 1.0)))
    kappa = float(sine_fit.get("kappa_init", sine_fit.get("lam_init", 0.0)))
    atten_end = a_floor + (1.0 - a_floor) * float(np.exp(-kappa * t[last_idx]))
    cyclic_end = sine_fit["f_init"] * atten_end * sine_fit["z"][last_idx]
    sine_fit["endpoint_pin"]["endpoint_residual"] = float(abs(y_residual[last_idx] - cyclic_end))
    return sine_fit


def build_forecast_extension(
    n_current: int,
    n_future_extra: int,
    sine_fit: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Calendar arrays for ``n_current + n_future_extra`` days (t_norm base = n_current)."""
    n_total = n_current + n_future_extra
    t_ext = (np.arange(n_total, dtype=np.float64) + 0.5) / n_current
    t_idx_ext = day_index(n_total, start=1)
    z_ext = sine_from_fit(t_ext, sine_fit)
    return t_ext, t_idx_ext, z_ext, n_total


# ---------------------------------------------------------------------------
# Synthetic warp demo (1_Introduction_to_Warp_Regression)
# ---------------------------------------------------------------------------


def matrix_decomp(x: np.ndarray, n: int, sr: int) -> np.ndarray:
    if x.ndim == 2:
        x = x[:, 0]
    x_i = np.full((n * sr,), np.nan, dtype=np.float32)
    pos = np.arange(0, n) * sr
    x_i[pos] = x
    ones = np.ones((1, x_i.shape[0]), dtype=np.float32)
    x_i = x_i.reshape(-1, 1)
    X = np.dot(x_i, ones)
    I = np.eye(n * sr)
    I[I == 0] = np.nan
    return X * I


def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    if k > 0:
        y = np.concatenate([np.full((k,), np.nan), x])
        y = y[: len(x)]
    elif k < 0:
        k = abs(k)
        y = np.concatenate([x, np.full((k,), np.nan)])
        y = y[k:]
    else:
        y = x
    return y


def X_shift(X: np.ndarray, sr: int, p: np.ndarray) -> np.ndarray:
    if np.max(p) > 0:
        X_out = np.pad(X.copy(), ((0, 0), (0, X.shape[0])), constant_values=np.nan)
    else:
        X_out = X.copy()
    for i in range(1, int(X.shape[0] / sr)):
        x_r = X_out[i * sr, :]
        x_shift = shift(x_r, int(p[i]))
        X_out[i * sr, :] = x_shift
    return X_out


def fill_nan_with_last_value_inplace(x: np.ndarray) -> np.ndarray:
    mask = np.isnan(x)
    last = x[0]
    for i in range(1, len(x)):
        if mask[i]:
            x[i] = last
        else:
            last = x[i]
    return x


def generate_example_waves(n: int, seed: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    phase = np.arange(1, n + 1) / 100.0
    phase = 2.0 * np.pi * phase
    y1 = np.sin(phase)
    y2 = y1 + np.random.normal(0.0, 0.5, size=n)
    return phase.astype(np.float64), y1.astype(np.float64), y2.astype(np.float64)


def generate_sample_perturbations(
    scale: float,
    n: int,
    truncate: bool = False,
    sr: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    perturbations = np.random.normal(loc=0.0, scale=(scale - 1), size=n).astype(int)
    if truncate:
        perturbations[perturbations <= -sr] = -sr
    return perturbations, np.cumsum(perturbations, dtype=np.float64)


def build_synthetic_dataset(
    n: int = 300,
    sr: int = 10,
    scale: float = 10.0,
    noise_std: float = 0.2,
    truncate_perturbations: bool = False,
    seed: int = 12,
) -> Dict[str, Any]:
    """Run WarpPureNumpy.ipynb cells 4–7: warp sin(phase), not raw phase."""
    _, x, y_clean = generate_example_waves(n, seed=seed)
    p_step, p_true = generate_sample_perturbations(scale, n, truncate=truncate_perturbations, sr=sr)
    X = matrix_decomp(x, n, sr)
    X_s = X_shift(X, sr, p_true)
    x_s = np.nanmedian(X_s, axis=0)
    x_out = fill_nan_with_last_value_inplace(x_s.copy())
    x_warped = x_out[::sr][:n].astype(np.float64)
    y = x_warped + np.random.normal(0.0, noise_std, size=n)
    return {
        "x": x.astype(np.float64),
        "phase": _,
        "y": y.astype(np.float64),
        "y_clean": y_clean,
        "x_warped": x_warped,
        "p_true": p_true.astype(np.float64),
        "p_step": p_step,
        "n": n,
        "sr": sr,
        "scale": scale,
        "noise_std": noise_std,
    }
