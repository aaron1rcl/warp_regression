"""Bitcoin data helpers and calendar utilities."""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..constants import BITCOIN_CSV
from ..plotting import format_date_axis  # noqa: F401 — notebook re-export

DATA_PATH = BITCOIN_CSV

N_CYCLE_PEAKS = 5
CYCLE_PEAK_YEARS = (2011, 2013, 2017, 2021, 2024)
CYCLE_PEAK_HALF_WINDOW_DAYS = 540
N_PATHS = 400
N_SINGLE_PATHS = 20
FORECAST_DAYS = 365
FORECAST_5YR_DAYS = int(5 * 365.25)
TEST_DAYS = 365
EPOCHS = 24000
N_KNOTS = 8
MLP_HIDDEN = 32
MLP_LAYERS = 2
FIT_LAMBDA = 0.5
WARP_LR = 0.2


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
    """Unit-interval calendar time for the sine driver: (i+½)/n, i=0…n−1."""
    return (np.arange(n, dtype=np.float64) + 0.5) / n


def split_bitcoin_holdout(
    n: int,
    test_days: int = TEST_DAYS,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, Any]:
    """Hold out the last `test_days` observations for out-of-sample forecast."""
    n_test = int(min(max(test_days, 1), n - 30))
    n_train = n - n_test
    train_idx = np.arange(n_train, dtype=int)
    test_idx = np.arange(n_train, n, dtype=int)
    out: Dict[str, Any] = {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_train": n_train,
        "n_test": n_test,
        "test_days": n_test,
    }
    if dates is not None:
        out["train_end_date"] = dates[n_train - 1]
        out["test_start_date"] = dates[n_train]
        out["dates_train"] = dates[train_idx]
        out["dates_test"] = dates[test_idx]
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


def _sine(
    t: np.ndarray,
    omega: float,
    phase: float,
    time_scale: float = 1.0,
    t_shift: float = 0.0,
) -> np.ndarray:
    return np.sin(2.0 * np.pi * omega * time_scale * t + phase + t_shift)


def sine_from_fit(t: np.ndarray, sine_fit: Dict[str, Any]) -> np.ndarray:
    return _sine(
        t,
        sine_fit["omega"],
        sine_fit["phase"],
        time_scale=sine_fit.get("time_scale", 1.0),
        t_shift=sine_fit.get("t_shift", 0.0),
    )


_sine_from_fit = sine_from_fit


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


def _t_z_candidates() -> np.ndarray:
    """Grid for presizing z in log(t_idx − z); z < 1 keeps t_idx − z > 0."""
    near_one = 1.0 - np.exp(np.linspace(-8.0, -0.01, 40))
    negative = -np.exp(np.linspace(0.0, np.log(2000.0), 40))
    return np.concatenate([near_one, negative])


def fit_log_trend(
    y: np.ndarray, t_idx: np.ndarray, z_grid: Optional[np.ndarray] = None
) -> Tuple[float, float, float, np.ndarray, Dict[str, Any]]:
    """Fit C + B·log(t_idx − z), searching z to avoid the log(1)=0 corner."""
    from ..prefit import analyze_log_trend

    out = analyze_log_trend(y, t_idx, z_grid=z_grid)
    return out.B, out.C, out.z, out.trend, out.pin


def fit_sine_peak_presize(
    y_residual: np.ndarray,
    y_prefit: np.ndarray,
    t: np.ndarray,
    dates: pd.DatetimeIndex,
    n_phase: int = 360,
) -> Dict[str, Any]:
    """Presize sine to N_CYCLE_PEAKS full cycles; endpoint-pin φ to macro peaks."""
    from ..prefit import prefit

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
    cyclic_end = sine_fit["f_init"] * (1.0 + sine_fit["d_init"] * (1.0 - t[last_idx])) * sine_fit["z"][last_idx]
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
    z_ext = _sine_from_fit(t_ext, sine_fit)
    return t_ext, t_idx_ext, z_ext, n_total

