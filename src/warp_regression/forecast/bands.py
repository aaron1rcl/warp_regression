"""Forecast band helpers (terror / error / combined)."""
from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Optional, Tuple

import numpy as np


def ci_band_params(ci: float) -> Tuple[float, float, float]:
    """Map central coverage ``ci`` (e.g. 0.95) to ``(pct_lo, pct_hi, z_err)``."""
    ci = float(ci)
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    alpha = 1.0 - ci
    pct_lo = 100.0 * (alpha / 2.0)
    pct_hi = 100.0 * (1.0 - alpha / 2.0)
    z_err = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
    return pct_lo, pct_hi, z_err


def interval_coverage(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> float:
    """Empirical fraction of ``y`` falling inside ``[lo, hi]`` (equal-length arrays)."""
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError(f"shape mismatch: y={y.shape}, lo={lo.shape}, hi={hi.shape}")
    if y.size == 0:
        return float("nan")
    return float(np.mean((y >= lo) & (y <= hi)))


def build_forecast_bands(
    paths_terror: np.ndarray,
    y_point: np.ndarray,
    sigma_y: float,
    ci: float = 0.95,
    *,
    pct_lo: Optional[float] = None,
    pct_hi: Optional[float] = None,
    z_err: Optional[float] = None,
    noise_seed: int = 2,
    err_margin: Optional[np.ndarray] = None,
    err_mean: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Terror, error, and combined predictive bands at coverage ``ci``."""
    p_lo, p_hi, z = ci_band_params(ci)
    if pct_lo is not None:
        p_lo = float(pct_lo)
    if pct_hi is not None:
        p_hi = float(pct_hi)
    if z_err is not None:
        z = float(z_err)

    paths_terror = np.asarray(paths_terror, dtype=np.float64)
    y_point = np.asarray(y_point, dtype=np.float64)
    t_q_lo, t_q50, t_q_hi = np.percentile(paths_terror, [p_lo, 50, p_hi], axis=0)

    if err_margin is None:
        err_margin_arr = np.full_like(y_point, float(z * sigma_y), dtype=np.float64)
    else:
        err_margin_arr = np.asarray(err_margin, dtype=np.float64)
        if err_margin_arr.ndim == 0:
            err_margin_arr = np.full_like(y_point, float(err_margin_arr), dtype=np.float64)

    err_lo = y_point - err_margin_arr
    err_hi = y_point + err_margin_arr

    rng = np.random.default_rng(noise_seed)
    sigma_draw = err_margin_arr / z
    e = rng.normal(
        float(err_mean),
        sigma_draw[np.newaxis, :],
        size=paths_terror.shape,
    )
    paths_combined = paths_terror + e
    c_q_lo, c_q50, c_q_hi = np.percentile(paths_combined, [p_lo, 50, p_hi], axis=0)

    return {
        "ci": float(ci),
        "pct_lo": float(p_lo),
        "pct_hi": float(p_hi),
        "z_err": float(z),
        "t_q_lo": t_q_lo,
        "t_q50": t_q50,
        "t_q_hi": t_q_hi,
        "err_lo": err_lo,
        "err_hi": err_hi,
        "c_q_lo": c_q_lo,
        "c_q_hi": c_q_hi,
        "c_q50": c_q50,
        "paths_combined": paths_combined,
        "err_margin": err_margin_arr,
        "err_draws": e,
    }


def log1p_error_band_counts(
    mu_log: np.ndarray,
    sigma_y: float,
    z: float = 1.96,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map homoscedastic log1p observation noise to count-scale error bands."""
    mu_log = np.asarray(mu_log, dtype=np.float64)
    center = np.expm1(mu_log)
    rel = z * float(sigma_y) / np.maximum(np.abs(mu_log), 0.25)
    rel = np.minimum(rel, 0.95)
    return np.maximum(center * (1.0 - rel), 0.0), center * (1.0 + rel)


def count_bands_from_log1p_forecast(
    bands: Dict[str, np.ndarray],
    y_point: np.ndarray,
    sigma_y: float,
    z: float = 1.96,
) -> Dict[str, np.ndarray]:
    """Transform log1p forecast bands to counts (terror/combined via ``expm1``)."""
    y_point = np.asarray(y_point, dtype=np.float64)
    err_lo_c, err_hi_c = log1p_error_band_counts(y_point, sigma_y, z=z)
    return {
        "t_q_lo": np.expm1(bands["t_q_lo"]),
        "t_q50": np.expm1(bands["t_q50"]),
        "t_q_hi": np.expm1(bands["t_q_hi"]),
        "err_lo": err_lo_c,
        "err_hi": err_hi_c,
        "c_q_lo": np.expm1(bands["c_q_lo"]),
        "c_q_hi": np.expm1(bands["c_q_hi"]),
    }
