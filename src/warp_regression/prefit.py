"""Periodic covariate presize before warp optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import DEFAULT_PATH_ANCHOR, PathAnchor
from .covariates.sine import (
    align_sine_to_macro_peaks,
    build_dual_sines_from_fit,
    eval_sine_driver,
    fit_dual_sine_log,
    refine_param_endpoint,
)
from .utilities.splits import cumsum_path_to_stored_path


@dataclass
class LogTrendAnalysis:
    """Log-time trend fit: C + B·log(t_idx − z)."""

    B: float
    C: float
    z: float
    trend: np.ndarray
    residual: np.ndarray
    pin: Dict[str, Any]


def log_shifted_day_index(t_idx: np.ndarray, z: float) -> np.ndarray:
    return np.log(t_idx - z)


def _t_z_candidates() -> np.ndarray:
    near_one = 1.0 - np.exp(np.linspace(-8.0, -0.01, 40))
    negative = -np.exp(np.linspace(0.0, np.log(2000.0), 40))
    return np.concatenate([near_one, negative])


def analyze_log_trend(
    y: np.ndarray,
    t_idx: np.ndarray,
    *,
    z_grid: Optional[np.ndarray] = None,
) -> LogTrendAnalysis:
    """Fit C + B·log(t_idx − z) and return trend + residual for sine prefit."""
    y = np.asarray(y, dtype=np.float64)
    t_idx = np.asarray(t_idx, dtype=np.float64)
    if z_grid is None:
        z_grid = _t_z_candidates()
    best_rss = np.inf
    best: Optional[Tuple[float, float, float, np.ndarray]] = None
    for z in z_grid:
        if np.any(t_idx - z <= 0):
            continue
        log_t = log_shifted_day_index(t_idx, float(z))
        X = np.column_stack([log_t, np.ones(len(y))])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        B, C = float(coef[0]), float(coef[1])
        trend = C + B * log_t
        rss = float(np.dot(y - trend, y - trend))
        if rss < best_rss:
            best_rss = rss
            best = (B, C, float(z), trend)
    if best is None:
        raise ValueError("no valid z in log(t − z) grid")
    B, C, z, trend = best
    z_global = z
    ep_before = float(abs(y[-1] - trend[-1]))

    def trend_at_z(zz: float) -> np.ndarray:
        if np.any(t_idx - zz <= 0):
            return trend
        log_t = log_shifted_day_index(t_idx, zz)
        X = np.column_stack([log_t, np.ones(len(y))])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return coef[1] + coef[0] * log_t

    z_span = float(z_grid.max() - z_grid.min())
    z_refined, _, _ = refine_param_endpoint(
        y, trend_at_z, z, (z - 0.25 * z_span, z + 0.25 * z_span)
    )
    if np.all(t_idx - z_refined > 0):
        log_t = log_shifted_day_index(t_idx, z_refined)
        X = np.column_stack([log_t, np.ones(len(y))])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        B, C = float(coef[0]), float(coef[1])
        trend = C + B * log_t
        z = z_refined

    pin = {
        "endpoint_before": ep_before,
        "endpoint_after": float(abs(y[-1] - trend[-1])),
        "z_before": z_global,
        "z_after": z,
    }
    return LogTrendAnalysis(B=B, C=C, z=z, trend=trend, residual=y - trend, pin=pin)


def _prefit_one_sine(
    y: np.ndarray,
    t: np.ndarray,
    *,
    omega: Optional[float] = None,
    omega_range: Tuple[float, float] = (0.5, 20.0),
    n_omega: int = 40,
    n_phase: int = 36,
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    y_c = y - y.mean()
    om_grid = np.array([omega], dtype=np.float64) if omega is not None else np.linspace(
        omega_range[0], omega_range[1], n_omega
    )
    best_corr, best_om, best_ph = -np.inf, float(om_grid[0]), 0.0
    for om in om_grid:
        for ph in np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False):
            z = eval_sine_driver(t, float(om), float(ph))
            z_c = z - z.mean()
            denom = np.linalg.norm(y_c) * (np.linalg.norm(z_c) + 1e-12)
            corr = float(np.dot(y_c, z_c) / (denom + 1e-12))
            if corr > best_corr:
                best_corr, best_om, best_ph = corr, float(om), float(ph)
    z = eval_sine_driver(t, best_om, best_ph)
    return {
        "omega": best_om,
        "phase": best_ph,
        "time_scale": 1.0,
        "t_shift": 0.0,
        "presize_corr": best_corr,
        "z": z,
        "t0": float(t[0]) if len(t) else 0.0,
        "dt": float(np.median(np.diff(t))) if len(t) > 1 else 1.0,
    }


def _prefit_one_sine_peaks(
    y: np.ndarray,
    t: np.ndarray,
    peak_idx: np.ndarray,
    *,
    omega: float,
    n_phase: int = 360,
    envelope_init: bool = False,
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    peak_idx = np.asarray(peak_idx, dtype=int)
    y_c = y - y.mean()
    best_peak_align, best_corr, best_ph = -np.inf, -np.inf, 0.0
    for ph in np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False):
        z = eval_sine_driver(t, omega, float(ph))
        peak_align = float(np.mean(z[peak_idx]))
        z_c = z - z.mean()
        denom = np.linalg.norm(y_c) * (np.linalg.norm(z_c) + 1e-12)
        corr = float(np.dot(y_c, z_c) / (denom + 1e-12))
        if peak_align > best_peak_align + 1e-9 or (
            abs(peak_align - best_peak_align) <= 1e-9 and corr > best_corr
        ):
            best_peak_align, best_corr, best_ph = peak_align, corr, float(ph)

    last_peak_pin = align_sine_to_macro_peaks(t, peak_idx, omega, best_ph)
    phase = float(last_peak_pin["phase"])
    time_scale = float(last_peak_pin["time_scale"])
    t_shift = float(last_peak_pin.get("t_shift", 0.0))
    z = eval_sine_driver(t, omega, phase, time_scale, t_shift)
    out: Dict[str, Any] = {
        "omega": float(omega),
        "phase": phase,
        "peak_align": best_peak_align,
        "presize_corr": best_corr,
        "time_scale": time_scale,
        "t_shift": t_shift,
        "t0": float(t[0]) if len(t) else 0.0,
        "dt": float(np.median(np.diff(t))) if len(t) > 1 else 1.0,
        "peak_k": last_peak_pin.get("peak_k"),
        "n_peaks": len(peak_idx),
        "peak_idx": peak_idx,
        "last_peak_pin": last_peak_pin,
        "endpoint_pin": {
            "last_macro_peak_idx": int(peak_idx[-1]),
            "last_sine_peak_idx": int(last_peak_pin.get("last_peak_idx", peak_idx[-1])),
            "last_peak_value": float(last_peak_pin.get("last_peak_value", z[peak_idx[-1]])),
        },
        "z": z,
    }
    if envelope_init:
        driver = t * z
        denom = float(np.dot(driver, driver) + 1e-12)
        f_init = float(np.dot(y, driver) / denom)
        d_vals: List[float] = []
        for pi in peak_idx:
            env = (1.0 - t[pi]) * f_init * z[pi]
            if abs(env) > 1e-6:
                d_vals.append((y[pi] - f_init * z[pi]) / env)
        d_init = float(max(np.median(d_vals), 0.0)) if d_vals else 0.0
        out["f_init"] = f_init
        out["d_init"] = d_init
    return out


@dataclass
class PrefitResult:
    n_sines: int
    covariates: Dict[str, np.ndarray] = field(default_factory=dict)
    sine_fit: Dict[str, Any] = field(default_factory=dict)
    path_ref: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def drivers(self) -> Dict[str, np.ndarray]:
        """Deprecated alias for ``covariates``."""
        return self.covariates


def prefit(
    y: np.ndarray,
    t: np.ndarray,
    *,
    n_sines: int = 1,
    covariate: Optional[np.ndarray] = None,
    driver: Optional[np.ndarray] = None,
    path_ref: Optional[np.ndarray] = None,
    years: Optional[np.ndarray] = None,
    peak_idx: Optional[np.ndarray] = None,
    omega: Optional[float] = None,
    omega_range: Tuple[float, float] = (0.5, 20.0),
    omega1_range: Tuple[float, float] = (8.0, 14.0),
    omega2_range: Tuple[float, float] = (1.5, 5.0),
    n_phase: int = 360,
    envelope_init: bool = False,
    t_full: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> PrefitResult:
    n_sines = int(n_sines)
    if covariate is None:
        covariate = driver
    if n_sines == 0:
        if covariate is None:
            raise ValueError("n_sines=0 requires covariate=")
        covariate = np.asarray(covariate, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        key = "x" if path_ref is not None else "covariate"
        return PrefitResult(
            n_sines=0,
            covariates={key: covariate},
            sine_fit={},
            path_ref=path_ref,
            meta=dict(kwargs),
        )

    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    if n_sines == 1:
        if peak_idx is not None:
            if omega is None:
                raise ValueError("peak-aligned single sine requires omega=")
            sine_fit = _prefit_one_sine_peaks(
                y, t, peak_idx, omega=float(omega), n_phase=n_phase, envelope_init=envelope_init
            )
        else:
            sine_fit = _prefit_one_sine(
                y, t, omega=omega, omega_range=omega_range, n_phase=min(n_phase, 72)
            )
        if t_full is not None:
            sine_fit["z_full"] = eval_sine_driver(
                np.asarray(t_full, dtype=np.float64),
                float(sine_fit["omega"]),
                float(sine_fit["phase"]),
                time_scale=float(sine_fit.get("time_scale", 1.0)),
                t_shift=float(sine_fit.get("t_shift", 0.0)),
            )
        return PrefitResult(
            n_sines=1,
            covariates={"t": t, "z": np.asarray(sine_fit["z"], dtype=np.float64)},
            sine_fit=sine_fit,
            meta=dict(kwargs),
        )

    if n_sines == 2:
        sine_fit = fit_dual_sine_log(
            y, t, years=years, omega1_range=omega1_range, omega2_range=omega2_range, **kwargs
        )
        z1, z2 = build_dual_sines_from_fit(t, sine_fit)
        return PrefitResult(
            n_sines=2,
            covariates={"t": t, "z1": z1, "z2": z2},
            sine_fit=sine_fit,
            meta={"years": years, **kwargs},
        )

    raise ValueError(f"n_sines must be 0, 1, or 2, got {n_sines}")


def prefit_synthetic_path(
    data: Dict[str, Any],
    n_train: int,
    *,
    sr: Optional[int] = None,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> PrefitResult:
    """Convenience: known driver + end-anchored path from synthetic data."""
    sr = int(sr if sr is not None else data["sr"])
    n_train = int(n_train)
    p_ref = cumsum_path_to_stored_path(
        data["p_true"], n_train, sr, path_anchor=path_anchor
    )
    return prefit(
        data["y"][:n_train],
        np.arange(n_train, dtype=np.float64),
        n_sines=0,
        driver=data["x"][:n_train],
        path_ref=p_ref,
        path_anchor=path_anchor,
        sr=sr,
        n_train=n_train,
    )
