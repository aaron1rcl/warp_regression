from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..utilities.metrics import _r2_rmse

def refine_param_endpoint(
    y: np.ndarray,
    pred_fn: Callable[[float], np.ndarray],
    param: float,
    bounds: Tuple[float, float],
    n_grid: int = 120,
) -> Tuple[float, float, float]:
    """1D search: minimize |y[-1] - pred_fn(param)[-1]|."""
    lo, hi = bounds
    grid = np.linspace(lo, hi, n_grid)
    best_p = float(param)
    best_before = float(abs(y[-1] - pred_fn(param)[-1]))
    best_after = best_before
    for p in grid:
        err = abs(y[-1] - pred_fn(p)[-1])
        if err < best_after:
            best_after, best_p = float(err), float(p)
    return best_p, best_before, best_after


def eval_sine_driver(
    t: np.ndarray,
    omega: float,
    phase: float,
    time_scale: float = 1.0,
    t_shift: float = 0.0,
) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return np.sin(2.0 * np.pi * omega * time_scale * t + phase + t_shift)


def phase_for_peak_at(
    t_val: float,
    omega: float,
    peak_k: int,
    time_scale: float = 1.0,
    t_shift: float = 0.0,
) -> float:
    return np.pi / 2.0 + 2.0 * np.pi * peak_k - 2.0 * np.pi * omega * time_scale * t_val - t_shift


def theoretical_sine_peak_indices(
    t: np.ndarray,
    omega: float,
    phase: float,
    time_scale: float = 1.0,
    t_shift: float = 0.0,
) -> np.ndarray:
    """Local maxima of the sine driver on the supplied calendar grid ``t``."""
    t = np.asarray(t, dtype=np.float64)
    n = len(t)
    z = eval_sine_driver(t, omega, phase, time_scale, t_shift)
    return np.array(
        [i for i in range(1, n - 1) if z[i] >= z[i - 1] and z[i] > z[i + 1]],
        dtype=int,
    )


def macro_peak_mismatch(
    macro_peak_idx: np.ndarray,
    sine_peak_idx: np.ndarray,
) -> float:
    """Sum of nearest-neighbor index distances (macro → sine peak)."""
    macro_peak_idx = np.asarray(macro_peak_idx, dtype=int)
    sine_peak_idx = np.asarray(sine_peak_idx, dtype=int)
    return float(
        sum(min(abs(int(mi) - int(sj)) for sj in sine_peak_idx) for mi in macro_peak_idx)
    )


def paired_peak_mismatch(
    macro_peak_idx: np.ndarray,
    sine_peak_idx: np.ndarray,
) -> float:
    """Order-preserving sum |macro[i] − sine[i]| (same peak count required)."""
    macro_peak_idx = np.asarray(macro_peak_idx, dtype=int)
    sine_peak_idx = np.asarray(sine_peak_idx, dtype=int)
    if len(macro_peak_idx) != len(sine_peak_idx):
        return float("inf")
    return float(np.sum(np.abs(macro_peak_idx - sine_peak_idx)))


def _endpoint_pin_sine_params(
    t: np.ndarray,
    macro_peak_idx: np.ndarray,
    omega: float,
) -> Tuple[float, float]:
    """Pin sin=1 at first macro peak; stretch so last macro is the (n-1)-th crest."""
    macro_peak_idx = np.asarray(macro_peak_idx, dtype=int)
    first_idx = int(macro_peak_idx[0])
    last_idx = int(macro_peak_idx[-1])
    n_macro = len(macro_peak_idx)
    span = float(t[last_idx] - t[first_idx])
    if span <= 0.0:
        return 0.0, 1.0
    time_scale = float((n_macro - 1) / (omega * span))
    phase = float(np.pi / 2.0 - 2.0 * np.pi * omega * time_scale * t[first_idx])
    return phase, time_scale


def _align_sine_candidate(
    t: np.ndarray,
    macro_peak_idx: np.ndarray,
    omega: float,
    phase: float,
    time_scale: float,
    phase_before: float,
    macro_align_before: float,
    peak_k: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    n = len(t)
    n_macro = len(macro_peak_idx)
    last_idx = int(macro_peak_idx[-1])
    sine_peak_idx = theoretical_sine_peak_indices(t, omega, float(phase), time_scale)
    if len(sine_peak_idx) != n_macro:
        return None
    z = eval_sine_driver(t, omega, float(phase), time_scale)
    paired_mm = paired_peak_mismatch(macro_peak_idx, sine_peak_idx)
    macro_align = float(np.mean(z[macro_peak_idx]))
    last_sine_idx = int(sine_peak_idx[-1])
    return {
        "phase": float(phase),
        "time_scale": float(time_scale),
        "t_shift": 0.0,
        "peak_k": peak_k,
        "last_peak_idx": last_sine_idx,
        "last_peak_value": float(z[last_idx]),
        "macro_peak_align": macro_align,
        "macro_peak_mismatch": macro_peak_mismatch(macro_peak_idx, sine_peak_idx),
        "paired_peak_mismatch": paired_mm,
        "phase_before": phase_before,
        "macro_peak_align_before": macro_align_before,
        "sine_peak_idx": sine_peak_idx.tolist(),
        "last_peak_offset": float(last_idx - last_sine_idx),
    }


def align_sine_to_macro_peaks(
    t: np.ndarray,
    macro_peak_idx: np.ndarray,
    omega: float,
    phase_coarse: float,
    time_scale_grid: Optional[np.ndarray] = None,
    n_phase: int = 240,
) -> Dict[str, Any]:
    """Align presized sine so macro peaks sit on sine crests.

    Primary fit: **endpoint pin** — sin=1 at the first and last macro peak
    (five lobes on train). Local refine keeps those endpoints as sine crests.
    """
    macro_peak_idx = np.asarray(macro_peak_idx, dtype=int)
    n_macro = len(macro_peak_idx)
    first_idx = int(macro_peak_idx[0])
    last_idx = int(macro_peak_idx[-1])
    n = len(t)
    phase_before = float(phase_coarse)
    coarse_z = eval_sine_driver(t, omega, phase_before)
    macro_align_before = float(np.mean(coarse_z[macro_peak_idx]))
    best_key: Optional[Tuple[float, float, float]] = None
    best: Optional[Dict[str, Any]] = None

    def consider(phase: float, time_scale: float, peak_k: Optional[int] = None) -> None:
        nonlocal best_key, best
        cand = _align_sine_candidate(
            t,
            macro_peak_idx,
            omega,
            phase,
            time_scale,
            phase_before,
            macro_align_before,
            peak_k=peak_k,
        )
        if cand is None:
            return
        sine_peak_idx = np.asarray(cand["sine_peak_idx"], dtype=int)
        if int(sine_peak_idx[0]) != first_idx or int(sine_peak_idx[-1]) != last_idx:
            return
        z = eval_sine_driver(t, omega, float(phase), time_scale)
        macro_sin = z[macro_peak_idx]
        middle_mm = paired_peak_mismatch(macro_peak_idx[1:-1], sine_peak_idx[1:-1])
        key = (
            middle_mm,
            -float(np.mean(macro_sin[1:-1])),
            -float(np.min(macro_sin)),
        )
        if best_key is None or key < best_key:
            best_key = key
            best = cand

    phase_end, ts_end = _endpoint_pin_sine_params(t, macro_peak_idx, omega)
    consider(phase_end, ts_end, peak_k=0)

    phase_span = max(0.2, 2.0 * np.pi / max(n_phase, 1))
    ts_span = 0.06
    phase_grid = np.linspace(phase_end - phase_span, phase_end + phase_span, n_phase)
    ts_grid = np.linspace(max(0.75, ts_end - ts_span), ts_end + ts_span, 48)
    if time_scale_grid is not None:
        ts_grid = np.unique(np.concatenate([ts_grid, np.asarray(time_scale_grid, dtype=float)]))

    for time_scale in ts_grid:
        for phase in phase_grid:
            consider(float(phase), float(time_scale))

    if best is None:
        z = eval_sine_driver(t, omega, phase_before)
        return {
            "phase": phase_before,
            "time_scale": 1.0,
            "t_shift": 0.0,
            "peak_k": None,
            "last_peak_idx": last_idx,
            "last_peak_value": float(z[last_idx]),
            "macro_peak_align": macro_align_before,
            "macro_peak_mismatch": float("inf"),
            "paired_peak_mismatch": float("inf"),
            "phase_before": phase_before,
            "macro_peak_align_before": macro_align_before,
            "sine_peak_idx": theoretical_sine_peak_indices(
                t, omega, phase_before, 1.0
            ).tolist(),
            "last_peak_offset": 0.0,
        }
    return best


# Backward-compatible alias
align_sine_last_macro_peak = align_sine_to_macro_peaks
align_sine_to_peaks = align_sine_to_macro_peaks


def _sine(t: np.ndarray, omega: float, phase: float) -> np.ndarray:
    return np.sin(2.0 * np.pi * omega * t + phase)


def find_big_peak_indices(y_log: np.ndarray, years: np.ndarray) -> np.ndarray:
    idx = []
    for i in range(1, len(y_log) - 1):
        if y_log[i] >= y_log[i - 1] and y_log[i] > y_log[i + 1]:
            idx.append(i)
    if not idx:
        return np.array([int(np.argmax(y_log))])
    peaks = np.array(idx, dtype=int)
    thresh = np.median(y_log[peaks])
    return peaks[y_log[peaks] >= thresh]


def fit_dual_sine_log(
    y_log: np.ndarray,
    t: np.ndarray,
    years: Optional[np.ndarray] = None,
    omega1_range: Tuple[float, float] = (8.0, 14.0),
    omega2_range: Tuple[float, float] = (1.5, 5.0),
) -> Dict[str, Any]:
    years = years if years is not None else np.arange(len(y_log))
    om1_grid = np.linspace(omega1_range[0], omega1_range[1], 25)
    om2_grid = np.linspace(omega2_range[0], omega2_range[1], 25)
    best_corr, om1, ph1 = -np.inf, float(om1_grid[0]), 0.0
    y_c = y_log - y_log.mean()
    for om in om1_grid:
        for ph in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            z = _sine(t, om, ph)
            z_c = z - z.mean()
            corr = float(np.dot(y_c, z_c) / (np.linalg.norm(y_c) * (np.linalg.norm(z_c) + 1e-12)))
            if corr > best_corr:
                best_corr, om1, ph1 = corr, float(om), float(ph)
    z1 = _sine(t, om1, ph1)
    resid = y_log - z1 * (np.dot(y_log, z1) / (np.dot(z1, z1) + 1e-12))
    big_peak_idx = find_big_peak_indices(y_log, years)
    best_score, om2, ph2, corr2 = -np.inf, float(om2_grid[0]), 0.0, 0.0
    for om in om2_grid:
        for ph in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            z2 = _sine(t, om, ph)
            z2_peaks = [i for i in range(1, len(z2) - 1) if z2[i] >= z2[i - 1] and z2[i] > z2[i + 1]]
            align = 0.0
            for bp in big_peak_idx:
                if z2_peaks:
                    align += float(np.exp(-0.5 * ((np.array(z2_peaks) - bp) / max(len(y_log) / 20, 1)) ** 2).max())
            align /= max(len(big_peak_idx), 1)
            z2_c = z2 - z2.mean()
            c2 = float(np.dot(resid - resid.mean(), z2_c) / ((np.linalg.norm(resid - resid.mean()) + 1e-12) * (np.linalg.norm(z2_c) + 1e-12)))
            score = 0.5 * align + 0.5 * c2
            if score > best_score:
                best_score, om2, ph2, corr2 = score, float(om), float(ph), c2
    z2 = _sine(t, om2, ph2)
    z2_peak_idx = np.array(
        [i for i in range(1, len(z2) - 1) if z2[i] >= z2[i - 1] and z2[i] > z2[i + 1]],
        dtype=int,
    )
    X = np.column_stack([z1, z2, np.ones_like(t)])
    coef, _, _, _ = np.linalg.lstsq(X, y_log, rcond=None)
    a1, a2, b = coef
    y_hat = X @ coef
    ss_res = float(np.sum((y_log - y_hat) ** 2))
    ss_tot = float(np.sum((y_log - y_log.mean()) ** 2)) + 1e-12
    t_arr = np.asarray(t, dtype=np.float64)
    dt = float(np.median(np.diff(t_arr))) if len(t_arr) > 1 else 1.0
    t0 = float(t_arr[0]) if len(t_arr) else 0.0
    # Index-unit periods from the prefit calendar step (years when dt is year-scaled).
    period1 = 1.0 / (om1 * dt)
    period2 = 1.0 / (om2 * dt)
    return {
        "sine1": {"omega": om1, "phase": ph1, "amplitude": float(a1), "period_years": period1},
        "sine2": {"omega": om2, "phase": ph2, "amplitude": float(a2), "peak_alignment": best_score, "period_years": period2},
        "intercept": float(b),
        "z1": z1,
        "z2": z2,
        "component1_log": a1 * z1,
        "component2_log": a2 * z2,
        "y_hat_log": y_hat,
        "r2_log": 1.0 - ss_res / ss_tot,
        "rmse_log": float(np.sqrt(ss_res / len(y_log))),
        "big_peak_idx": big_peak_idx,
        "big_peak_years": years[big_peak_idx],
        "z2_peak_idx": z2_peak_idx,
        "t0": t0,
        "dt": dt,
    }


def build_dual_sines_from_fit(t: np.ndarray, sine_fit: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    s1, s2 = sine_fit["sine1"], sine_fit["sine2"]
    return _sine(t, s1["omega"], s1["phase"]), _sine(t, s2["omega"], s2["phase"])


def sine_wave(
    t: np.ndarray,
    omega: float,
    phase: float,
    time_scale: float = 1.0,
    t_shift: float = 0.0,
) -> np.ndarray:
    return eval_sine_driver(t, omega, phase, time_scale, t_shift)


@dataclass
class SineSpec:
    omega: float
    phase: float
    time_scale: float = 1.0
    t_shift: float = 0.0


def build_sine_features(t: np.ndarray, specs: List[SineSpec]) -> np.ndarray:
    """Stack sine driver columns from arbitrary specs."""
    cols = [
        sine_wave(t, s.omega, s.phase, s.time_scale, s.t_shift)
        for s in specs
    ]
    return np.column_stack(cols) if cols else np.empty((len(t), 0))


def presize_dual_sine(
    y: np.ndarray,
    t: np.ndarray,
    peak_idx: Optional[np.ndarray] = None,
    years: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Lynx presize recipe: dual correlated sines on log counts."""
    return fit_dual_sine_log(y, t, years=years, **kwargs)


def presize_log_trend_sine(
    y: np.ndarray,
    t_idx: np.ndarray,
    t: np.ndarray,
    dates: pd.DatetimeIndex,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Deprecated: use analyze_log_trend + prefit(n_sines=1)."""
    from ..prefit import analyze_log_trend, prefit
    from ..utilities.bitcoin import detect_btc_cycle_peaks

    prep = analyze_log_trend(y, t_idx)
    peak_idx = detect_btc_cycle_peaks(y, dates)
    pf = prefit(
        prep.residual,
        t,
        n_sines=1,
        peak_idx=peak_idx,
        omega=float(kwargs.get("omega", 5.0)),
        envelope_init=True,
        **kwargs,
    )
    return {
        "trend": {"B": prep.B, "C": prep.C, "z": prep.z, "trend": prep.trend, "pin": prep.pin},
        "sine_fit": pf.sine_fit,
        "z": pf.sine_fit["z"],
    }

