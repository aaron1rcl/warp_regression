"""Cycle-length uncertainty from terror-scaled warp path sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .constants import DEFAULT_PATH_ANCHOR, PathAnchor
from .core.warp import soft_warp_sine_numpy
from .forecast import sample_warp_paths_future


@dataclass
class CycleLengthAnalysis:
    """Distribution of next-cycle lengths under stochastic warp continuation."""

    lengths: np.ndarray
    mean_cycle_length: float
    sigma_t: float
    n_paths: int
    n_obs: int
    unit: str = "index"
    anchor: int = 0


def nominal_cycle_length(
    n_calendar: int,
    omega: float,
    *,
    time_scale: float = 1.0,
) -> float:
    """Nominal peak-to-peak spacing in index units (no warp noise)."""
    denom = float(omega) * float(time_scale)
    if denom <= 0:
        raise ValueError("omega * time_scale must be positive")
    return float(n_calendar) / denom


def _resolve_sine_params(
    sine_fit: Dict[str, Any],
    n_calendar: int,
    *,
    component: str = "primary",
) -> Dict[str, float]:
    """Extract sine driver parameters and nominal period from a fit dict."""
    if "sine1" in sine_fit:
        key = "sine1" if component != "secondary" else "sine2"
        s = sine_fit[key]
        mean_len = float(s.get("period_years", n_calendar / float(s["omega"])))
        return {
            "omega": float(s["omega"]),
            "phase": float(s["phase"]),
            "time_scale": float(s.get("time_scale", 1.0)),
            "t_shift": float(s.get("t_shift", 0.0)),
            "mean_cycle_length": mean_len,
        }
    if "omega" not in sine_fit:
        raise ValueError("sine_fit must contain omega or sine1/sine2")
    time_scale = float(sine_fit.get("time_scale", 1.0))
    return {
        "omega": float(sine_fit["omega"]),
        "phase": float(sine_fit["phase"]),
        "time_scale": time_scale,
        "t_shift": float(sine_fit.get("t_shift", 0.0)),
        "mean_cycle_length": nominal_cycle_length(
            n_calendar,
            float(sine_fit["omega"]),
            time_scale=time_scale,
        ),
    }


def _local_maxima(y: np.ndarray, min_sep: int, *, crest_only: bool = False) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 3:
        return np.array([], dtype=int)
    peaks: list[int] = []
    for i in range(1, n - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            if crest_only and y[i] <= 0.0:
                continue
            if not peaks or i - peaks[-1] >= min_sep:
                peaks.append(i)
    return np.asarray(peaks, dtype=int)


def _last_crest_at_or_before(
    y: np.ndarray,
    anchor: int,
    *,
    mean_cycle_length: float,
    min_sep: int,
) -> Optional[int]:
    """Most recent positive crest at or before ``anchor``."""
    win_start = max(1, anchor - int(1.25 * mean_cycle_length))
    peaks = _local_maxima(y[win_start : anchor + 1], min_sep, crest_only=True)
    if peaks.size:
        return int(peaks[-1]) + win_start
    return None


def _first_crest_after(y: np.ndarray, after: int, *, min_sep: int) -> Optional[int]:
    """First positive crest strictly after ``after``, at least ``min_sep`` away."""
    i = after + min_sep
    n = len(y)
    while i < n - 1:
        if y[i] > 0.0 and y[i] >= y[i - 1] and y[i] > y[i + 1]:
            return i
        i += 1
    return None


def _warped_sine_on_path(
    p: np.ndarray,
    n_calendar: int,
    sine_params: Dict[str, float],
    *,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    """sin(2πω·time_scale·t + φ) sampled at warped index ``p``."""
    return soft_warp_sine_numpy(
        p,
        n_calendar,
        sine_params["omega"],
        sine_params["phase"],
        time_scale=sine_params["time_scale"],
        t_shift=sine_params["t_shift"],
        path_anchor=path_anchor,
    )


def _next_cycle_length_on_path(
    y: np.ndarray,
    anchor: int,
    mean_cycle_length: float,
) -> Optional[float]:
    """Peak-to-peak length: last crest at/before anchor → first crest after anchor."""
    min_sep = max(1, int(0.35 * mean_cycle_length))
    last_peak = _last_crest_at_or_before(
        y, anchor, mean_cycle_length=mean_cycle_length, min_sep=min_sep
    )
    if last_peak is None:
        return None
    next_peak = _first_crest_after(y, max(anchor, last_peak), min_sep=min_sep)
    if next_peak is None or next_peak <= anchor:
        return None
    length = float(next_peak - last_peak)
    lo = 0.5 * mean_cycle_length
    hi = 2.5 * mean_cycle_length
    if length < lo or length > hi:
        return None
    return length


def _resolve_warp_params(
    fit: Optional[Dict[str, Any]],
    model: Any,
    *,
    n_obs: Optional[int],
    n_knots: Optional[int],
) -> Tuple[np.ndarray, float, int, int]:
    if fit is not None:
        p_train = np.asarray(fit["warp"]["p"], dtype=np.float64)
        sigma_t = float(fit["warp"]["sigma_t"])
        n_obs_use = int(n_obs) if n_obs is not None else len(p_train)
        n_knots_use = int(
            n_knots
            if n_knots is not None
            else fit.get("n_knots", fit["warp"].get("n_knots", 8))
        )
        return p_train[:n_obs_use], sigma_t, n_obs_use, n_knots_use
    if model is not None:
        import torch

        with torch.no_grad():
            p_train = model.path().detach().cpu().numpy()
            sigma_t = float(torch.exp(model.log_sigma_t))
        n_obs_use = int(n_obs) if n_obs is not None else int(model.n)
        n_knots_use = int(n_knots) if n_knots is not None else int(model.n_knots)
        return p_train, sigma_t, n_obs_use, n_knots_use
    raise ValueError("pass fit=... or model=...")


def _resolve_sine_fit(
    sine_fit: Optional[Dict[str, Any]],
    fit: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    sf = sine_fit
    if sf is None and fit is not None:
        if "sine_fit" in fit:
            sf = fit["sine_fit"]
        elif "sine1" in fit:
            sf = fit
        elif "omega" in fit:
            sf = fit
    if sf is None:
        raise ValueError("pass sine_fit or a fit dict that includes sine parameters")
    return sf


def sample_next_cycle_lengths(
    p_train: np.ndarray,
    sigma_t: float,
    n_knots: int,
    sine_params: Dict[str, float],
    *,
    n_calendar: int,
    n_paths: int = 500,
    seed: int = 0,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
    horizon_factor: float = 2.5,
) -> np.ndarray:
    """Sample next-cycle lengths: warped sine on stochastic warp paths.

    For each path, evaluate ``sin(2πωt+φ)`` at the warped index, locate the
    last crest at/before the train end, then the first crest strictly after
    the train end. One peak-to-peak spacing per path.
    """
    mean_cycle_length = float(sine_params["mean_cycle_length"])
    n_obs = len(p_train)
    anchor = n_obs - 1
    horizon = int(np.ceil(horizon_factor * mean_cycle_length))
    n_total = n_obs + max(horizon, 1)

    paths = sample_warp_paths_future(
        p_train,
        n_total,
        n_obs,
        sigma_t,
        n_knots,
        n_paths,
        seed,
        path_anchor=path_anchor,
    )

    lengths: list[float] = []
    for k in range(n_paths):
        y = _warped_sine_on_path(
            paths[k],
            n_calendar,
            sine_params,
            path_anchor=path_anchor,
        )
        length = _next_cycle_length_on_path(y, anchor, mean_cycle_length)
        if length is not None:
            lengths.append(length)
    return np.asarray(lengths, dtype=np.float64)


def analyze_cycle_lengths(
    *,
    fit: Optional[Dict[str, Any]] = None,
    model: Any = None,
    sine_fit: Optional[Dict[str, Any]] = None,
    n_obs: Optional[int] = None,
    n_calendar: Optional[int] = None,
    n_knots: Optional[int] = None,
    n_paths: int = 500,
    seed: int = 0,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
    unit: str = "index",
    component: str = "primary",
) -> CycleLengthAnalysis:
    """Infer next-cycle length distribution from fitted σ_t and a sine driver.

    Pass a warp **fit** dict (Bitcoin/Lynx) or a fitted **WarpParametricModel**
    (synthetic), plus ``sine_fit`` (or sine fields on ``fit``). Cycle lengths
    are measured on ``sin(2πωt+φ)`` evaluated at stochastically continued warp
    paths — no presized driver grids.
    """
    p_train, sigma_t, n_obs_use, n_knots_use = _resolve_warp_params(
        fit, model, n_obs=n_obs, n_knots=n_knots
    )
    sf = _resolve_sine_fit(sine_fit, fit)
    n_cal = int(
        n_calendar
        if n_calendar is not None
        else fit.get("n_calendar", n_obs_use) if fit else n_obs_use
    )
    sine_params = _resolve_sine_params(sf, n_cal, component=component)

    lengths = sample_next_cycle_lengths(
        p_train,
        sigma_t,
        n_knots_use,
        sine_params,
        n_calendar=n_cal,
        n_paths=n_paths,
        seed=seed,
        path_anchor=path_anchor,
    )
    return CycleLengthAnalysis(
        lengths=lengths,
        mean_cycle_length=float(sine_params["mean_cycle_length"]),
        sigma_t=sigma_t,
        n_paths=n_paths,
        n_obs=n_obs_use,
        unit=unit,
        anchor=n_obs_use - 1,
    )


def plot_cycle_length_distribution(
    analysis: CycleLengthAnalysis,
    ax: Optional[plt.Axes] = None,
    *,
    bins: int = 30,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    color: str = "steelblue",
) -> plt.Axes:
    """Histogram of sampled next-cycle lengths with nominal mean marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    lengths = analysis.lengths
    if lengths.size == 0:
        ax.text(0.5, 0.5, "No peaks found in sampled paths", ha="center", va="center", transform=ax.transAxes)
        return ax
    ax.hist(lengths, bins=bins, color=color, alpha=0.75, edgecolor="white", density=True, label="sampled")
    ax.axvline(
        analysis.mean_cycle_length,
        color="C3",
        lw=2,
        ls="--",
        label=f"nominal mean = {analysis.mean_cycle_length:.1f} {analysis.unit}",
    )
    sample_mean = float(np.mean(lengths))
    ax.axvline(
        sample_mean,
        color="k",
        lw=1.2,
        ls=":",
        label=f"sample mean = {sample_mean:.1f} {analysis.unit}",
    )
    xlab = xlabel or f"Next cycle length ({analysis.unit})"
    ax.set_xlabel(xlab)
    ax.set_ylabel("density")
    ax.set_title(
        title
        or f"Cycle length distribution  (σ_t={analysis.sigma_t:.3f}, n={analysis.n_paths} paths)"
    )
    ax.legend(fontsize=9)
    return ax
