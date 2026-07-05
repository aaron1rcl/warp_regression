"""Matplotlib helpers for warp forecast notebooks."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _dates_array(dates: Sequence[Any]) -> np.ndarray:
    if isinstance(dates, pd.Series):
        return dates.to_numpy()
    if isinstance(dates, pd.DatetimeIndex):
        return dates.to_numpy()
    return np.asarray(dates)


def format_date_axis(ax: plt.Axes, dates: Sequence[Any]) -> None:
    dates_arr = _dates_array(dates)
    span_years = (pd.Timestamp(dates_arr[-1]) - pd.Timestamp(dates_arr[0])).days / 365.25
    if span_years > 8:
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    elif span_years > 3:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")


def plot_realisations(
    ax: plt.Axes,
    dates: Sequence[Any],
    y_obs: np.ndarray,
    preds: np.ndarray,
    y_point: np.ndarray,
    forecast_start: int,
    *,
    hist_start: int = 0,
    obs_label: str = "observed",
    test_label: str = "test (holdout)",
    point_label: str = "point forecast",
    alpha: float = 0.35,
    lw: float = 0.9,
) -> None:
    """Panel of sampled forecast paths plus point forecast."""
    dates_arr = _dates_array(dates)
    n = len(y_obs)
    ax.plot(dates_arr[hist_start:forecast_start], y_obs[hist_start:forecast_start], color="0.75", lw=lw, label=obs_label)
    if forecast_start < n:
        ax.plot(dates_arr[forecast_start:n], y_obs[forecast_start:n], "ko", ms=3, alpha=0.55, label=test_label)
    for k in range(preds.shape[0]):
        ax.plot(dates_arr, preds[k], color="C0", alpha=alpha, lw=0.7)
    ax.plot(dates_arr, y_point, color="C3", lw=1.6, label=point_label)
    ax.axvline(dates_arr[forecast_start], color="0.5", ls="--", lw=0.8)
    ax.legend(loc="upper left", fontsize=9)


def plot_bands(
    ax: plt.Axes,
    dates: Sequence[Any],
    y_obs: np.ndarray,
    y_point: np.ndarray,
    bands: Dict[str, np.ndarray],
    forecast_start: int,
    *,
    hist_start: int = 0,
    obs_label: str = "observed",
    test_label: str = "test (holdout)",
    show_terror: bool = True,
    show_error: bool = True,
    show_combined: bool = True,
) -> None:
    """95% terror, error, and combined predictive bands."""
    dates_arr = _dates_array(dates)
    n = len(y_obs)
    ax.plot(dates_arr[hist_start:forecast_start], y_obs[hist_start:forecast_start], color="0.75", lw=0.9, label=obs_label)
    if forecast_start < n:
        ax.plot(dates_arr[forecast_start:n], y_obs[forecast_start:n], "ko", ms=3, alpha=0.55, label=test_label)
    ax.plot(dates_arr, y_point, color="C3", lw=1.4, label="point forecast")
    if show_terror:
        ax.fill_between(dates_arr, bands["t_q_lo"], bands["t_q_hi"], color="C0", alpha=0.15, label="95% terror")
    if show_error:
        ax.fill_between(dates_arr, bands["err_lo"], bands["err_hi"], color="C2", alpha=0.12, label="95% error")
    if show_combined:
        ax.fill_between(dates_arr, bands["c_q_lo"], bands["c_q_hi"], color="C1", alpha=0.18, label="95% combined")
    ax.axvline(dates_arr[forecast_start], color="0.5", ls="--", lw=0.8)
    ax.legend(loc="upper left", fontsize=9)


def plot_warp_offset(
    ax: plt.Axes,
    dates: Sequence[Any],
    p: np.ndarray,
    *,
    anchor_date: Optional[Any] = None,
    label: str = "warp offset (p−i)",
) -> None:
    """Plot stored path offset p − i over calendar."""
    dates_arr = _dates_array(dates)
    off = p - np.arange(len(p), dtype=np.float64)
    ax.plot(dates_arr, off, "C2", lw=1.2, label=label)
    if anchor_date is not None:
        ax.axvline(anchor_date, color="0.5", ls="--", lw=0.8)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.legend(loc="best", fontsize=9)
