"""Shared matplotlib helpers for warp-regression notebooks.

Every notebook (synthetic, Lynx, Bitcoin) repeats the same handful of plot
shapes: a train/test fit panel, a warp-offset trace, forecast bands, sampled
forecast "spaghetti", and a dual-objective scatter. These live here once so
the notebooks can call a function instead of re-typing 15 lines of
``fill_between``/``plot`` boilerplate per cell.

All functions take an already-created ``Axes`` (or tuple of axes) and a
generic ``x`` array — it can be integer indices, years, or dates, matplotlib
handles all three the same way.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    """Pick a sensible year/month locator based on the span of ``dates``."""
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


def plot_warp_offset(
    ax: plt.Axes,
    x: Sequence[Any],
    p: np.ndarray,
    *,
    anchor_x: Optional[Any] = None,
    label: str = "warp offset (p−i)",
    color: str = "C3",
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot the warp offset ``p(i) − i`` against ``x`` (index, year, or date)."""
    p = np.asarray(p, dtype=np.float64)
    off = p - np.arange(len(p), dtype=np.float64)
    ax.plot(x, off, color=color, lw=1.2, label=label)
    if anchor_x is not None:
        ax.axvline(anchor_x, color="0.5", ls="--", lw=0.8)
    ax.axhline(0.0, color="0.7", lw=0.6)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    return ax


def plot_fit_with_residual(
    axes: Tuple[plt.Axes, plt.Axes],
    x_train: Sequence[Any],
    y_train: np.ndarray,
    y_fit: np.ndarray,
    *,
    x_test: Optional[Sequence[Any]] = None,
    y_test: Optional[np.ndarray] = None,
    extra_lines: Optional[List[Tuple[Sequence[Any], np.ndarray, Dict[str, Any]]]] = None,
    extra_scatter: Optional[List[Tuple[Sequence[Any], np.ndarray, Dict[str, Any]]]] = None,
    train_label: str = "train",
    test_label: str = "test (holdout)",
    fit_label: str = "fit",
    ylabel: str = "y",
    residual_ylabel: str = "residual",
    xlabel: Optional[str] = None,
    forecast_start_x: Optional[Any] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Axes, plt.Axes]:
    """Two-panel pattern: observed vs fitted curve, then residual underneath.

    ``extra_lines``/``extra_scatter`` let a caller layer on driver components,
    peak markers, etc. without re-implementing the base panel.
    """
    ax0, ax1 = axes
    ax0.plot(x_train, y_train, "g.", ms=4, alpha=0.8, label=train_label)
    if x_test is not None and y_test is not None:
        ax0.plot(x_test, y_test, "ko", ms=4, alpha=0.5, label=test_label)
    ax0.plot(x_train, y_fit, "C1", lw=1.5, label=fit_label)
    for x, y, kwargs in extra_lines or []:
        ax0.plot(x, y, **kwargs)
    for x, y, kwargs in extra_scatter or []:
        ax0.scatter(x, y, **kwargs)
    ax0.set_ylabel(ylabel)
    ax0.legend(fontsize=8)
    if title:
        ax0.set_title(title)

    residual = np.asarray(y_train, dtype=np.float64) - np.asarray(y_fit, dtype=np.float64)
    ax1.plot(x_train, residual, "C3")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_ylabel(residual_ylabel)
    if xlabel:
        ax1.set_xlabel(xlabel)

    if forecast_start_x is not None:
        for ax in axes:
            ax.axvline(forecast_start_x, color="gray", ls="--", lw=0.8)
    return ax0, ax1


def plot_before_after_warp(
    ax: plt.Axes,
    x_train: Sequence[Any],
    y_train: np.ndarray,
    y_before: np.ndarray,
    y_after: np.ndarray,
    *,
    x_test: Optional[Sequence[Any]] = None,
    y_test: Optional[np.ndarray] = None,
    r2_before: Optional[float] = None,
    r2_after: Optional[float] = None,
    before_label: str = "before warp",
    after_label: str = "after warp",
    train_label: str = "train",
    test_label: str = "test (holdout)",
    extra_lines: Optional[List[Tuple[Sequence[Any], np.ndarray, Dict[str, Any]]]] = None,
    forecast_start_x: Optional[Any] = None,
    ylabel: str = "y",
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Show what the warp buys you: fit before vs after, shaded by the gap."""
    b_label = f"{before_label} (R²={r2_before:.2f})" if r2_before is not None else before_label
    a_label = f"{after_label} (R²={r2_after:.2f})" if r2_after is not None else after_label

    ax.plot(x_train, y_train, "g.", ms=5, alpha=0.7, label=train_label)
    if x_test is not None and y_test is not None:
        ax.plot(x_test, y_test, "ko", ms=5, alpha=0.45, label=test_label)
    ax.plot(x_train, y_before, color="gray", ls="--", lw=2, label=b_label)
    ax.plot(x_train, y_after, color="C1", ls="-", lw=2, label=a_label)
    ax.fill_between(x_train, y_before, y_after, color="C1", alpha=0.2, label="warp adjustment")
    for x, y, kwargs in extra_lines or []:
        ax.plot(x, y, **kwargs)
    if forecast_start_x is not None:
        ax.axvline(forecast_start_x, color="gray", ls="--", lw=0.8)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    return ax


def plot_forecast_bands(
    ax: plt.Axes,
    x_train: Sequence[Any],
    y_train: np.ndarray,
    x_bands: Sequence[Any],
    bands: Dict[str, np.ndarray],
    *,
    x_test: Optional[Sequence[Any]] = None,
    y_test: Optional[np.ndarray] = None,
    point_lines: Optional[List[Tuple[Sequence[Any], np.ndarray, Dict[str, Any]]]] = None,
    forecast_start_x: Optional[Any] = None,
    show_terror: bool = True,
    show_error: bool = False,
    show_combined: bool = True,
    train_label: str = "train",
    test_label: str = "test (holdout)",
    terror_color: str = "steelblue",
    error_color: str = "darkorange",
    combined_color: str = "mediumpurple",
    ylabel: str = "y",
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    legend_loc: str = "upper left",
) -> plt.Axes:
    """The recurring holdout-forecast panel: observed + terror/error/combined bands.

    ``bands`` is a dict as returned by ``build_forecast_bands`` (already
    sliced to align with ``x_bands``, e.g. ``bands["t_q_lo"][test_idx]``).
    ``point_lines`` draws point-forecast curve(s), e.g.
    ``[(x_train, y_point_train, {"color": "black", "lw": 2, "label": "point fit"}),
       (x_test, y_point_test, {"color": "black", "lw": 2, "ls": "--", "label": "point forecast"})]``.
    """
    ax.plot(x_train, y_train, color="green", lw=1.2, alpha=0.75, label=train_label)
    if x_test is not None and y_test is not None:
        ax.plot(x_test, y_test, "ko", ms=4, alpha=0.6, label=test_label)

    ci_pct = int(round(100.0 * float(bands.get("ci", 0.95))))
    if show_combined:
        ax.fill_between(
            x_bands, bands["c_q_lo"], bands["c_q_hi"],
            color=combined_color, alpha=0.28, label=f"total uncertainty {ci_pct}% CI",
            zorder=0,
        )
    if show_terror:
        ax.fill_between(
            x_bands, bands["t_q_lo"], bands["t_q_hi"],
            color=terror_color, alpha=0.35, label=f"time-warp uncertainty {ci_pct}% CI",
            zorder=1,
        )
    if show_error:
        ax.fill_between(
            x_bands, bands["err_lo"], bands["err_hi"],
            color=error_color, alpha=0.2, label=f"error {ci_pct}% CI",
            zorder=2,
        )

    for x, y, kwargs in point_lines or []:
        ax.plot(x, y, **kwargs)

    if forecast_start_x is not None:
        ax.axvline(forecast_start_x, color="gray", ls="--", lw=1, label="forecast start")

    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.legend(loc=legend_loc, fontsize=8)
    return ax


def plot_realisation_spaghetti(
    ax: plt.Axes,
    x: Sequence[Any],
    paths: np.ndarray,
    x_point: Sequence[Any],
    y_point: np.ndarray,
    *,
    x_obs: Optional[Sequence[Any]] = None,
    y_obs: Optional[np.ndarray] = None,
    x_test: Optional[Sequence[Any]] = None,
    y_test: Optional[np.ndarray] = None,
    obs_label: str = "observed",
    test_label: str = "test (holdout)",
    point_label: str = "point forecast",
    path_color: str = "C0",
    path_cmap: Optional[str] = None,
    path_alpha: float = 0.12,
    path_lw: float = 0.8,
    forecast_start_x: Optional[Any] = None,
    ylabel: str = "y",
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Many faint sampled realisations plus the point forecast on top.

    If ``path_cmap`` is set (e.g. ``\"turbo\"``), each realisation gets its own
    colour from that colormap; otherwise all paths use ``path_color``.
    """
    if x_obs is not None and y_obs is not None:
        ax.plot(x_obs, y_obs, color="0.75", lw=0.9, label=obs_label)
    if x_test is not None and y_test is not None:
        ax.plot(x_test, y_test, "ko", ms=3, alpha=0.55, label=test_label)
    n_paths = int(paths.shape[0])
    if path_cmap is not None and n_paths > 0:
        cmap = plt.get_cmap(path_cmap)
        colors = [cmap(k / max(n_paths - 1, 1)) for k in range(n_paths)]
    else:
        colors = [path_color] * n_paths
    for k in range(n_paths):
        ax.plot(x, paths[k], color=colors[k], alpha=path_alpha, lw=path_lw)
    ax.plot(x_point, y_point, color="black", lw=2, label=point_label)
    if forecast_start_x is not None:
        ax.axvline(forecast_start_x, color="0.5", ls="--", lw=0.8)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    return ax


def plot_dual_objective_scatter(
    ax: plt.Axes,
    series: List[Dict[str, Any]],
    *,
    xlabel: str = "-error NLL (obj_err)",
    ylabel: str = "-time LL (obj_time)",
    title: Optional[str] = None,
) -> plt.Axes:
    """Scatter of (obj_err, obj_time) points, one per fit variant to compare.

    Each entry in ``series`` is a dict with keys ``x``, ``y``, ``label``, and
    optionally ``s`` (marker size) and ``marker``.
    """
    for s in series:
        ax.scatter([s["x"]], [s["y"]], s=s.get("s", 80), marker=s.get("marker", "o"), label=s["label"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return ax
