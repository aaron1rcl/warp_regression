"""Optional ETS-style smoothing on train residuals after warp fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from .forecast import build_forecast_bands

ResidualSmoothKind = Literal["local_level", "local_trend"]


@dataclass
class ResidualSmoothFit:
    """Fitted Holt exponential smoother on warp residuals."""

    kind: ResidualSmoothKind
    level: float
    trend: float
    alpha: float
    beta: float
    sigma: float
    fitted: np.ndarray
    n_train: int


def _holt_smooth(
    residual: np.ndarray,
    alpha: float,
    beta: float,
    *,
    with_trend: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    r = np.asarray(residual, dtype=np.float64)
    n = len(r)
    level = np.zeros(n, dtype=np.float64)
    trend = np.zeros(n, dtype=np.float64)
    level[0] = r[0]
    trend[0] = (r[1] - r[0]) if n > 1 and with_trend else 0.0
    for t in range(1, n):
        prev_level = level[t - 1]
        prev_trend = trend[t - 1]
        if with_trend:
            level[t] = alpha * r[t] + (1.0 - alpha) * (prev_level + prev_trend)
            trend[t] = beta * (level[t] - prev_level) + (1.0 - beta) * prev_trend
        else:
            level[t] = alpha * r[t] + (1.0 - alpha) * prev_level
    return level, trend


def _holt_fitted(
    residual: np.ndarray,
    alpha: float,
    beta: float,
    *,
    with_trend: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    level, trend = _holt_smooth(residual, alpha, beta, with_trend=with_trend)
    fitted = np.zeros_like(residual, dtype=np.float64)
    fitted[0] = level[0]
    for t in range(1, len(residual)):
        fitted[t] = level[t - 1] + (trend[t - 1] if with_trend else 0.0)
    return level, trend, fitted


def _grid_search_holt(
    residual: np.ndarray,
    *,
    with_trend: bool,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    best_sse = np.inf
    best: Tuple[float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    alphas = np.linspace(0.05, 0.95, 19)
    betas = np.linspace(0.05, 0.95, 19) if with_trend else (0.0,)
    for alpha in alphas:
        for beta in betas:
            level, trend, fitted = _holt_fitted(residual, alpha, beta, with_trend=with_trend)
            sse = float(np.sum((residual - fitted) ** 2))
            if sse < best_sse:
                best_sse = sse
                best = (float(alpha), float(beta), level, trend, fitted)
    assert best is not None
    return best


def fit_residual_smooth(
    residual: np.ndarray,
    kind: ResidualSmoothKind = "local_trend",
) -> ResidualSmoothFit:
    """Fit Holt local level (A,N,N) or local linear trend (A,A,N) on train residuals."""
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 1:
        raise ValueError("residual must be 1-D")
    if len(residual) < 2:
        raise ValueError("residual must have length >= 2")
    with_trend = kind == "local_trend"
    alpha, beta, level, trend, fitted = _grid_search_holt(residual, with_trend=with_trend)
    innov = residual[1:] - fitted[1:]
    sigma = float(np.std(innov)) if len(innov) > 1 else 0.0
    return ResidualSmoothFit(
        kind=kind,
        level=float(level[-1]),
        trend=float(trend[-1]) if with_trend else 0.0,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        fitted=fitted,
        n_train=len(residual),
    )


def forecast_residual(res_fit: ResidualSmoothFit, n_future: int) -> np.ndarray:
    """Point forecast of residuals h=1..n_future steps beyond train end."""
    n_future = int(n_future)
    if n_future <= 0:
        return np.zeros(0, dtype=np.float64)
    h = np.arange(1, n_future + 1, dtype=np.float64)
    if res_fit.kind == "local_trend":
        return res_fit.level + h * res_fit.trend
    return np.full(n_future, res_fit.level, dtype=np.float64)


def forecast_residual_std(res_fit: ResidualSmoothFit, horizons: np.ndarray) -> np.ndarray:
    """Approximate residual forecast standard deviation by horizon."""
    h = np.maximum(np.asarray(horizons, dtype=np.float64), 1.0)
    if res_fit.kind == "local_trend":
        return res_fit.sigma * np.sqrt(h)
    return np.full_like(h, res_fit.sigma)


def residual_adjustment(
    res_fit: ResidualSmoothFit,
    n_total: int,
    n_obs: int,
) -> np.ndarray:
    """Full-length additive residual correction (in-sample fitted + future forecast)."""
    n_obs = int(n_obs)
    n_total = int(n_total)
    add = np.zeros(n_total, dtype=np.float64)
    add[:n_obs] = res_fit.fitted[:n_obs]
    n_future = n_total - n_obs
    if n_future > 0:
        add[n_obs:] = forecast_residual(res_fit, n_future)
    return add


def apply_residual_forecast(
    fc: dict,
    res_fit: ResidualSmoothFit,
    n_obs: int,
    sigma_y: float,
    *,
    noise_seed: int = 2,
) -> dict:
    """Compose warp forecast with residual smoother and rebuild bands."""
    n_obs = int(n_obs)
    n_total = int(fc["n_total"])
    add = residual_adjustment(res_fit, n_total, n_obs)

    fc = dict(fc)
    fc["preds"] = fc["preds"] + add[np.newaxis, :]
    if "preds_ci" in fc:
        fc["preds_ci"] = fc["preds_ci"] + add[np.newaxis, :]
    fc["y_point"] = fc["y_point"] + add

    n_future = n_total - n_obs
    err_margin = np.full(n_total, 1.96 * float(sigma_y), dtype=np.float64)
    if n_future > 0:
        h = np.arange(1, n_future + 1, dtype=np.float64)
        res_std = forecast_residual_std(res_fit, h)
        err_margin[n_obs:] = 1.96 * np.sqrt(float(sigma_y) ** 2 + res_std ** 2)

    paths_ci = fc.get("preds_ci", fc["preds"])
    fc["bands"] = build_forecast_bands(
        paths_ci,
        fc["y_point"],
        float(sigma_y),
        noise_seed=noise_seed,
        err_margin=err_margin,
    )
    fc["bands"]["c_q50"] = fc["y_point"]
    fc["residual_smooth"] = res_fit
    fc["residual_add"] = add
    return fc
