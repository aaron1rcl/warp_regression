"""Moving-average residual layer added on top of a warp forecast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forecast import build_forecast_bands, ci_band_params


@dataclass
class ResidualSmoothFit:
    """MA-smoothed residual level for additive correction of warp forecasts.

    ``level`` is the mean of the last ``horizon`` training residuals; the
    forecast is a flat constant equal to ``level`` for all future steps.
    ``fitted`` holds the rolling-MA values over the training window (for
    in-sample display).  ``sigma`` is the std of the last ``horizon``
    residuals, used to widen combined bands in ``apply_residual_forecast``.
    """

    level: float
    sigma: float
    fitted: np.ndarray
    n_train: int
    horizon: int


def fit_residual_smooth(
    residual: np.ndarray,
    *,
    horizon: int,
) -> ResidualSmoothFit:
    """Fit a moving-average residual layer on ``residual = y_obs − ŷ_warp``.

    Parameters
    ----------
    residual:
        1-D array of training residuals.
    horizon:
        Look-back window (and forecast horizon).  The level is
        ``mean(residual[-horizon:])``; the in-sample fitted series is a
        causal rolling mean with the same window width.
    """
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 1 or len(residual) < 1:
        raise ValueError("residual must be a non-empty 1-D array")
    h = max(1, min(int(horizon), len(residual)))

    level = float(np.mean(residual[-h:]))
    sigma = float(np.std(residual[-h:])) if h > 1 else 0.0

    # Causal rolling MA with window h (shrinking window at the start)
    n = len(residual)
    fitted = np.empty(n, dtype=np.float64)
    for t in range(n):
        w = min(t + 1, h)
        fitted[t] = float(np.mean(residual[t + 1 - w : t + 1]))

    return ResidualSmoothFit(
        level=level,
        sigma=sigma,
        fitted=fitted,
        n_train=n,
        horizon=h,
    )


def forecast_residual(res_fit: ResidualSmoothFit, n_future: int) -> np.ndarray:
    """Flat residual forecast: ``level`` repeated for ``n_future`` steps."""
    n_future = int(n_future)
    if n_future <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.full(n_future, res_fit.level, dtype=np.float64)


def forecast_residual_std(res_fit: ResidualSmoothFit, horizons: np.ndarray) -> np.ndarray:
    """Residual forecast uncertainty: constant ``sigma`` across all horizons."""
    return np.full(len(np.asarray(horizons)), res_fit.sigma, dtype=np.float64)


def residual_adjustment(
    res_fit: ResidualSmoothFit,
    n_total: int,
    n_obs: int,
) -> np.ndarray:
    """Full-length additive correction (rolling-MA in-sample + flat forecast out-of-sample)."""
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
    """Compose warp forecast with the MA residual layer and rebuild bands."""
    n_obs = int(n_obs)
    n_total = int(fc["n_total"])
    add = residual_adjustment(res_fit, n_total, n_obs)

    fc = dict(fc)
    fc["preds"] = fc["preds"] + add[np.newaxis, :]
    if "preds_ci" in fc:
        fc["preds_ci"] = fc["preds_ci"] + add[np.newaxis, :]
    fc["y_point"] = fc["y_point"] + add

    ci = float(fc.get("ci", 0.95))
    _, _, z = ci_band_params(ci)
    n_future = n_total - n_obs
    err_margin = np.full(n_total, z * float(sigma_y), dtype=np.float64)
    if n_future > 0:
        h_arr = np.arange(1, n_future + 1, dtype=np.float64)
        res_std = forecast_residual_std(res_fit, h_arr)
        err_margin[n_obs:] = z * np.sqrt(float(sigma_y) ** 2 + res_std ** 2)

    paths_ci = fc.get("preds_ci", fc["preds"])
    fc["bands"] = build_forecast_bands(
        paths_ci,
        fc["y_point"],
        float(sigma_y),
        ci=ci,
        noise_seed=noise_seed,
        err_margin=err_margin,
    )
    fc["residual_smooth"] = res_fit
    fc["residual_add"] = add
    return fc
