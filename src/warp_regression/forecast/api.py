"""High-level forecast entry points."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor

from ..core.path import DEFAULT_PATH_ANCHOR, PathAnchor, knot_positions
from .residual import apply_residual_forecast, fit_residual_smooth
from .state import as_forecast_state, forecast_from_state


def predict_realisations_torch(
    model: Any,
    x: Optional[Union[Tensor, np.ndarray]] = None,
    n_draws: int = 400,
    horizon: int = 60,
    seed: int = 0,
    *,
    covariates: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Sample train-window path uncertainty for a fitted ``WarpModel``."""
    if not getattr(model, "is_fitted", False):
        raise RuntimeError("model not fitted")
    if model.path_mode != "identity":
        raise ValueError("predict_realisations_torch currently supports path_mode='identity'")

    n = int(model.n)
    h = min(int(horizon), n)
    anchor = model.path_anchor
    path = model.primary_path
    with torch.no_grad():
        p_fit = path.path(n=n).cpu().numpy()
        sigma_t = float(path.sigma_t.detach())

    rng = np.random.default_rng(seed)
    _, rw_interval = knot_positions(n, path.n_knots)
    sigma_knot = float(sigma_t * rw_interval)
    knot_x = np.linspace(0.0, float(n - 1), path.n_knots)
    idx = np.arange(n, dtype=np.float64)
    off_knots_fit = np.interp(knot_x, idx, p_fit - idx)

    cov = dict(covariates or {})
    if x is not None:
        cov.setdefault("x", x)
    for k, v in getattr(model, "_covariates", {}).items():
        cov.setdefault(k, v)
    cov_t = {
        k: (v if isinstance(v, Tensor) else torch.as_tensor(v, dtype=torch.float32))
        for k, v in cov.items()
    }
    cal_t = {
        k: torch.as_tensor(v, dtype=torch.float32)
        for k, v in getattr(model, "_calendar", {}).items()
    }

    paths = np.zeros((n_draws, h))
    for k in range(n_draws):
        dev = np.zeros(path.n_knots, dtype=np.float64)
        if anchor == "end":
            for j in range(1, path.n_knots):
                dev[path.n_knots - 1 - j] = dev[path.n_knots - j] + rng.normal(0.0, sigma_knot)
        else:
            for j in range(1, path.n_knots):
                dev[j] = dev[j - 1] + rng.normal(0.0, sigma_knot)
        off_k = off_knots_fit + dev
        off_k[-1 if anchor == "end" else 0] = 0.0
        p = idx + np.interp(idx, knot_x, off_k)
        p[-1 if anchor == "end" else 0] = float(n - 1) if anchor == "end" else 0.0
        with torch.no_grad():
            y_full, _ = model.predict_torch(
                cov_t,
                cal_t,
                p=torch.tensor(p, dtype=torch.float32),
                sine_fit=getattr(model, "_sine_fit", None),
            )
        paths[k] = y_full.detach().cpu().numpy()[:h]
    return paths


def _predict_forecast_model(
    model: Any,
    x_train: Optional[Tensor],
    n_future: int,
    n_draws: int,
    seed: int,
    noise_seed: int,
    n_paths_ci: Optional[int] = None,
    sine_fit: Optional[Dict[str, Any]] = None,
    *,
    ci: float = 0.95,
    y_train: Optional[np.ndarray] = None,
    residual_horizon: Optional[int] = None,
) -> Dict[str, Any]:
    from .state import _synthetic_covariate_extended

    n_train = int(model.n)
    n_total = n_train + int(n_future)
    x_full = None
    if x_train is not None and len(x_train) >= n_total:
        x_full = x_train.detach().cpu().numpy()[:n_total]
    elif x_train is not None and len(x_train) >= n_train:
        x_full = _synthetic_covariate_extended(n_total + n_train, sine_fit=sine_fit)
        x_full[:n_train] = x_train.detach().cpu().numpy()[:n_train]
    state = as_forecast_state(
        model,
        x_full=x_full,
        sine_fit=sine_fit if sine_fit is not None else getattr(model, "_sine_fit", None),
    )
    fc = forecast_from_state(
        state,
        int(n_future),
        n_draws=n_draws,
        n_paths_ci=n_paths_ci,
        seed=seed,
        noise_seed=noise_seed,
        ci=ci,
    )
    if residual_horizon is not None and y_train is not None:
        y_hat_tr = state.predict_train(state.p)
        res_fit = fit_residual_smooth(
            np.asarray(y_train, dtype=np.float64)[:n_train] - y_hat_tr,
            horizon=int(residual_horizon),
        )
        fc = apply_residual_forecast(fc, res_fit, n_train, fc["sigma_y"], noise_seed=noise_seed)
    elif getattr(model, "residual_smooth", None) is not None:
        fc = apply_residual_forecast(
            fc, model.residual_smooth, n_train, fc["sigma_y"], noise_seed=noise_seed
        )
    return fc


def predict_forecast_realisations_torch(
    model: Optional[Any] = None,
    x_train: Optional[Tensor] = None,
    n_future: int = 0,
    n_draws: int = 400,
    n_paths_ci: Optional[int] = None,
    seed: int = 0,
    noise_seed: int = 2,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
    *,
    ci: float = 0.95,
    fit: Optional[Dict[str, Any]] = None,
    sine_fit: Optional[Dict[str, Any]] = None,
    t_idx: Optional[np.ndarray] = None,
    t_norm: Optional[np.ndarray] = None,
    z_full: Optional[np.ndarray] = None,
    n_obs: Optional[int] = None,
    y_train: Optional[np.ndarray] = None,
    residual_horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """Sample warp-path forecast realisations and terror/error/combined bands."""
    del path_anchor
    if fit is not None:
        if t_idx is None or t_norm is None or z_full is None or n_obs is None:
            raise ValueError("calendar forecast requires fit, t_idx, t_norm, z_full, n_obs")
        n_total = len(z_full)
        n_future_use = int(n_future) if n_future else n_total - int(n_obs)
        state = as_forecast_state(
            fit,
            t_idx_full=t_idx,
            t_norm_full=t_norm,
            z_full=z_full,
            n_obs=int(n_obs),
        )
        fc = forecast_from_state(
            state,
            n_future_use,
            n_draws=n_draws,
            n_paths_ci=n_paths_ci,
            seed=seed,
            noise_seed=noise_seed,
            ci=ci,
        )
        res_fit = fit.get("residual_smooth")
        if res_fit is None and y_train is not None and residual_horizon is not None:
            y_hat_tr = state.predict_train(state.p)
            res_fit = fit_residual_smooth(
                np.asarray(y_train, dtype=np.float64)[: int(n_obs)] - y_hat_tr,
                horizon=int(residual_horizon),
            )
        if res_fit is not None:
            fc = apply_residual_forecast(
                fc, res_fit, int(n_obs), fc["sigma_y"], noise_seed=noise_seed
            )
        return fc

    if model is None:
        raise ValueError("pass model (WarpModel) or fit=... (calendar observation)")
    return _predict_forecast_model(
        model,
        x_train,
        n_future,
        n_draws,
        seed,
        noise_seed,
        n_paths_ci=n_paths_ci,
        sine_fit=sine_fit,
        ci=ci,
        y_train=y_train,
        residual_horizon=residual_horizon,
    )
