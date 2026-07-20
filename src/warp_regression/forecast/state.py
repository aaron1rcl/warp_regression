"""Forecast state construction and path-based forecasting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..core.path import DEFAULT_PATH_ANCHOR, PathAnchor, point_forecast_path
from ..core.warp import (
    soft_warp_sine_torch,
    soft_warp_torch,
)
from .bands import build_forecast_bands
from .components import observation_components, predict_components
from .paths import sample_warp_paths_future, sample_warp_paths_future_knots


PredictFn = Callable[[np.ndarray], np.ndarray]
ExtendPredictFn = Callable[[np.ndarray, int], PredictFn]


def _synthetic_covariate_extended(
    n_total: int,
    sine_fit: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Extend a unit-period sine on calendar grid ``(i+1)/100`` (synthetic demo)."""
    if sine_fit:
        z_full = sine_fit.get("z_full")
        if z_full is not None and len(z_full) >= n_total:
            return np.asarray(z_full[:n_total], dtype=np.float64)
        from ..covariates.sine import sine_from_fit

        t_ext = np.arange(1, n_total + 1, dtype=np.float64) / 100.0
        return sine_from_fit(t_ext, sine_fit)
    idx = np.arange(1, n_total + 1, dtype=np.float64)
    return np.sin(2.0 * np.pi * idx / 100.0)


@dataclass
class ForecastState:
    """Normalized fit state for series-agnostic forecasting."""

    p: np.ndarray
    sigma_t: float
    sigma_y: float
    n_knots: int
    n_train: int
    path_mode: str
    path_anchor: PathAnchor
    predict_train: PredictFn
    extend_predict: ExtendPredictFn
    B: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)




def _dual_predict_fns(
    z1: np.ndarray,
    z2: np.ndarray,
    observation: Any,
) -> Tuple[PredictFn, Callable[[Tensor], Tensor]]:
    """Predict helpers for dual-sine observation models."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    z1_t = torch.tensor(z1, dtype=torch.float32)
    z2_t = torch.tensor(z2, dtype=torch.float32)

    def predict_np(p: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return predict_torch(torch.tensor(p, dtype=torch.float32)).detach().numpy()

    def predict_torch(p: Tensor) -> Tensor:
        z1w = soft_warp_torch(z1_t, p)
        z2w = soft_warp_torch(z2_t, p)
        return observation({"z1": z1w, "z2": z2w}, {})

    return predict_np, predict_torch


def _as_forecast_state_dual(
    fit: Dict[str, Any],
    *,
    z1_full: np.ndarray,
    z2_full: np.ndarray,
    n_train: Optional[int] = None,
) -> ForecastState:
    warp = fit["warp"]
    p = np.asarray(warp["p"], dtype=np.float64)
    n_tr = int(n_train if n_train is not None else len(p))
    p = p[:n_tr]
    n_knots = int(warp.get("n_knots", fit.get("n_knots", 14)))
    B = warp.get("B")
    if B is not None:
        B = np.asarray(B, dtype=np.float64)
    z1_full = np.asarray(z1_full, dtype=np.float64)
    z2_full = np.asarray(z2_full, dtype=np.float64)
    obs = fit.get("observation")
    if obs is None:
        raise ValueError("dual fit state requires fit['observation']")
    predict_train, predict_train_torch = _dual_predict_fns(
        z1_full[:n_tr], z2_full[:n_tr], obs
    )

    def extend_predict(p_tr: np.ndarray, n_future: int) -> PredictFn:
        n_total = n_tr + int(n_future)
        if len(z1_full) < n_total or len(z2_full) < n_total:
            raise ValueError("z1_full/z2_full must cover n_train + n_future")
        predict_full, _ = _dual_predict_fns(
            z1_full[:n_total], z2_full[:n_total], obs
        )
        return predict_full

    return ForecastState(
        p=p,
        sigma_t=float(warp["sigma_t"]),
        sigma_y=float(warp["sigma_y"]),
        n_knots=n_knots,
        n_train=n_tr,
        path_mode="identity",
        path_anchor=DEFAULT_PATH_ANCHOR,
        predict_train=predict_train,
        extend_predict=extend_predict,
        B=B,
        meta={
            "kind": fit.get("kind", "dual"),
            "fit": fit,
            "predict_train_torch": predict_train_torch,
        },
    )


def _as_forecast_state_calendar(
    fit: Dict[str, Any],
    *,
    t_idx_full: np.ndarray,
    t_norm_full: np.ndarray,
    z_full: np.ndarray,
    n_obs: Optional[int] = None,
) -> ForecastState:
    warp = fit["warp"]
    p = np.asarray(warp["p"], dtype=np.float64)
    n_tr = int(n_obs if n_obs is not None else len(p))
    p = p[:n_tr]
    n_knots = int(fit.get("n_knots", warp.get("n_knots", 14)))
    B = warp.get("B")
    if B is not None:
        B = np.asarray(B, dtype=np.float64)
    obs = fit.get("observation")
    if obs is None:
        raise ValueError("calendar fit state requires fit['observation']")
    sine_fit = fit.get("sine_fit")
    n_cal = int(fit.get("n_calendar", n_tr))
    t_idx_full = np.asarray(t_idx_full, dtype=np.float64)
    t_norm_full = np.asarray(t_norm_full, dtype=np.float64)
    z_full = np.asarray(z_full, dtype=np.float64)
    pred_kw: Dict[str, Any] = {}
    if sine_fit is not None:
        pred_kw.update({"sine_fit": sine_fit, "n_norm": n_cal})
    t_idx_tr = torch.tensor(t_idx_full[:n_tr], dtype=torch.float32)
    t_norm_tr = torch.tensor(t_norm_full[:n_tr], dtype=torch.float32)
    z_tr_t = torch.tensor(z_full[:n_tr], dtype=torch.float32)

    def predict_train(p_tr: np.ndarray) -> np.ndarray:
        return predict_components(
            obs,
            t_idx_full[:n_tr],
            t_norm_full[:n_tr],
            z_full[:n_tr],
            p_tr,
            **pred_kw,
        )[2]

    def predict_train_torch(p: Tensor) -> Tensor:
        if sine_fit is not None:
            zw = soft_warp_sine_torch(
                p,
                n_cal,
                sine_fit["omega"],
                sine_fit["phase"],
                time_scale=sine_fit.get("time_scale", 1.0),
                t_shift=sine_fit.get("t_shift", 0.0),
            )
        else:
            zw = soft_warp_torch(z_tr_t, p)
        return obs({"z": zw}, {"t_idx": t_idx_tr, "t_norm": t_norm_tr})

    def extend_predict(p_tr: np.ndarray, n_future: int) -> PredictFn:
        n_total = n_tr + int(n_future)
        if len(z_full) < n_total:
            raise ValueError("z_full must cover n_obs + n_future")

        def predict_full(p_full: np.ndarray) -> np.ndarray:
            return predict_components(
                obs,
                t_idx_full[:n_total],
                t_norm_full[:n_total],
                z_full[:n_total],
                p_full,
                **pred_kw,
            )[2]

        return predict_full

    return ForecastState(
        p=p,
        sigma_t=float(warp["sigma_t"]),
        sigma_y=float(warp["sigma_y"]),
        n_knots=n_knots,
        n_train=n_tr,
        path_mode="identity",
        path_anchor=DEFAULT_PATH_ANCHOR,
        predict_train=predict_train,
        extend_predict=extend_predict,
        B=B,
        meta={
            "kind": "calendar_obs",
            "fit": fit,
            "predict_train_torch": predict_train_torch,
        },
    )


def as_forecast_state(
    source: Union[Dict[str, Any], Any],
    *,
    x_full: Optional[np.ndarray] = None,
    z1_full: Optional[np.ndarray] = None,
    z2_full: Optional[np.ndarray] = None,
    t_idx_full: Optional[np.ndarray] = None,
    t_norm_full: Optional[np.ndarray] = None,
    z_full: Optional[np.ndarray] = None,
    n_train: Optional[int] = None,
    n_obs: Optional[int] = None,
    sine_fit: Optional[Dict[str, Any]] = None,
    covariates_full: Optional[Dict[str, np.ndarray]] = None,
    calendar_full: Optional[Dict[str, np.ndarray]] = None,
) -> ForecastState:
    """Normalize a ``WarpModel`` or fit dict into ``ForecastState``."""
    from ..model import WarpModel

    if isinstance(source, WarpModel):
        return source.as_forecast_state(
            covariates_full=covariates_full,
            calendar_full=calendar_full,
            sine_fit=sine_fit,
            n_train=n_train,
            n_obs=n_obs,
            z1_full=z1_full,
            z2_full=z2_full,
            x_full=x_full,
            t_idx_full=t_idx_full,
            t_norm_full=t_norm_full,
            z_full=z_full,
        )

    fit = source
    kind = fit.get("kind")
    if kind in ("dual_linear", "dual_mlp") or (
        z1_full is not None and z2_full is not None and "observation" in fit
    ):
        if z1_full is None or z2_full is None:
            raise ValueError("dual fit state requires z1_full and z2_full")
        return _as_forecast_state_dual(fit, z1_full=z1_full, z2_full=z2_full, n_train=n_train)

    if kind in ("log_trend_sine", "calendar_obs") or (
        t_idx_full is not None and z_full is not None and "observation" in fit
    ):
        if t_idx_full is None or t_norm_full is None or z_full is None:
            raise ValueError("calendar observation fit requires t_idx_full, t_norm_full, z_full")
        return _as_forecast_state_calendar(
            fit,
            t_idx_full=t_idx_full,
            t_norm_full=t_norm_full,
            z_full=z_full,
            n_obs=n_obs if n_obs is not None else n_train,
        )

    raise ValueError("unrecognized forecast source; pass WarpModel or a known fit dict")



def forecast_from_state(
    state: ForecastState,
    n_future: int,
    *,
    n_draws: int = 400,
    n_paths_ci: Optional[int] = None,
    seed: int = 0,
    noise_seed: int = 2,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """Sample future warps from the fitted path.

    Start-pin: MAP train path is kept through the second-to-last knot; the last
    train knot gets a terror-scale kick, then the usual future knot RW continues
    from there. ``ci`` sets the nominal central coverage for predictive bands
    (default 0.95).

    Two ensembles on purpose:
    * knot-level RW (``paths_p``) → smooth predictive curves ``preds``
    * per-index RW (``paths_p_ci``) → denser sample for terror / combined CI bands
    """
    n_train = int(state.n_train)
    n_future = int(n_future)
    n_total = n_train + n_future
    n_ci = n_paths_ci if n_paths_ci is not None else max(n_draws, 400)
    predict_full = state.extend_predict(state.p, n_future)
    path_anchor = state.path_anchor

    paths_p = sample_warp_paths_future_knots(
        state.p,
        n_total,
        n_train,
        state.sigma_t,
        state.n_knots,
        n_draws,
        seed,
        path_anchor=path_anchor,
    )
    paths_p_ci = sample_warp_paths_future(
        state.p,
        n_total,
        n_train,
        state.sigma_t,
        state.n_knots,
        n_ci,
        seed + 1000,
        path_anchor=path_anchor,
    )
    p_det = point_forecast_path(
        state.p, n_total, path_mode=state.path_mode, path_anchor=path_anchor
    )
    preds = np.stack([predict_full(paths_p[k]) for k in range(n_draws)])
    preds_ci = np.stack([predict_full(paths_p_ci[k]) for k in range(n_ci)])
    y_point = predict_full(p_det)
    bands = build_forecast_bands(
        preds_ci, y_point, state.sigma_y, ci=ci, noise_seed=noise_seed
    )
    return {
        "preds": preds,
        "preds_ci": preds_ci,
        "y_point": y_point,
        "paths_p": paths_p,
        "paths_p_ci": paths_p_ci,
        "p_det": p_det,
        "bands": bands,
        "ci": float(ci),
        "n_train": n_train,
        "n_obs": n_train,
        "n_total": n_total,
        "n_future": n_future,
        "sigma_y": float(state.sigma_y),
        "sigma_t": float(state.sigma_t),
        "path_anchor": path_anchor,
        "terminal_offset": float(state.p[-1] - (n_train - 1)) if state.path_mode == "identity" else float(state.p[-1]),
        "quantiles": {
            "q10": bands["t_q_lo"],
            "q50": bands["t_q50"],
            "q90": bands["t_q_hi"],
        },
    }



