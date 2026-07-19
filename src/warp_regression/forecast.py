from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor

from .constants import DEFAULT_PATH_ANCHOR, PathAnchor
from .core.path import (
    knot_positions,
    point_forecast_path,
)
from .core.warp import (
    soft_warp_numpy,
    soft_warp_prefit_sine_numpy,
    soft_warp_prefit_sine_torch,
    soft_warp_torch,
)
from .readouts.parametric import WarpParametricModel, evaluate_model


def _synthetic_driver_extended(
    n_total: int,
    sine_fit: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Extend the warp driver on the synthetic calendar grid ``(i+1)/100``."""
    if sine_fit:
        z_full = sine_fit.get("z_full")
        if z_full is not None and len(z_full) >= n_total:
            return np.asarray(z_full[:n_total], dtype=np.float64)
        from .readouts.log_trend_sine import sine_from_fit

        t_ext = np.arange(1, n_total + 1, dtype=np.float64) / 100.0
        return sine_from_fit(t_ext, sine_fit)
    idx = np.arange(1, n_total + 1, dtype=np.float64)
    return np.sin(2.0 * np.pi * idx / 100.0)


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
    """Terror, error, and combined predictive bands at coverage ``ci``.

    Terror: percentiles of warp-path forecasts (timing uncertainty through readout).
    Error: diagnostic band around the point forecast (observation noise only).
    Combined: percentiles of ``r + e`` where ``r`` is a terror path and
    ``e_t ~ N(μ, σ_t²)`` is drawn iid per time step (and independently per
    realisation). If ``err_margin`` is time-varying, ``σ_t = err_margin_t / z_err``.

    ``ci`` is the nominal central coverage (default 0.95). Optional ``pct_lo`` /
    ``pct_hi`` / ``z_err`` override the values derived from ``ci``.
    """
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
    """Map homoscedastic log1p observation noise to count-scale error bands.

    For large ``μ = log1p(c)``, a fixed log1p increment ``z·σ`` is a relative
    fluctuation ``≈ z·σ/μ`` on the count scale: ``ĉ·(1 ± z·σ/μ)``. Using
    ``expm1(μ ± zσ)`` instead over-widens the orange band versus terror on the
    count-scale plots (especially at cycle peaks).
    """
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


# ---------------------------------------------------------------------------
# Forecast state + sampling
# ---------------------------------------------------------------------------

PredictFn = Callable[[np.ndarray], np.ndarray]
ExtendPredictFn = Callable[[np.ndarray, int], PredictFn]


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
    readout: Any,
    *,
    reverse_path: bool = False,
) -> Tuple[PredictFn, Callable[[Tensor], Tensor]]:
    """Predict helpers for prepare/forecast.

    Prepared paths are calendar-start-pinned with ``stored == applied``, so
    soft-warp must not reverse (``reverse_path=False``). Reversing would map
    future RW noise onto the past and collapse forecast terror bands.
    """
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    z1_t = torch.tensor(z1, dtype=torch.float32)
    z2_t = torch.tensor(z2, dtype=torch.float32)

    if isinstance(readout, dict) and "f" in readout and "g" in readout:
        f_net, g_net = readout["f"], readout["g"]
        f_net.eval()
        g_net.eval()

        def predict_np(p: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return predict_torch(torch.tensor(p, dtype=torch.float32)).numpy()

        def predict_torch(p: Tensor) -> Tensor:
            z1w = soft_warp_torch(z1_t, p, reverse_path=reverse_path)
            z2w = soft_warp_torch(z2_t, p, reverse_path=reverse_path)
            return f_net(z1w) + g_net(z2w)

        return predict_np, predict_torch

    def predict_np(p: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return predict_torch(torch.tensor(p, dtype=torch.float32)).detach().numpy()

    def predict_torch(p: Tensor) -> Tensor:
        z1w = soft_warp_torch(z1_t, p, reverse_path=reverse_path)
        z2w = soft_warp_torch(z2_t, p, reverse_path=reverse_path)
        return readout(z1w, z2w)

    return predict_np, predict_torch


def _as_forecast_state_parametric(
    model: WarpParametricModel,
    *,
    x_full: Optional[np.ndarray] = None,
    sine_fit: Optional[Dict[str, Any]] = None,
) -> ForecastState:
    with torch.no_grad():
        p = model.path().detach().cpu().numpy()
        sigma_y = float(torch.exp(model.log_sigma))
        sigma_t = float(torch.exp(model.log_sigma_t))
        B = model.B.detach().cpu().numpy().copy()
    n_train = int(model.n)
    path_mode = model.path_mode
    fit_anchor: PathAnchor = getattr(model, "path_anchor", DEFAULT_PATH_ANCHOR)
    use_prefit_sine = (
        sine_fit is not None and sine_fit.get("dt") is not None and sine_fit.get("t0") is not None
    )
    if not use_prefit_sine:
        if x_full is not None and len(x_full) >= n_train:
            x_tr = np.asarray(x_full[:n_train], dtype=np.float64)
        else:
            x_tr = _synthetic_driver_extended(n_train, sine_fit=sine_fit)
        x_tr_t = torch.tensor(x_tr, dtype=torch.float32)

    def _readout(z: Tensor) -> Tensor:
        return torch.exp(model.log_A) * z + model.C

    def predict_train(p_tr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            p_t = torch.tensor(p_tr, dtype=torch.float32)
            if use_prefit_sine:
                z = soft_warp_prefit_sine_torch(p_t, sine_fit, path_mode=path_mode)
            else:
                z = soft_warp_torch(x_tr_t, p_t, reverse_path=False, path_mode=path_mode)
            return _readout(z).detach().cpu().numpy()

    def predict_train_torch(p: Tensor) -> Tensor:
        if use_prefit_sine:
            z = soft_warp_prefit_sine_torch(p, sine_fit, path_mode=path_mode)
        else:
            z = soft_warp_torch(x_tr_t, p, reverse_path=False, path_mode=path_mode)
        return _readout(z)

    def extend_predict(p_tr: np.ndarray, n_future: int) -> PredictFn:
        n_total = n_train + int(n_future)
        if not use_prefit_sine:
            if x_full is not None and len(x_full) >= n_total:
                x_ext = np.asarray(x_full[:n_total], dtype=np.float64)
            else:
                # Headroom so expansion can pull known future sine into the window.
                x_ext = _synthetic_driver_extended(n_total + n_train, sine_fit=sine_fit)
            x_t = torch.tensor(x_ext, dtype=torch.float32)

        def predict_full(p_full: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                p_t = torch.tensor(p_full, dtype=torch.float32)
                if use_prefit_sine:
                    z = soft_warp_prefit_sine_torch(p_t, sine_fit, path_mode=path_mode)
                else:
                    z = soft_warp_torch(x_t, p_t, reverse_path=False, path_mode=path_mode)
                return _readout(z).detach().cpu().numpy()

        return predict_full

    return ForecastState(
        p=np.asarray(p, dtype=np.float64),
        sigma_t=sigma_t,
        sigma_y=sigma_y,
        n_knots=int(model.n_knots),
        n_train=n_train,
        path_mode=path_mode,
        path_anchor=fit_anchor,
        predict_train=predict_train,
        extend_predict=extend_predict,
        B=B,
        meta={"kind": "parametric", "sine_fit": sine_fit, "predict_train_torch": predict_train_torch},
    )


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
    readout = fit["readout"]
    predict_train, predict_train_torch = _dual_predict_fns(
        z1_full[:n_tr], z2_full[:n_tr], readout, reverse_path=False
    )

    def extend_predict(p_tr: np.ndarray, n_future: int) -> PredictFn:
        n_total = n_tr + int(n_future)
        if len(z1_full) < n_total or len(z2_full) < n_total:
            raise ValueError("z1_full/z2_full must cover n_train + n_future")
        predict_full, _ = _dual_predict_fns(
            z1_full[:n_total], z2_full[:n_total], readout, reverse_path=False
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


def _as_forecast_state_bitcoin(
    fit: Dict[str, Any],
    *,
    t_idx_full: np.ndarray,
    t_norm_full: np.ndarray,
    z_full: np.ndarray,
    n_obs: Optional[int] = None,
) -> ForecastState:
    from .readouts.log_trend_sine import predict_components
    from .core.warp import soft_warp_sine_torch

    warp = fit["warp"]
    p = np.asarray(warp["p"], dtype=np.float64)
    n_tr = int(n_obs if n_obs is not None else len(p))
    p = p[:n_tr]
    n_knots = int(fit.get("n_knots", warp.get("n_knots", 14)))
    B = warp.get("B_spline", warp.get("B"))
    if B is not None:
        B = np.asarray(B, dtype=np.float64)
    readout = fit["readout"]
    sine_fit = fit.get("sine_fit")
    n_cal = int(fit.get("n_calendar", n_tr))
    t_idx_full = np.asarray(t_idx_full, dtype=np.float64)
    t_norm_full = np.asarray(t_norm_full, dtype=np.float64)
    z_full = np.asarray(z_full, dtype=np.float64)
    pred_kw: Dict[str, Any] = {"path_anchor": DEFAULT_PATH_ANCHOR}
    if sine_fit is not None:
        pred_kw.update({"sine_fit": sine_fit, "n_norm": n_cal})
    t_idx_tr = torch.tensor(t_idx_full[:n_tr], dtype=torch.float32)
    t_norm_tr = torch.tensor(t_norm_full[:n_tr], dtype=torch.float32)
    z_tr_t = torch.tensor(z_full[:n_tr], dtype=torch.float32)

    def predict_train(p_tr: np.ndarray) -> np.ndarray:
        return predict_components(
            readout,
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
                reverse_path=False,
            )
        else:
            zw = soft_warp_torch(z_tr_t, p, reverse_path=False)
        return readout(t_idx_tr, t_norm_tr, zw)

    def extend_predict(p_tr: np.ndarray, n_future: int) -> PredictFn:
        n_total = n_tr + int(n_future)
        if len(z_full) < n_total:
            raise ValueError("z_full must cover n_obs + n_future")

        def predict_full(p_full: np.ndarray) -> np.ndarray:
            return predict_components(
                readout,
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
            "kind": "log_trend_sine",
            "fit": fit,
            "predict_train_torch": predict_train_torch,
        },
    )


def as_forecast_state(
    source: Union[WarpParametricModel, Dict[str, Any]],
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
) -> ForecastState:
    """Normalize a parametric model or fit dict into ``ForecastState``."""
    if isinstance(source, WarpParametricModel):
        return _as_forecast_state_parametric(source, x_full=x_full, sine_fit=sine_fit)

    fit = source
    kind = fit.get("kind")
    if kind in ("dual_linear", "dual_mlp") or (z1_full is not None and z2_full is not None and "readout" in fit):
        if z1_full is None or z2_full is None:
            raise ValueError("dual fit state requires z1_full and z2_full")
        return _as_forecast_state_dual(fit, z1_full=z1_full, z2_full=z2_full, n_train=n_train)

    if kind == "log_trend_sine" or (t_idx_full is not None and z_full is not None and "readout" in fit):
        if t_idx_full is None or t_norm_full is None or z_full is None:
            raise ValueError("bitcoin fit state requires t_idx_full, t_norm_full, z_full")
        return _as_forecast_state_bitcoin(
            fit,
            t_idx_full=t_idx_full,
            t_norm_full=t_norm_full,
            z_full=z_full,
            n_obs=n_obs if n_obs is not None else n_train,
        )

    raise ValueError("unrecognized forecast source; pass WarpParametricModel or a known fit dict")



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



# ---------------------------------------------------------------------------
# Lynx forecast
# ---------------------------------------------------------------------------


def per_index_rw_sigma(sigma_t: float, n: int, n_knots: int) -> float:
    """Per-index RW std consistent with knot-segment terror scale sigma_t."""
    _, rw = knot_positions(n, n_knots)
    return float(sigma_t * np.sqrt(max(rw, 1e-8)))


def per_year_sigma(sigma_t: float, n_train: int, n_knots: int) -> float:
    return per_index_rw_sigma(sigma_t, n_train, n_knots)


def sample_warp_paths_future(
    p_train: np.ndarray,
    n_total: int,
    n_train: int,
    sigma_t: float,
    n_knots: int,
    n_paths: int = 500,
    seed: int = 0,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    """Per-index future RW, with a terror-scale kick at the free train end.

    Under ``path_anchor='start'`` the MAP train path is kept through
    ``n_train-2``, the terminal offset gets one N(0, sigma) kick, and the
    future RW continues from that kicked end (last index not frozen).
    """
    rng = np.random.default_rng(seed)
    sigma_step = per_year_sigma(sigma_t, n_train, n_knots)
    offset_train = p_train - np.arange(n_train, dtype=np.float64)
    paths = np.zeros((n_paths, n_total), dtype=np.float64)
    idx = np.arange(n_total, dtype=np.float64)
    for k in range(n_paths):
        off = np.zeros(n_total)
        off[:n_train] = offset_train
        if path_anchor == "start" and n_train >= 1:
            off[n_train - 1] = offset_train[n_train - 1] + rng.normal(0.0, sigma_step)
        for i in range(n_train, n_total):
            off[i] = off[i - 1] + rng.normal(0.0, sigma_step)
        paths[k] = idx + off
        if path_anchor == "end":
            paths[k, :n_train] = p_train
            paths[k, n_train - 1] = float(n_train - 1)
        else:
            if n_train > 1:
                paths[k, : n_train - 1] = p_train[: n_train - 1]
            paths[k, 0] = 0.0
    return paths


def extended_knot_positions(n_train: int, n_total: int, n_knots: int) -> np.ndarray:
    """Knot grid: train ``linspace(0, n_train-1, n_knots)``, extended with same spacing."""
    train_knots = np.linspace(0.0, float(n_train - 1), n_knots)
    if n_total <= n_train:
        return train_knots
    step = (n_train - 1) / max(n_knots - 1, 1)
    xs = list(train_knots)
    x = float(n_train - 1)
    while x < n_total - 1 - 1e-9:
        x += step
        xs.append(min(x, float(n_total - 1)))
        if xs[-1] >= n_total - 1 - 1e-9:
            break
    if xs[-1] < n_total - 1 - 1e-9:
        xs.append(float(n_total - 1))
    return np.unique(np.asarray(xs, dtype=np.float64))


def sample_warp_paths_future_knots(
    p_train: np.ndarray,
    n_total: int,
    n_train: int,
    sigma_t: float,
    n_knots: int,
    n_paths: int = 500,
    seed: int = 0,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    """Future warp paths via knot-level RW + piecewise-linear offset (train knot spacing).

    Under ``path_anchor='start'``:
      1. kick the last train knot by N(0, sigma_knot),
      2. continue the knot RW from that kicked knot into the future
         (last train knot is not frozen at the MAP).

    Earlier train knots stay at the MAP. Pair with ``sample_warp_paths_future``
    for denser per-index RW when building terror/confidence bands.
    """
    rng = np.random.default_rng(seed)
    _, rw_interval = knot_positions(n_train, n_knots)
    sigma_knot = float(sigma_t * rw_interval)
    knot_x = extended_knot_positions(n_train, n_total, n_knots)
    idx_train = np.arange(n_train, dtype=np.float64)
    off_train = p_train - idx_train
    off_anchor = np.interp(knot_x, idx_train, off_train)
    n_train_knots = len(np.linspace(0.0, float(n_train - 1), n_knots))
    idx_full = np.arange(n_total, dtype=np.float64)
    # Keep MAP through the second-to-last train knot; last knot may move.
    freeze_n = int(np.floor(knot_x[max(n_train_knots - 2, 0)])) + 1 if n_train_knots >= 2 else 1
    freeze_n = min(max(freeze_n, 1), n_train)
    paths = np.zeros((n_paths, n_total), dtype=np.float64)
    for k in range(n_paths):
        off_k = off_anchor.copy()
        if path_anchor == "start" and n_train_knots >= 1:
            i_last = n_train_knots - 1
            off_k[i_last] = off_anchor[i_last] + rng.normal(0.0, sigma_knot)
            for j in range(n_train_knots, len(knot_x)):
                off_k[j] = off_k[j - 1] + rng.normal(0.0, sigma_knot)
        else:
            for j in range(n_train_knots, len(knot_x)):
                off_k[j] = off_k[j - 1] + rng.normal(0.0, sigma_knot)
        off_full = np.interp(idx_full, knot_x, off_k)
        paths[k] = idx_full + off_full
        if path_anchor == "end":
            paths[k, :n_train] = p_train
            paths[k, n_train - 1] = float(n_train - 1)
        else:
            paths[k, :freeze_n] = p_train[:freeze_n]
            paths[k, 0] = 0.0
    return paths


@torch.no_grad()
def predict_log_from_path(z1: np.ndarray, z2: np.ndarray, p: np.ndarray, readout: Dict[str, Any]) -> np.ndarray:
    f_net, g_net = readout["f"], readout["g"]
    f_net.eval()
    g_net.eval()
    z1w, z2w = soft_warp_numpy(z1, p), soft_warp_numpy(z2, p)
    t1 = torch.tensor(z1w, dtype=torch.float32)
    t2 = torch.tensor(z2w, dtype=torch.float32)
    return (f_net(t1) + g_net(t2)).numpy()


def forecast_lynx_holdout_paths(
    fit: Dict[str, Any],
    z1: np.ndarray,
    z2: np.ndarray,
    split: Dict[str, Any],
    n_paths: int = 500,
    seed: int = 0,
    noise_seed: int = 2,
    *,
    ci: float = 0.95,
    y_train: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    n_total, n_train = len(z1), int(split["n_train"])
    test_idx, train_idx = split["test_idx"], split["train_idx"]
    n_future = n_total - n_train
    state = as_forecast_state(fit, z1_full=z1, z2_full=z2, n_train=n_train)
    out = forecast_from_state(
        state,
        n_future,
        n_draws=n_paths,
        n_paths_ci=n_paths,
        seed=seed,
        noise_seed=noise_seed,
        ci=ci,
    )

    bands = out["bands"]
    y_point = out["y_point"]
    y_tr, y_te = fit.get("_y_log_train"), fit.get("_y_log_test")
    if y_tr is not None:
        out["corr_log_train"] = float(np.corrcoef(y_point[train_idx], y_tr)[0, 1])
        out["corr_log_median_train"] = float(np.corrcoef(bands["t_q50"][train_idx], y_tr)[0, 1])
    if y_te is not None:
        out["corr_log_test"] = float(np.corrcoef(y_point[test_idx], y_te)[0, 1])
        out["corr_log_median_test"] = float(np.corrcoef(bands["t_q50"][test_idx], y_te)[0, 1])
        out["rmse_log_test"] = float(np.sqrt(np.mean((y_point[test_idx] - y_te) ** 2)))
        out["coverage_combined_test"] = interval_coverage(
            y_te, bands["c_q_lo"][test_idx], bands["c_q_hi"][test_idx]
        )
        out["coverage_combined_95_test"] = out["coverage_combined_test"]  # back-compat
        out["coverage_nominal"] = float(ci)
    return out


def predict_realisations_torch(
    model: WarpParametricModel,
    x: Tensor,
    n_draws: int = 400,
    horizon: int = 60,
    seed: int = 0,
) -> np.ndarray:
    """Delegate to :func:`warp_regression.readouts.parametric.predict_realisations_torch`."""
    from .readouts.parametric import predict_realisations_torch as _predict

    return _predict(model, x, n_draws=n_draws, horizon=horizon, seed=seed)


def _predict_forecast_parametric(
    model: WarpParametricModel,
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
    n_train = model.n
    n_total = n_train + int(n_future)
    x_full = None
    if x_train is not None and len(x_train) >= n_total:
        x_full = x_train.detach().cpu().numpy()[:n_total]
    elif x_train is not None and len(x_train) >= n_train:
        x_full = _synthetic_driver_extended(n_total + n_train, sine_fit=sine_fit)
        x_full[:n_train] = x_train.detach().cpu().numpy()[:n_train]
    state = as_forecast_state(model, x_full=x_full, sine_fit=sine_fit)
    fc = forecast_from_state(
        state,
        int(n_future),
        n_draws=n_draws,
        n_paths_ci=n_paths_ci,
        seed=seed,
        noise_seed=noise_seed,
        ci=ci,
    )
    # Optional residual MA layer (train residuals → flat future correction).
    if y_train is not None and residual_horizon is not None:
        from .residual_smooth import apply_residual_forecast, fit_residual_smooth

        y_hat_tr = state.predict_train(state.p)
        res_fit = fit_residual_smooth(
            np.asarray(y_train, dtype=np.float64)[:n_train] - y_hat_tr,
            horizon=int(residual_horizon),
        )
        fc = apply_residual_forecast(fc, res_fit, n_train, fc["sigma_y"], noise_seed=noise_seed)
    elif hasattr(model, "residual_smooth") and getattr(model, "residual_smooth") is not None:
        from .residual_smooth import apply_residual_forecast

        fc = apply_residual_forecast(
            fc, model.residual_smooth, n_train, fc["sigma_y"], noise_seed=noise_seed
        )
    return fc


def predict_forecast_realisations_torch(
    model: Optional[WarpParametricModel] = None,
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
    """Sample warp-path forecast realisations and terror/error/combined bands.

    ``ci`` is the nominal central coverage for percentile bands (default 0.95).
    Samples from the fitted (start-pinned) path with frozen σ_t/σ_y. When
    ``sine_fit`` carries ``t0``/``dt``, the prefit sine is evaluated analytically
    so warp expansion can read known future cycle values. Optionally apply a
    residual MA layer via ``residual_horizon`` or ``fit['residual_smooth']``.
    """
    del path_anchor  # path anchor comes from the fitted model/state
    if fit is not None:
        if t_idx is None or t_norm is None or z_full is None or n_obs is None:
            raise ValueError("Bitcoin forecast requires fit, t_idx, t_norm, z_full, n_obs")
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
            from .residual_smooth import fit_residual_smooth

            y_hat_tr = state.predict_train(state.p)
            res_fit = fit_residual_smooth(
                np.asarray(y_train, dtype=np.float64)[: int(n_obs)] - y_hat_tr,
                horizon=int(residual_horizon),
            )
        if res_fit is not None:
            from .residual_smooth import apply_residual_forecast

            fc = apply_residual_forecast(
                fc,
                res_fit,
                int(n_obs),
                fc["sigma_y"],
                noise_seed=noise_seed,
            )
        return fc

    if model is None:
        raise ValueError("pass model (synthetic) or fit=... (bitcoin)")
    return _predict_forecast_parametric(
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

