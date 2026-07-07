from __future__ import annotations


from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor

from .constants import DEFAULT_PATH_ANCHOR, PathAnchor
from .core.path import knot_positions, point_forecast_path
from .core.warp import path_for_warp_numpy, soft_warp_numpy
from .readouts.parametric import WarpParametricModel, evaluate_model

def build_forecast_bands(
    paths_terror: np.ndarray,
    y_point: np.ndarray,
    sigma_y: float,
    pct_lo: float = 2.5,
    pct_hi: float = 97.5,
    z_err: float = 1.96,
    noise_seed: int = 2,
    err_margin: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Terror, error, and combined predictive bands at a common coverage level.

    Terror: percentiles of warp-path forecasts (timing uncertainty through readout).
    Error: diagnostic band around the point forecast (observation noise only).
    Combined: percentiles of ``paths_terror + ε`` with independent
    ``ε ~ N(0, σ_y²)`` on each path and index (joint Monte Carlo).
    """
    paths_terror = np.asarray(paths_terror, dtype=np.float64)
    y_point = np.asarray(y_point, dtype=np.float64)
    t_q_lo, t_q50, t_q_hi = np.percentile(paths_terror, [pct_lo, 50, pct_hi], axis=0)

    if err_margin is None:
        err_margin_arr = np.full_like(y_point, float(z_err * sigma_y), dtype=np.float64)
    else:
        err_margin_arr = np.asarray(err_margin, dtype=np.float64)
        if err_margin_arr.ndim == 0:
            err_margin_arr = np.full_like(y_point, float(err_margin_arr), dtype=np.float64)

    err_lo = y_point - err_margin_arr
    err_hi = y_point + err_margin_arr

    rng = np.random.default_rng(noise_seed)
    sigma_draw = err_margin_arr / z_err
    noise = rng.normal(0.0, 1.0, size=paths_terror.shape) * sigma_draw[np.newaxis, :]
    paths_combined = paths_terror + noise
    c_q_lo, c_q50, c_q_hi = np.percentile(paths_combined, [pct_lo, 50, pct_hi], axis=0)

    return {
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
    rng = np.random.default_rng(seed)
    sigma_step = per_year_sigma(sigma_t, n_train, n_knots)
    offset_train = p_train - np.arange(n_train, dtype=np.float64)
    paths = np.zeros((n_paths, n_total), dtype=np.float64)
    idx = np.arange(n_total, dtype=np.float64)
    for k in range(n_paths):
        off = np.zeros(n_total)
        off[:n_train] = offset_train
        for i in range(n_train, n_total):
            off[i] = off[i - 1] + rng.normal(0.0, sigma_step)
        paths[k] = idx + off
        paths[k, :n_train] = p_train
        if path_anchor == "end":
            paths[k, n_train - 1] = float(n_train - 1)
        else:
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

    Used for smooth single-path forecast realisations. Pair with ``sample_warp_paths_future``
    for per-index RW when building terror/confidence bands.
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
    paths = np.zeros((n_paths, n_total), dtype=np.float64)
    for k in range(n_paths):
        off_k = off_anchor.copy()
        for j in range(n_train_knots, len(knot_x)):
            off_k[j] = off_k[j - 1] + rng.normal(0.0, sigma_knot)
        off_full = np.interp(idx_full, knot_x, off_k)
        paths[k] = idx_full + off_full
        paths[k, :n_train] = p_train
        if path_anchor == "end":
            paths[k, n_train - 1] = float(n_train - 1)
        else:
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
) -> Dict[str, Any]:
    n_total, n_train = len(z1), split["n_train"]
    test_idx, train_idx = split["test_idx"], split["train_idx"]
    p_train = fit["warp"]["p"][:n_train]
    sigma_t = fit["warp"]["sigma_t"]
    sigma_y = fit["warp"]["sigma_y"]
    n_knots = fit["warp"].get("n_knots", 14)
    paths_p = sample_warp_paths_future(p_train, n_total, n_train, sigma_t, n_knots, n_paths, seed)
    preds = np.stack([predict_log_from_path(z1, z2, paths_p[k], fit["readout"]) for k in range(n_paths)])
    p_det = point_forecast_path(p_train, n_total)
    y_point = predict_log_from_path(z1, z2, p_det, fit["readout"])
    bands = build_forecast_bands(preds, y_point, sigma_y, noise_seed=noise_seed)
    q10, q50, q90 = bands["t_q_lo"], bands["t_q50"], bands["t_q_hi"]
    out = {
        "paths_p": paths_p,
        "preds": preds,
        "y_point": y_point,
        "n_train": n_train,
        "n_future": n_total - n_train,
        "sigma_t": sigma_t,
        "sigma_y": sigma_y,
        "bands": bands,
        "quantiles": {"q10": q10, "q50": q50, "q90": q90},
    }
    y_tr, y_te = fit.get("_y_log_train"), fit.get("_y_log_test")
    if y_tr is not None:
        out["corr_log_train"] = float(np.corrcoef(y_point[train_idx], y_tr)[0, 1])
        out["corr_log_median_train"] = float(np.corrcoef(bands["t_q50"][train_idx], y_tr)[0, 1])
    if y_te is not None:
        out["corr_log_test"] = float(np.corrcoef(y_point[test_idx], y_te)[0, 1])
        out["corr_log_median_test"] = float(np.corrcoef(bands["t_q50"][test_idx], y_te)[0, 1])
        out["rmse_log_test"] = float(np.sqrt(np.mean((y_point[test_idx] - y_te) ** 2)))
        in_combined = (y_te >= bands["c_q_lo"][test_idx]) & (y_te <= bands["c_q_hi"][test_idx])
        out["coverage_combined_95_test"] = float(in_combined.mean())
    return out


def predict_realisations_torch(
    model: WarpParametricModel,
    x: Tensor,
    n_draws: int = 400,
    horizon: int = 60,
    seed: int = 0,
) -> np.ndarray:
    """Sample path uncertainty over the first `horizon` time indices."""
    rng = np.random.default_rng(seed)
    n = model.n
    h = min(int(horizon), n)
    with torch.no_grad():
        p_fit = model.path().cpu().numpy()
        sigma_t = float(torch.exp(model.log_sigma_t))
    sigma_step = per_index_rw_sigma(sigma_t, n, model.n_knots)
    x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    idx = np.arange(n, dtype=np.float64)
    p_apply_fit = path_for_warp_numpy(p_fit, path_mode=model.path_mode)
    off_apply_fit = p_apply_fit - idx
    offset_stored_fit = p_fit - idx
    paths = np.zeros((n_draws, h))
    for k in range(n_draws):
        if model.path_mode == "identity":
            off_apply = off_apply_fit.copy()
            for j in range(1, h):
                off_apply[j] = off_apply[j - 1] + rng.normal(0.0, sigma_step)
            off_stored = offset_stored_fit.copy()
            for j in range(h):
                off_stored[int(n - 1 - j)] = off_apply[j]
            p = idx + off_stored
            p[0] = 0.0
        else:
            p = p_fit.copy()
            p_apply = path_for_warp_numpy(p, path_mode=model.path_mode)
            for j in range(1, h):
                p_apply[j] = p_apply[j - 1] + rng.normal(0.0, sigma_step)
            for j in range(h):
                p[int(n - 1 - j)] = p_apply[j]
        with torch.no_grad():
            y_full = model.predict(x_t, torch.tensor(p, dtype=torch.float32)).detach().cpu().numpy()
        paths[k] = y_full[:h]
    return paths


def _synthetic_driver_extended(
    n_total: int,
    sine_fit: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Extend the warp driver on the synthetic calendar grid ``(i+1)/100``.

    When ``sine_fit`` is supplied (from ``prefit(n_sines=1)``), reuse its fitted
    ω/phase — or ``z_full`` when long enough — instead of the unwarped template sine.
    """
    if sine_fit:
        z_full = sine_fit.get("z_full")
        if z_full is not None and len(z_full) >= n_total:
            return np.asarray(z_full[:n_total], dtype=np.float64)
        from .readouts.log_trend_sine import sine_from_fit

        t_ext = np.arange(1, n_total + 1, dtype=np.float64) / 100.0
        return sine_from_fit(t_ext, sine_fit)
    idx = np.arange(1, n_total + 1, dtype=np.float64)
    return np.sin(2.0 * np.pi * idx / 100.0)


def _predict_forecast_parametric(
    model: WarpParametricModel,
    x_train: Tensor,
    n_future: int,
    n_draws: int,
    seed: int,
    noise_seed: int,
    path_anchor: PathAnchor,
    n_paths_ci: Optional[int] = None,
    sine_fit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n_train = model.n
    n_total = n_train + int(n_future)
    n_ci = n_paths_ci if n_paths_ci is not None else max(n_draws, 400)
    with torch.no_grad():
        p_train = model.path().cpu().numpy()
        sigma_y = float(torch.exp(model.log_sigma))
        sigma_t = float(torch.exp(model.log_sigma_t))
    if sine_fit is not None:
        x_ext = _synthetic_driver_extended(n_total, sine_fit=sine_fit)
    elif x_train is not None and len(x_train) >= n_total:
        x_ext = x_train.detach().cpu().numpy()[:n_total]
    else:
        x_ext = _synthetic_driver_extended(n_total)
    x_t = torch.tensor(x_ext, dtype=torch.float32)
    paths_p = sample_warp_paths_future_knots(
        p_train, n_total, n_train, sigma_t, model.n_knots, n_draws, seed, path_anchor=path_anchor
    )
    paths_p_ci = sample_warp_paths_future(
        p_train, n_total, n_train, sigma_t, model.n_knots, n_ci, seed + 1000, path_anchor=path_anchor
    )
    p_det = point_forecast_path(p_train, n_total, path_mode=model.path_mode, path_anchor=path_anchor)
    preds = np.stack([
        model.predict(x_t, torch.tensor(paths_p[k], dtype=torch.float32)).detach().cpu().numpy()
        for k in range(n_draws)
    ])
    preds_ci = np.stack([
        model.predict(x_t, torch.tensor(paths_p_ci[k], dtype=torch.float32)).detach().cpu().numpy()
        for k in range(n_ci)
    ])
    y_point = model.predict(x_t, torch.tensor(p_det, dtype=torch.float32)).detach().cpu().numpy()
    bands = build_forecast_bands(preds_ci, y_point, sigma_y, noise_seed=noise_seed)
    return {
        "preds": preds,
        "y_point": y_point,
        "paths_p": paths_p,
        "paths_p_ci": paths_p_ci,
        "p_det": p_det,
        "bands": bands,
        "n_train": n_train,
        "n_obs": n_train,
        "n_total": n_total,
        "n_future": n_future,
        "sigma_y": sigma_y,
        "sigma_t": sigma_t,
    }


@torch.no_grad()
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
    fit: Optional[Dict[str, Any]] = None,
    sine_fit: Optional[Dict[str, Any]] = None,
    t_idx: Optional[np.ndarray] = None,
    t_norm: Optional[np.ndarray] = None,
    z_full: Optional[np.ndarray] = None,
    n_obs: Optional[int] = None,
) -> Dict[str, Any]:
    """Sample warp-path forecast realisations and 95% terror/error/combined bands.

    **Synthetic:** pass ``model`` (+ unused ``x_train`` for API compat) and ``n_future``.
    **Bitcoin:** pass ``fit``, ``t_idx``, ``t_norm``, ``z_full``, ``n_obs`` (``n_future`` optional).

    Single-path draws use knot-level RW + piecewise-linear offset (train knot spacing).
    Confidence bands use per-index RW paths (``n_paths_ci``, default 400).
    """
    if fit is not None:
        if t_idx is None or t_norm is None or z_full is None or n_obs is None:
            raise ValueError("Bitcoin forecast requires fit, t_idx, t_norm, z_full, n_obs")
        from .readouts.log_trend_sine import forecast_future

        n_total = len(z_full)
        n_future_use = int(n_future) if n_future else n_total - int(n_obs)
        fc = forecast_future(
            fit,
            t_idx,
            t_norm,
            z_full,
            n_obs=int(n_obs),
            n_paths=n_draws,
            n_paths_ci=n_paths_ci,
            seed=seed,
        )
        fc["sigma_t"] = float(fit["warp"]["sigma_t"])
        fc["sigma_y"] = float(fit["warp"]["sigma_y"])
        fc["n_train"] = int(n_obs)
        fc["n_future"] = n_future_use
        res_fit = fit.get("residual_smooth")
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
        model, x_train, n_future, n_draws, seed, noise_seed, path_anchor,
        n_paths_ci=n_paths_ci, sine_fit=sine_fit,
    )

