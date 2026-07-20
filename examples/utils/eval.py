"""Example-only model scoring helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor

from warp_regression.core.training import compute_dual_loss
from warp_regression.core.warp import soft_warp_numpy
from warp_regression.model import WarpModel
from warp_regression.utilities.metrics import EvalReport


def evaluate_model(
    model: WarpModel,
    y: Union[Tensor, np.ndarray],
    *,
    covariates: Optional[Dict[str, Union[Tensor, np.ndarray]]] = None,
    x: Optional[Union[Tensor, np.ndarray]] = None,
    sine_fit: Optional[Dict[str, Any]] = None,
    use_discrete_warp: bool = False,
    ll_target: Optional[tuple[float, float]] = None,
) -> EvalReport:
    """Score a fitted affine/synthetic ``WarpModel`` under the dual objective."""
    if not model.is_fitted:
        raise RuntimeError("model not fitted")
    y_np = y.detach().cpu().numpy() if isinstance(y, Tensor) else np.asarray(y, dtype=np.float64)
    y_t = torch.as_tensor(y_np, dtype=torch.float32)
    cov = dict(covariates or {})
    if x is not None:
        cov.setdefault("x", x)
    for k, v in model._covariates.items():
        cov.setdefault(k, v)
    cov_t = {
        k: (v if isinstance(v, Tensor) else torch.as_tensor(v, dtype=torch.float32))
        for k, v in cov.items()
    }
    cal_t = {
        k: torch.as_tensor(v, dtype=torch.float32) for k, v in model._calendar.items()
    }
    sf = sine_fit if sine_fit is not None else model._sine_fit
    with torch.no_grad():
        y_hat_t, p_t = model.predict_torch(cov_t, cal_t, sine_fit=sf)
        _, vals = compute_dual_loss(
            y_t,
            y_hat_t,
            p_t,
            model.log_sigma,
            model.primary_path.log_sigma_t,
            model.primary_path.n_knots,
            1.0,
            path_mode=model.path_mode,
        )
        y_hat = y_hat_t.detach().cpu().numpy()
        p = p_t.detach().cpu().numpy()
    mse = float(np.mean((y_np - y_hat) ** 2))
    corr = float(np.corrcoef(y_np, y_hat)[0, 1]) if len(y_np) > 1 else 0.0
    obj_err, obj_time = float(vals.obj_err), float(vals.obj_time)
    discrete_mse = None
    lin = model.observation._linear if model.observation is not None else None
    if use_discrete_warp and lin is not None:
        name = lin.inputs[0]
        x_arr = cov.get(name)
        if x_arr is not None:
            x_np = x_arr.detach().cpu().numpy() if isinstance(x_arr, Tensor) else np.asarray(x_arr)
            x_w = soft_warp_numpy(x_np, p)
            A = float(lin.A.detach())
            C = float(lin.C.detach())
            discrete_mse = float(np.mean((y_np - (A * x_w + C)) ** 2))
    ll_distance = None
    if ll_target is not None:
        ll_distance = math.hypot(obj_err - ll_target[0], obj_time - ll_target[1])
    return EvalReport(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        corr=corr,
        obj_err=obj_err,
        obj_time=obj_time,
        err_nll=-obj_err,
        time_ll=-obj_time,
        ll_distance=ll_distance,
        discrete_mse=discrete_mse,
    )


def roll_forward_synthetic(
    y: np.ndarray,
    *,
    horizon: int = 100,
    step: int = 20,
    t0_first: int = 100,
    ci: float = 0.95,
    n_draws: int = 200,
    epochs: int = 2000,
    yaml_name: str = "synthetic.yaml",
    verbose: bool = True,
) -> list[Dict[str, Any]]:
    """Expanding-window holdout check for the synthetic example.

    For each origin ``t0``, prefit + fit on ``y[:t0]``, forecast ``horizon``
    steps, and record combined / terror coverage plus next-cycle length stats.
    """
    from warp_regression import (
        WarpModel,
        analyze_cycle_lengths,
        interval_coverage,
        prefit,
        predict_forecast_realisations_torch,
    )
    from warp_regression.covariates.sine import eval_sine_driver

    y = np.asarray(y, dtype=np.float64)
    n_all = int(y.shape[0])
    origins = list(range(t0_first, n_all - horizon + 1, step))
    rows: list[Dict[str, Any]] = []

    if verbose:
        print(
            f"roll-forward: origins={origins[0]}…{origins[-1]} step={step}  "
            f"horizon={horizon}  ci={ci:.0%}  n_windows={len(origins)}"
        )
        print(f"{'t0':>5}  {'total':>8}  {'warp':>8}  {'cycle_med':>10}  {'σ_t':>7}")
        print("-" * 48)

    for i, t0 in enumerate(origins):
        y_tr = y[:t0]
        y_te = y[t0 : t0 + horizon]
        t_tr = (np.arange(t0) + 1).astype(np.float64) / 100.0
        t_full = (np.arange(n_all) + 1).astype(np.float64) / 100.0
        pf = prefit(y_tr, t_tr, n_sines=1, t_full=t_full)
        sf = pf.sine_fit
        t_ext = (np.arange(t0 + horizon) + 1).astype(np.float64) / 100.0
        x_ext = eval_sine_driver(
            t_ext,
            sf["omega"],
            sf["phase"],
            time_scale=sf.get("time_scale", 1.0),
            t_shift=sf.get("t_shift", 0.0),
        )
        m = WarpModel.from_yaml(yaml_name, n=t0)
        m.fit(
            y_tr,
            covariates={"x": x_ext},
            sine_fit=sf,
            epochs=epochs,
            lr=0.03,
            seed=i,
            fit_lambda=0.5,
        )
        fc_w = predict_forecast_realisations_torch(
            m,
            n_future=horizon,
            n_draws=n_draws,
            n_paths_ci=n_draws,
            seed=10 + i,
            noise_seed=20 + i,
            sine_fit=sf,
            ci=ci,
        )
        bands_w = fc_w["bands"]
        cov_tot = interval_coverage(y_te, bands_w["c_q_lo"][t0:], bands_w["c_q_hi"][t0:])
        cov_warp = interval_coverage(y_te, bands_w["t_q_lo"][t0:], bands_w["t_q_hi"][t0:])
        cyc = analyze_cycle_lengths(
            model=m, sine_fit=sf, n_paths=300, seed=30 + i, path_anchor="start"
        )
        row = {
            "t0": t0,
            "coverage_total": cov_tot,
            "coverage_warp": cov_warp,
            "cycle_median": float(np.median(cyc.lengths)),
            "cycle_nominal": float(cyc.mean_cycle_length),
            "sigma_t": float(m.sigma_t.detach()),
        }
        rows.append(row)
        if verbose:
            print(
                f"{t0:5d}  {cov_tot:8.1%}  {cov_warp:8.1%}  "
                f"{row['cycle_median']:10.1f}  {row['sigma_t']:7.3f}"
            )

    if verbose and rows:
        cov_tot_all = np.array([r["coverage_total"] for r in rows])
        cov_warp_all = np.array([r["coverage_warp"] for r in rows])
        print("-" * 48)
        print(
            f"OVERALL  total {cov_tot_all.mean():.1%}  "
            f"warp {cov_warp_all.mean():.1%}  (nominal {ci:.0%})"
        )
    return rows
