"""Example-only WarpModel fit wrappers (Lynx / dual-sine notebooks)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from warp_regression.core.warp import soft_warp_numpy
from warp_regression.covariates.sine import build_dual_sines_from_fit, fit_dual_sine_log
from warp_regression.model import WarpModel
from warp_regression.utilities.metrics import _r2_rmse


def _linear_config(n_knots: int, epochs: int, lr: float, fit_lambda: Optional[float], seed: int) -> Dict[str, Any]:
    return {
        "n_knots": n_knots,
        "path_anchor": "start",
        "path_mode": "identity",
        "path_groups": {"shared": {}},
        "blocks": [
            {"id": "z1", "path_group": "shared", "input": "z1", "covariate_kind": "array"},
            {"id": "z2", "path_group": "shared", "input": "z2", "covariate_kind": "array"},
        ],
        "observation": {"terms": [{"kind": "linear", "inputs": ["z1", "z2"]}]},
        "train": {"epochs": epochs, "lr": lr, "fit_lambda": fit_lambda, "seed": seed},
    }


def _mlp_config(
    n_knots: int, epochs: int, lr: float, fit_lambda: Optional[float], seed: int, hidden: int, layers: int
) -> Dict[str, Any]:
    return {
        "n_knots": n_knots,
        "path_anchor": "start",
        "path_mode": "identity",
        "path_groups": {"shared": {}},
        "blocks": [
            {"id": "z1", "path_group": "shared", "input": "z1", "covariate_kind": "array"},
            {"id": "z2", "path_group": "shared", "input": "z2", "covariate_kind": "array"},
        ],
        "observation": {
            "terms": [
                {
                    "kind": "mlp_on_warped",
                    "inputs": ["z1", "z2"],
                    "hidden": hidden,
                    "layers": layers,
                    "act": "gelu",
                }
            ]
        },
        "train": {"epochs": epochs, "lr": lr, "fit_lambda": fit_lambda, "seed": seed},
    }


def fit_dual_sine_shared_warp(
    y_log: np.ndarray,
    t: np.ndarray,
    years: Optional[np.ndarray] = None,
    n_knots: int = 14,
    epochs: int = 1500,
    lr: float = 0.03,
    fit_lambda: Optional[float] = 0.5,
    seed: int = 0,
    sine_fit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if sine_fit is None:
        sine_fit = fit_dual_sine_log(y_log, t, years=years)
    z1, z2 = build_dual_sines_from_fit(t, sine_fit)
    model = WarpModel.from_dict(
        _linear_config(n_knots, epochs, lr, fit_lambda, seed), n=len(y_log)
    )
    res = model.fit(y_log, covariates={"z1": z1, "z2": z2}, sine_fit=sine_fit)
    r2_u, _ = _r2_rmse(y_log, sine_fit["y_hat_log"])
    p = res.fit["warp"]["p"]
    return {
        **sine_fit,
        "z1": z1,
        "z2": z2,
        "z1_warped": soft_warp_numpy(z1, p),
        "z2_warped": soft_warp_numpy(z2, p),
        "y_hat_log": res.y_hat,
        "y_hat_log_unwarped": sine_fit["y_hat_log"],
        "r2_log": res.r2,
        "r2_log_unwarped": r2_u,
        "rmse_log": res.rmse,
        "warp": res.fit["warp"],
        "n_knots": n_knots,
        "observation": res.fit["observation"],
        "kind": res.fit.get("kind"),
        "_y_train": np.asarray(y_log, dtype=np.float64),
        "_model": model,
    }


def fit_dual_sine_shared_warp_nonlinear(
    y_log: np.ndarray,
    t: np.ndarray,
    years: Optional[np.ndarray] = None,
    n_knots: int = 14,
    epochs: int = 2500,
    hidden: int = 32,
    n_hidden_layers: int = 2,
    lr: float = 0.03,
    fit_lambda: Optional[float] = 0.5,
    seed: int = 0,
    sine_fit: Optional[Dict[str, Any]] = None,
    B_init: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """MLP dual fit. Pass ``B_init`` from a prior linear fit (notebook-style).

    If ``B_init`` is omitted, fits a short linear model first and copies ``B``
    (compat for older call sites). Prefer explicit two-model flow in notebooks.
    """
    if sine_fit is None:
        sine_fit = fit_dual_sine_log(y_log, t, years=years)
    z1, z2 = build_dual_sines_from_fit(t, sine_fit)
    lin_y_hat = None
    lin_r2 = None
    if B_init is None:
        lin = fit_dual_sine_shared_warp(
            y_log,
            t,
            years=years,
            n_knots=n_knots,
            epochs=max(400, epochs // 4),
            lr=lr,
            fit_lambda=fit_lambda,
            seed=seed,
            sine_fit=sine_fit,
        )
        B_init = lin["warp"]["B"]
        lin_y_hat = lin["y_hat_log"]
        lin_r2 = lin["r2_log"]
    model = WarpModel.from_dict(
        _mlp_config(n_knots, epochs, lr, fit_lambda, seed, hidden, n_hidden_layers),
        n=len(y_log),
    )
    res = model.fit(
        y_log,
        covariates={"z1": z1, "z2": z2},
        sine_fit=sine_fit,
        B_init=torch.as_tensor(B_init, dtype=torch.float32),
    )
    out = {
        **sine_fit,
        "z1": z1,
        "z2": z2,
        "y_hat_log": res.y_hat,
        "r2_log": res.r2,
        "rmse_log": res.rmse,
        "warp": res.fit["warp"],
        "n_knots": n_knots,
        "observation": res.fit["observation"],
        "kind": res.fit.get("kind"),
        "_y_train": np.asarray(y_log, dtype=np.float64),
        "_model": model,
    }
    if lin_y_hat is not None:
        out["y_hat_log_linear_warp"] = lin_y_hat
        out["r2_log_linear_warp"] = lin_r2
    return out
