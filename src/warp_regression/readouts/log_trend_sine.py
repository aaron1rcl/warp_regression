"""Bitcoin log-trend + warped sine — thin wrappers over WarpModel + utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ..constants import DEFAULT_PATH_ANCHOR, PathAnchor
from ..core.warp import soft_warp_numpy, soft_warp_sine_numpy
from ..model import WarpModel
from ..plotting import format_date_axis
from ..utilities.bitcoin import (  # noqa: F401 — re-exports for notebooks
    CYCLE_PEAK_HALF_WINDOW_DAYS,
    CYCLE_PEAK_YEARS,
    DATA_PATH,
    EPOCHS,
    FIT_LAMBDA,
    FORECAST_5YR_DAYS,
    FORECAST_DAYS,
    MLP_HIDDEN,
    MLP_LAYERS,
    N_CYCLE_PEAKS,
    N_KNOTS,
    N_PATHS,
    N_SINGLE_PATHS,
    TEST_DAYS,
    WARP_LR,
    build_forecast_extension,
    day_index,
    detect_btc_cycle_peaks,
    fetch_bitcoin_daily,
    fit_log_trend,
    fit_sine_peak_presize,
    log_price,
    log_shifted_day_index,
    normalized_time,
    omega_from_peak_times,
    price_from_log,
    sine_from_fit,
    split_bitcoin_holdout,
)
from ..utilities.bitcoin import sine_from_fit as _sine_from_fit
from ..utilities.metrics import _r2_rmse

# Notebooks: `import warp_regression.readouts.log_trend_sine as btc`
__all__ = [
    "format_date_axis",
    "fetch_bitcoin_daily",
    "log_price",
    "normalized_time",
    "day_index",
    "split_bitcoin_holdout",
    "fit_log_trend",
    "fit_sine_peak_presize",
    "detect_btc_cycle_peaks",
    "build_forecast_extension",
    "fit_warp_model",
    "predict_components",
    "WarpReadout",
    "EPOCHS",
    "N_KNOTS",
    "FIT_LAMBDA",
    "WARP_LR",
    "N_SINGLE_PATHS",
    "FORECAST_5YR_DAYS",
    "N_CYCLE_PEAKS",
]


def _bitcoin_config(
    n_knots: int,
    epochs: int,
    lr: float,
    fit_lambda: float,
    seed: int,
    mlp_hidden: int,
    mlp_layers: int,
) -> Dict[str, Any]:
    return {
        "n_knots": n_knots,
        "path_anchor": "start",
        "path_mode": "identity",
        "path_groups": {"shared": {}},
        "blocks": [{"id": "z", "path_group": "shared", "input": "z", "sample": "array"}],
        "observation": {
            "terms": [
                {"kind": "log_trend"},
                {
                    "kind": "envelope_sine",
                    "inputs": ["z"],
                    "hidden": mlp_hidden,
                    "layers": mlp_layers,
                    "act": "relu",
                },
            ]
        },
        "train": {"epochs": epochs, "lr": lr, "fit_lambda": fit_lambda, "seed": seed},
    }


class WarpReadout:
    """Deprecated name — use observation model via WarpModel."""

    pass


def warped_sine_driver(
    p: np.ndarray,
    sine_fit: Dict[str, Any],
    n_norm: int,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    return soft_warp_sine_numpy(
        p,
        n_norm,
        sine_fit["omega"],
        sine_fit["phase"],
        time_scale=sine_fit.get("time_scale", 1.0),
        t_shift=sine_fit.get("t_shift", 0.0),
        path_anchor=path_anchor,
    )


def predict_components(
    readout: Any,
    t_idx: np.ndarray,
    t_norm: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    sine_fit: Optional[Dict[str, Any]] = None,
    n_norm: Optional[int] = None,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sine_fit is not None:
        n_ref = n_norm if n_norm is not None else len(t_norm)
        z_w = warped_sine_driver(p, sine_fit, n_ref, path_anchor=path_anchor)
    else:
        z_w = soft_warp_numpy(z, p, path_anchor=path_anchor)
    with torch.no_grad():
        ti = torch.tensor(t_idx, dtype=torch.float32)
        tn = torch.tensor(t_norm, dtype=torch.float32)
        zw = torch.tensor(z_w, dtype=torch.float32)
        if hasattr(readout, "trend_forward"):
            trend = readout.trend_forward(ti).detach().numpy()
            cyclic = readout.cyclic_forward(tn, zw).detach().numpy()
        else:
            y = readout(ti, tn, zw).detach().numpy()
            return y * 0.0, y, y
    return trend, cyclic, trend + cyclic


def fit_warp_model(
    y: np.ndarray,
    t_idx: np.ndarray,
    t: np.ndarray,
    sine_fit: Dict[str, Any],
    n_knots: int = N_KNOTS,
    epochs: int = EPOCHS,
    lr: float = WARP_LR,
    fit_lambda: float = FIT_LAMBDA,
    mlp_hidden: int = MLP_HIDDEN,
    mlp_layers: int = MLP_LAYERS,
    seed: int = 0,
) -> Dict[str, Any]:
    n = len(y)
    z = np.asarray(sine_fit["z"], dtype=np.float64)[:n]
    n_calendar = int(round(0.5 / float(t[0]))) if len(t) else n
    model = WarpModel.from_dict(
        _bitcoin_config(n_knots, epochs, lr, fit_lambda, seed, mlp_hidden, mlp_layers),
        n=n,
    )
    res = model.fit(
        y,
        covariates={"z": z},
        calendar={"t_idx": np.asarray(t_idx, dtype=np.float64), "t_norm": np.asarray(t, dtype=np.float64)},
        sine_fit=sine_fit,
    )
    readout = res.fit["readout"]
    p = res.fit["warp"]["p"]
    trend, cyclic, y_hat = predict_components(readout, t_idx, t, z, p, sine_fit=sine_fit, n_norm=n_calendar)
    with torch.no_grad():
        f_t = readout.f_net(torch.tensor(t, dtype=torch.float32)).numpy() if readout.f_net else np.zeros(n)
        atten_t = readout.atten(torch.tensor(t, dtype=torch.float32)).numpy()
    return {
        "sine_fit": sine_fit,
        "y_hat": y_hat,
        "trend": trend,
        "cyclic": cyclic,
        "f_t": f_t,
        "atten_t": atten_t,
        "z_warped": soft_warp_numpy(z, p, path_anchor=DEFAULT_PATH_ANCHOR),
        "r2": res.r2,
        "rmse": res.rmse,
        "B": float(readout.B.detach()) if readout.B is not None else 0.0,
        "C": float(readout.C.detach()) if readout.C is not None else 0.0,
        "d": float(readout.d.detach()),
        "t_z": float(readout.z_shift.detach()),
        "readout": readout,
        "mlp_hidden": mlp_hidden,
        "mlp_layers": mlp_layers,
        "warp": res.fit["warp"],
        "n_knots": n_knots,
        "epochs": epochs,
        "n_calendar": n_calendar,
        "_y_train": np.asarray(y, dtype=np.float64),
        "kind": "log_trend_sine",
        "_model": model,
    }


def forecast_future(
    fit: Dict[str, Any],
    t_idx_full: np.ndarray,
    t_norm_full: np.ndarray,
    z_full: np.ndarray,
    n_obs: int,
    n_paths: int = N_PATHS,
    n_paths_ci: Optional[int] = None,
    seed: int = 1,
) -> Dict[str, Any]:
    from ..forecast import predict_forecast_realisations_torch

    return predict_forecast_realisations_torch(
        fit=fit,
        t_idx=t_idx_full,
        t_norm=t_norm_full,
        z_full=z_full,
        n_obs=n_obs,
        n_future=len(z_full) - n_obs,
        n_draws=n_paths,
        n_paths_ci=n_paths_ci,
        seed=seed,
        noise_seed=seed + 1,
    )
