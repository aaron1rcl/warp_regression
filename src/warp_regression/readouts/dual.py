from __future__ import annotations


from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..constants import DEFAULT_PATH_ANCHOR
from ..core.path import path_from_B_torch
from ..core.training import WarpTrainConfig, train_dual_warp
from ..core.warp import soft_warp_numpy, soft_warp_torch
from ..drivers.sine import build_dual_sines_from_fit, fit_dual_sine_log
from ..utilities.metrics import _r2_rmse

class ScalarReadout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, z1_w: Tensor, z2_w: Tensor) -> Tensor:
        z1_w = z1_w.to(dtype=self.linear.weight.dtype)
        z2_w = z2_w.to(dtype=self.linear.weight.dtype)
        return self.linear(torch.stack([z1_w, z2_w], dim=-1)).squeeze(-1)

    def fit_ls(self, z1_w: np.ndarray, z2_w: np.ndarray, y: np.ndarray) -> None:
        Z = np.column_stack([z1_w, z2_w, np.ones(len(y))])
        coef, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(coef[:2], dtype=torch.float32).view(1, 2))
            self.linear.bias.copy_(torch.tensor([coef[2]], dtype=torch.float32))


def fit_dual_sine_shared_warp(
    y_log: np.ndarray,
    t: np.ndarray,
    years: Optional[np.ndarray] = None,
    n_knots: int = 14,
    epochs: int = 1500,
    lr: float = 0.03,
    fit_lambda: Optional[float] = 1.0,
    seed: int = 0,
    sine_fit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n = len(y_log)
    if sine_fit is None:
        sine_fit = fit_dual_sine_log(y_log, t, years=years)
    z1, z2 = build_dual_sines_from_fit(t, sine_fit)
    z1_t, z2_t = torch.tensor(z1, dtype=torch.float32), torch.tensor(z2, dtype=torch.float32)
    y_t = torch.tensor(y_log, dtype=torch.float32)
    readout = ScalarReadout()

    def forward_fn(B: Tensor, _lsy: Tensor, _lst: Tensor) -> Tuple[Tensor, Tensor]:
        p = path_from_B_torch(B, n, n_knots, "identity")
        z1w, z2w = soft_warp_torch(z1_t, p), soft_warp_torch(z2_t, p)
        readout.fit_ls(z1w.detach().cpu().numpy(), z2w.detach().cpu().numpy(), y_log)
        return readout(z1w, z2w), p

    result = train_dual_warp(
        WarpTrainConfig(n=n, n_knots=n_knots, epochs=epochs, lr=lr, path_mode="identity", fit_lambda=fit_lambda, seed=seed),
        y_t,
        forward_fn,
        extra_params=list(readout.parameters()),
    )
    p = result["warp"]["p"]
    z1w, z2w = soft_warp_numpy(z1, p), soft_warp_numpy(z2, p)
    y_hat = readout(torch.tensor(z1w, dtype=torch.float32), torch.tensor(z2w, dtype=torch.float32)).detach().numpy()
    r2, rmse = _r2_rmse(y_log, y_hat)
    r2_u, _ = _r2_rmse(y_log, sine_fit["y_hat_log"])
    w = result["warp"]
    return {
        **sine_fit,
        "z1_warped": z1w,
        "z2_warped": z2w,
        "y_hat_log": y_hat,
        "y_hat_log_unwarped": sine_fit["y_hat_log"],
        "r2_log": r2,
        "r2_log_unwarped": r2_u,
        "rmse_log": rmse,
        "warp": {
            "p": p,
            "B": result["B"].numpy(),
            "obj_err": w["obj_err"],
            "obj_time": w["obj_time"],
            "sigma_y": w["sigma_y"],
            "sigma_t": w["sigma_t"],
            "max_abs_offset": float(np.max(np.abs(p - np.arange(n)))),
        },
        "readout": readout,
    }


class _DriverMLP(nn.Module):
    def __init__(self, hidden: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = 1
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z.unsqueeze(-1)).squeeze(-1)


def fit_dual_sine_shared_warp_nonlinear(
    y_log: np.ndarray,
    t: np.ndarray,
    years: Optional[np.ndarray] = None,
    n_knots: int = 14,
    epochs: int = 2500,
    hidden: int = 32,
    n_hidden_layers: int = 2,
    lr: float = 0.03,
    fit_lambda: Optional[float] = 1.0,
    seed: int = 0,
    sine_fit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    lin = fit_dual_sine_shared_warp(
        y_log, t, years=years, n_knots=n_knots, epochs=max(400, epochs // 4),
        lr=lr, fit_lambda=fit_lambda, seed=seed, sine_fit=sine_fit,
    )
    n = len(y_log)
    z1, z2 = lin["z1"], lin["z2"]
    z1_t, z2_t = torch.tensor(z1, dtype=torch.float32), torch.tensor(z2, dtype=torch.float32)
    y_t = torch.tensor(y_log, dtype=torch.float32)
    f_net = _DriverMLP(hidden, n_hidden_layers)
    g_net = _DriverMLP(hidden, n_hidden_layers)
    B_init = torch.tensor(lin["warp"]["B"], dtype=torch.float32)

    def forward_fn(B: Tensor, _lsy: Tensor, _lst: Tensor) -> Tuple[Tensor, Tensor]:
        p = path_from_B_torch(B, n, n_knots, "identity")
        z1w, z2w = soft_warp_torch(z1_t, p), soft_warp_torch(z2_t, p)
        return f_net(z1w) + g_net(z2w), p

    result = train_dual_warp(
        WarpTrainConfig(n=n, n_knots=n_knots, epochs=epochs, lr=lr, path_mode="identity", fit_lambda=fit_lambda, seed=seed),
        y_t,
        forward_fn,
        extra_params=list(f_net.parameters()) + list(g_net.parameters()),
        B_init=B_init,
    )
    p = result["warp"]["p"]
    z1w, z2w = soft_warp_numpy(z1, p), soft_warp_numpy(z2, p)
    z1w_t = torch.tensor(z1w, dtype=torch.float32)
    z2w_t = torch.tensor(z2w, dtype=torch.float32)
    y_hat = (f_net(z1w_t) + g_net(z2w_t)).detach().numpy()
    r2, rmse = _r2_rmse(y_log, y_hat)
    return {
        **{k: v for k, v in lin.items() if k not in ("y_hat_log", "r2_log", "rmse_log")},
        "component1_log": f_net(z1w_t).detach().numpy(),
        "component2_log": g_net(z2w_t).detach().numpy(),
        "y_hat_log": y_hat,
        "y_hat_log_linear_warp": lin["y_hat_log"],
        "r2_log": r2,
        "r2_log_linear_warp": lin["r2_log"],
        "rmse_log": rmse,
        "readout": {"f": f_net, "g": g_net},
        "warp": {
            **lin["warp"],
            "p": p,
            "B": result["B"].numpy(),
            "obj_err": result["warp"]["obj_err"],
            "obj_time": result["warp"]["obj_time"],
            "sigma_y": result["warp"]["sigma_y"],
            "sigma_t": result["warp"]["sigma_t"],
            "max_abs_offset": float(np.max(np.abs(p - np.arange(n)))),
        },
    }


