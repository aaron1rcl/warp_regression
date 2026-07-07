from __future__ import annotations


import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..constants import NOTEBOOK_LL_TARGET
from ..core.path import path_from_B_torch
from ..core.training import WarpTrainConfig, compute_dual_loss, train_dual_warp
from ..core.warp import path_for_warp_numpy, soft_warp_numpy, soft_warp_torch
from ..utilities.metrics import EvalReport

class WarpParametricModel(nn.Module):
    def __init__(
        self,
        n: int,
        n_knots: int = 8,
        sr: int = 10,
        A_init: float = 2.7,
        C_init: float = 0.5,
        path_mode: str = "identity",
    ) -> None:
        super().__init__()
        self.n, self.n_knots, self.sr = int(n), int(n_knots), int(sr)
        self.path_mode = path_mode
        self.B = nn.Parameter(torch.zeros(n_knots))
        self.log_A = nn.Parameter(torch.tensor(math.log(A_init), dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor(C_init, dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_t = nn.Parameter(torch.tensor(-0.5))

    def path(self, B: Optional[Tensor] = None) -> Tensor:
        b = self.B if B is None else B
        return path_from_B_torch(b, self.n, self.n_knots, self.path_mode)

    def predict(self, x: Tensor, p: Optional[Tensor] = None) -> Tensor:
        if p is None:
            p = self.path()
        z = soft_warp_torch(x, p, path_mode=self.path_mode)
        return torch.exp(self.log_A) * z + self.C

    def losses(self, x: Tensor, y: Tensor, lam: float = 1.0) -> Dict[str, Tensor]:
        p = self.path()
        y_hat = self.predict(x, p)
        loss, vals = compute_dual_loss(
            y, y_hat, p, self.log_sigma, self.log_sigma_t, self.n_knots, lam, path_mode=self.path_mode
        )
        return {"loss": loss, "y_hat": y_hat, "p": p, "obj_err": torch.tensor(vals.obj_err), "obj_time": torch.tensor(vals.obj_time)}

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        epochs: int = 1000,
        lr: float = 0.03,
        seed: int = 0,
        fit_lambda: Optional[float] = 0.5,
    ) -> Dict[str, Any]:
        cfg = WarpTrainConfig(
            n=self.n,
            n_knots=self.n_knots,
            epochs=epochs,
            lr=lr,
            path_mode=self.path_mode,
            fit_lambda=fit_lambda,
            seed=seed,
        )

        def forward_fn(B: Tensor, _lsy: Tensor, _lst: Tensor) -> Tuple[Tensor, Tensor]:
            p = path_from_B_torch(B, self.n, self.n_knots, self.path_mode)
            z = soft_warp_torch(x, p, path_mode=self.path_mode)
            Z = torch.stack([z, torch.ones_like(z)], dim=1)
            coef = torch.linalg.lstsq(Z, y.unsqueeze(1)).solution.squeeze(1)
            return coef[0] * z + coef[1], p

        result = train_dual_warp(cfg, y, forward_fn, extra_params=[], B_init=self.B.data)
        with torch.no_grad():
            self.B.copy_(result["B"])
            self.log_sigma.copy_(result["log_sigma"])
            self.log_sigma_t.copy_(result["log_sigma_t"])
            p = self.path()
            z = soft_warp_torch(x, p, path_mode=self.path_mode)
            Z = torch.stack([z, torch.ones_like(z)], dim=1)
            coef = torch.linalg.lstsq(Z, y.unsqueeze(1)).solution.squeeze(1)
            self.log_A.copy_(torch.log(coef[0].clamp(min=1e-6)))
            self.C.copy_(coef[1])
        return result


def evaluate_model(
    model: WarpParametricModel,
    x: Tensor,
    y: Tensor,
    use_discrete_warp: bool = False,
    n: Optional[int] = None,
    sr: Optional[int] = None,
) -> EvalReport:
    n = n or model.n
    sr = sr or model.sr
    with torch.no_grad():
        d = model.losses(x, y, lam=1.0)
        y_hat, p = d["y_hat"].cpu().numpy(), d["p"].cpu().numpy()
        y_np = y.cpu().numpy()
    mse = float(np.mean((y_np - y_hat) ** 2))
    corr = float(np.corrcoef(y_np, y_hat)[0, 1]) if len(y_np) > 1 else 0.0
    obj_err, obj_time = float(d["obj_err"]), float(d["obj_time"])
    discrete_mse = None
    if use_discrete_warp:
        x_w = soft_warp_numpy(x.cpu().numpy(), p, reverse_path=True, path_mode=model.path_mode)
        y_disc = float(torch.exp(model.log_A).detach()) * x_w + float(model.C.detach())
        discrete_mse = float(np.mean((y_np - y_disc) ** 2))
    return EvalReport(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        corr=corr,
        obj_err=obj_err,
        obj_time=obj_time,
        err_nll=-obj_err,
        time_ll=-obj_time,
        ll_distance=math.hypot(obj_err - NOTEBOOK_LL_TARGET[0], obj_time - NOTEBOOK_LL_TARGET[1]),
        discrete_mse=discrete_mse,
    )


def predict_realisations_torch(
    model: WarpParametricModel,
    x: Tensor,
    n_draws: int = 400,
    horizon: int = 60,
    seed: int = 0,
) -> np.ndarray:
    """Sample path uncertainty over the first `horizon` time indices.

    RW noise is applied in applied-time offset space (indices 0..h-1), then
    mapped back to the stored path (reverse application convention).
    """
    rng = np.random.default_rng(seed)
    n = model.n
    h = min(int(horizon), n)
    from ..forecast import per_index_rw_sigma

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

