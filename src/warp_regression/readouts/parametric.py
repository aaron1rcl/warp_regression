from __future__ import annotations


import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..constants import DEFAULT_PATH_ANCHOR, NOTEBOOK_LL_TARGET, PathAnchor
from ..core.path import path_from_B_torch
from ..core.training import WarpTrainConfig, compute_dual_loss, train_dual_warp
from ..core.warp import soft_warp_numpy, soft_warp_torch
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
        path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
    ) -> None:
        super().__init__()
        self.n, self.n_knots, self.sr = int(n), int(n_knots), int(sr)
        self.path_mode = path_mode
        self.path_anchor: PathAnchor = path_anchor
        self.B = nn.Parameter(torch.zeros(n_knots))
        self.log_A = nn.Parameter(torch.tensor(math.log(A_init), dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor(C_init, dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_t = nn.Parameter(torch.tensor(-0.5))

    def path(self, B: Optional[Tensor] = None, path_anchor: Optional[PathAnchor] = None) -> Tensor:
        b = self.B if B is None else B
        anchor = self.path_anchor if path_anchor is None else path_anchor
        return path_from_B_torch(b, self.n, self.n_knots, self.path_mode, path_anchor=anchor)

    def predict(
        self,
        x: Tensor,
        p: Optional[Tensor] = None,
        path_anchor: Optional[PathAnchor] = None,
    ) -> Tensor:
        anchor = self.path_anchor if path_anchor is None else path_anchor
        if p is None:
            p = self.path(path_anchor=anchor)
        z = soft_warp_torch(x, p, path_mode=self.path_mode, path_anchor=anchor)
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
            p = path_from_B_torch(B, self.n, self.n_knots, self.path_mode, path_anchor=self.path_anchor)
            z = soft_warp_torch(x, p, path_mode=self.path_mode, path_anchor=self.path_anchor)
            Z = torch.stack([z, torch.ones_like(z)], dim=1)
            coef = torch.linalg.lstsq(Z, y.unsqueeze(1)).solution.squeeze(1)
            return coef[0] * z + coef[1], p

        result = train_dual_warp(cfg, y, forward_fn, extra_params=[], B_init=self.B.data)
        with torch.no_grad():
            self.B.copy_(result["B"])
            self.log_sigma.copy_(result["log_sigma"])
            self.log_sigma_t.copy_(result["log_sigma_t"])
            p = self.path()
            z = soft_warp_torch(x, p, path_mode=self.path_mode, path_anchor=self.path_anchor)
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
        x_w = soft_warp_numpy(x.cpu().numpy(), p, reverse_path=False, path_mode=model.path_mode)
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
    """Sample path uncertainty over the first ``horizon`` time indices.

    Paths are the MAP fit plus a terror-scale knot RW deviation (start- or
    end-pinned). Stored path is applied as-is (no reverse).
    """
    from ..core.path import knot_positions

    n = model.n
    h = min(int(horizon), n)
    anchor = getattr(model, "path_anchor", DEFAULT_PATH_ANCHOR)
    if model.path_mode != "identity":
        raise ValueError("predict_realisations_torch currently supports path_mode='identity'")

    with torch.no_grad():
        p_fit = model.path().cpu().numpy()
        sigma_t = float(torch.exp(model.log_sigma_t).detach())

    rng = np.random.default_rng(seed)
    _, rw_interval = knot_positions(n, model.n_knots)
    sigma_knot = float(sigma_t * rw_interval)
    knot_x = np.linspace(0.0, float(n - 1), model.n_knots)
    idx = np.arange(n, dtype=np.float64)
    off_knots_fit = np.interp(knot_x, idx, p_fit - idx)

    x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    paths = np.zeros((n_draws, h))
    for k in range(n_draws):
        dev = np.zeros(model.n_knots, dtype=np.float64)
        if anchor == "end":
            for j in range(1, model.n_knots):
                dev[model.n_knots - 1 - j] = dev[model.n_knots - j] + rng.normal(0.0, sigma_knot)
        else:
            for j in range(1, model.n_knots):
                dev[j] = dev[j - 1] + rng.normal(0.0, sigma_knot)
        off_k = off_knots_fit + dev
        off_k[-1 if anchor == "end" else 0] = 0.0
        p = idx + np.interp(idx, knot_x, off_k)
        p[-1 if anchor == "end" else 0] = float(n - 1) if anchor == "end" else 0.0
        with torch.no_grad():
            y_full = model.predict(
                x_t, torch.tensor(p, dtype=torch.float32)
            ).detach().cpu().numpy()
        paths[k] = y_full[:h]
    return paths

