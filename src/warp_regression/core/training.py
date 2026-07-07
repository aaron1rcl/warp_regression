"""Warp regression module."""
from __future__ import annotations


import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .path import path_from_B_torch, stored_path_offset_torch

def reverse_softplus_torch(x: Tensor) -> Tensor:
    out = x.clone()
    mask = x > -10
    if mask.any():
        out[mask] = -torch.log1p(torch.exp(-x[mask]))
    return out


def gaussian_error_nll(
    resid: Tensor,
    sigma: Tensor,
    sample_weights: Optional[Tensor] = None,
) -> Tensor:
    var = sigma**2 + 1e-8
    term = 0.5 * (resid**2 / var + torch.log(2 * math.pi * var))
    if sample_weights is None:
        return torch.sum(term)
    return torch.sum(sample_weights * term)


def terror_path_for_likelihood_torch(p: Tensor, path_mode: str = "identity") -> Tensor:
    """RW likelihood on warp offsets (identity), not coordinate ramp."""
    return stored_path_offset_torch(p, path_mode=path_mode)


def terror_knot_values(n: int, n_knots: int) -> Tuple[np.ndarray, float]:
    """Knot grid and RW segment width matching WarpPureNumpy."""
    k_values = np.arange(0, n_knots, dtype=np.float64) * (float(n) / float(n_knots))
    rw_width = float(n / n_knots)
    return k_values, rw_width


def conditional_posterior_torch(d: Tensor, sd: Tensor, n_steps: float) -> Tuple[Tensor, Tensor]:
    """WarpPureNumpy conditional_posterior (u argument ignored; set from d)."""
    u = d / n_steps
    post_sd = torch.sqrt((sd**2) * (n_steps - 1.0) / n_steps)
    return u, post_sd


def px_z_torch(u1: Tensor, u2: Tensor, sd1: Tensor, sd2: Tensor) -> Tensor:
    """WarpPureNumpy px_z."""
    alpha = torch.sqrt(1 / (2 * sd2**2) + 1 / (2 * sd1**2))
    alpha_beta = u2 / (2 * sd2**2) + u1 / (2 * sd1**2)
    gamma = (u2**2) / (2 * sd2**2) + (u1**2) / (2 * sd1**2)
    beta = alpha_beta / alpha
    p1 = 1 / (2 * sd1 * sd2 * alpha * math.sqrt(math.pi))
    p2 = torch.exp(-gamma + beta**2)
    return p1 * p2


def expected_likelihood_torch(d: Tensor, sd: Tensor, n_steps: float) -> Tensor:
    """WarpPureNumpy expected_likelihood."""
    post_u, post_sd = conditional_posterior_torch(d, sd, n_steps)
    el = torch.log(px_z_torch(torch.zeros((), device=d.device, dtype=d.dtype), post_u, sd, post_sd))
    return reverse_softplus_torch(el * n_steps)


def terror_likelihood_torch(p: Tensor, sigma_t: Tensor, n_knots: int) -> Tensor:
    """WarpPureNumpy terror_likelihood on path values p (offsets for identity mode)."""
    n = int(p.shape[0])
    k_values, rw_width = terror_knot_values(n, n_knots)
    k_values = np.append(k_values, n - 1.0)
    total = torch.zeros((), device=p.device, dtype=p.dtype)
    for i in range(1, len(k_values)):
        x1, x2 = int(k_values[i - 1]), int(k_values[i])
        d = p[x2] - p[x1]
        total = total + expected_likelihood_torch(d, sigma_t, rw_width)
    return total


@dataclass
class DualLossValues:
    obj_err: float
    obj_time: float
    sigma_y: float
    sigma_t: float
    lam: float


def compute_dual_loss(
    y: Tensor,
    y_hat: Tensor,
    p: Tensor,
    log_sigma: Tensor,
    log_sigma_t: Tensor,
    n_knots: int,
    lam: float,
    path_mode: str = "identity",
    sample_weights: Optional[Tensor] = None,
) -> Tuple[Tensor, DualLossValues]:
    sigma_y = torch.exp(log_sigma)
    sigma_t = torch.exp(log_sigma_t)
    obj_err = gaussian_error_nll(y - y_hat, sigma_y, sample_weights)
    p_terror = terror_path_for_likelihood_torch(p, path_mode=path_mode)
    terror_ll = terror_likelihood_torch(p_terror, sigma_t, n_knots)
    loss = lam * obj_err - (1.0 - lam) * terror_ll
    vals = DualLossValues(
        obj_err=float(obj_err.detach()),
        obj_time=float((-terror_ll).detach()),
        sigma_y=float(sigma_y.detach()),
        sigma_t=float(sigma_t.detach()),
        lam=float(lam),
    )
    return loss, vals


def _lambda_schedule(epoch: int, epochs: int, fit_lambda: Optional[float]) -> float:
    if fit_lambda is not None:
        return float(fit_lambda)
    frac = epoch / max(epochs - 1, 1)
    return float(0.25 + 0.35 * frac)


@dataclass
class WarpTrainConfig:
    n: int
    n_knots: int
    epochs: int = 1000
    lr: float = 0.03
    path_mode: str = "identity"
    fit_lambda: Optional[float] = None
    seed: int = 0


def train_dual_warp(
    config: WarpTrainConfig,
    y: Tensor,
    forward_fn: Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]],
    extra_params: Optional[Iterable[Tensor]] = None,
    B_init: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
) -> Dict[str, Any]:
    torch.manual_seed(config.seed)
    B = (B_init.clone() if B_init is not None else torch.zeros(config.n_knots)).detach()
    B = nn.Parameter(B.float())
    log_sigma = nn.Parameter(torch.tensor(0.0))
    log_sigma_t = nn.Parameter(torch.tensor(-0.5))
    params: List[Tensor] = [B, log_sigma, log_sigma_t]
    if extra_params:
        params.extend(list(extra_params))
    opt = torch.optim.Adam(params, lr=config.lr)
    history: List[DualLossValues] = []
    best: Dict[str, Any] = {}
    best_loss = float("inf")

    for ep in range(config.epochs):
        opt.zero_grad()
        lam = _lambda_schedule(ep, config.epochs, config.fit_lambda)
        y_hat, p = forward_fn(B, log_sigma, log_sigma_t)
        loss, vals = compute_dual_loss(
            y,
            y_hat,
            p,
            log_sigma,
            log_sigma_t,
            config.n_knots,
            lam,
            path_mode=config.path_mode,
            sample_weights=sample_weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()
        history.append(vals)
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            with torch.no_grad():
                y_hat_b, p_b = forward_fn(B, log_sigma, log_sigma_t)
            best = {
                "B": B.detach().cpu().clone(),
                "p": p_b.detach().cpu().numpy(),
                "y_hat": y_hat_b.detach().cpu().numpy(),
                **vals.__dict__,
            }

    return {
        "B": best["B"],
        "log_sigma": log_sigma.detach().cpu().clone(),
        "log_sigma_t": log_sigma_t.detach().cpu().clone(),
        "history": history,
        "warp": best,
        "config": config,
    }

