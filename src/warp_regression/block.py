"""Portable warp path + single-series WarpRegression block."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .constants import DEFAULT_PATH_ANCHOR, PathAnchor
from .core.path import path_from_B_torch
from .core.training import terror_likelihood_torch, terror_path_for_likelihood_torch
from .core.warp import (
    soft_warp_numpy,
    soft_warp_prefit_sine_numpy,
    soft_warp_sine_numpy,
    soft_warp_torch,
    soft_warp_prefit_sine_torch,
    soft_warp_sine_torch,
)
SampleKind = Literal["array", "analytic_sine", "prefit_sine"]


class WarpPath(nn.Module):
    """Shared knot parameters B and terror scale σ_t for one or more warp blocks."""

    def __init__(
        self,
        n: int,
        n_knots: int = 8,
        *,
        path_mode: str = "identity",
        path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
        B_init: Optional[Union[Tensor, np.ndarray]] = None,
        log_sigma_t_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.n_knots = int(n_knots)
        self.path_mode = path_mode
        self.path_anchor: PathAnchor = path_anchor
        if B_init is None:
            B0 = torch.zeros(n_knots)
        else:
            B0 = torch.as_tensor(B_init, dtype=torch.float32).reshape(-1)
            if B0.numel() != n_knots:
                raise ValueError(f"B_init length {B0.numel()} != n_knots={n_knots}")
        self.B = nn.Parameter(B0.float())
        self.log_sigma_t = nn.Parameter(torch.tensor(float(log_sigma_t_init), dtype=torch.float32))

    @property
    def sigma_t(self) -> Tensor:
        return torch.exp(self.log_sigma_t)

    def path(self, B: Optional[Tensor] = None, n: Optional[int] = None) -> Tensor:
        b = self.B if B is None else B
        return path_from_B_torch(
            b,
            int(n if n is not None else self.n),
            self.n_knots,
            self.path_mode,
            path_anchor=self.path_anchor,
        )

    def terror_nll(self, p: Optional[Tensor] = None) -> Tensor:
        """Negative terror log-likelihood (−terror_ll) for dual loss."""
        if p is None:
            p = self.path()
        p_terror = terror_path_for_likelihood_torch(p, path_mode=self.path_mode)
        return -terror_likelihood_torch(p_terror, self.sigma_t, self.n_knots)

    def sample_paths(
        self,
        n_total: int,
        n_train: Optional[int] = None,
        *,
        n_paths: int = 500,
        seed: int = 0,
        knot_level: bool = True,
    ) -> np.ndarray:
        from .forecast import sample_warp_paths_future, sample_warp_paths_future_knots

        n_tr = int(n_train if n_train is not None else self.n)
        with torch.no_grad():
            p_train = self.path(n=n_tr).detach().cpu().numpy()
            sigma_t = float(self.sigma_t.detach())
        fn = sample_warp_paths_future_knots if knot_level else sample_warp_paths_future
        return fn(
            p_train,
            int(n_total),
            n_tr,
            sigma_t,
            self.n_knots,
            n_paths=n_paths,
            seed=seed,
            path_anchor=self.path_anchor,
        )


class WarpRegression(nn.Module):
    """Single-series warp block: soft-warps one covariate under a WarpPath.

    Portable to other PyTorch problems. Owns path-level forecast helpers via
    the attached ``WarpPath`` (which may be shared across blocks).
    """

    def __init__(
        self,
        path: WarpPath,
        *,
        sample: SampleKind = "array",
        name: str = "x",
    ) -> None:
        super().__init__()
        self.path_module = path
        self.sample: SampleKind = sample
        self.name = name

    @property
    def path_anchor(self) -> PathAnchor:
        return self.path_module.path_anchor

    def path_tensor(self, B: Optional[Tensor] = None, n: Optional[int] = None) -> Tensor:
        return self.path_module.path(B=B, n=n)

    def terror_nll(self, p: Optional[Tensor] = None) -> Tensor:
        return self.path_module.terror_nll(p)

    def warp(
        self,
        x: Tensor,
        p: Optional[Tensor] = None,
        *,
        sine_fit: Optional[Dict[str, Any]] = None,
        n_norm: Optional[int] = None,
    ) -> Tensor:
        if p is None:
            p = self.path_tensor(n=int(x.shape[0]))
        pm = self.path_module
        if self.sample == "array":
            return soft_warp_torch(x, p, path_mode=pm.path_mode, path_anchor=pm.path_anchor)
        if self.sample == "analytic_sine":
            if sine_fit is None:
                raise ValueError("analytic_sine sample requires sine_fit")
            n_ref = int(n_norm if n_norm is not None else x.shape[0])
            return soft_warp_sine_torch(
                p,
                n_ref,
                float(sine_fit["omega"]),
                float(sine_fit["phase"]),
                time_scale=float(sine_fit.get("time_scale", 1.0)),
                t_shift=float(sine_fit.get("t_shift", 0.0)),
                reverse_path=False,
            )
        if self.sample == "prefit_sine":
            if sine_fit is None:
                raise ValueError("prefit_sine sample requires sine_fit")
            return soft_warp_prefit_sine_torch(p, sine_fit, path_mode=pm.path_mode)
        raise ValueError(f"unknown sample kind: {self.sample}")

    def warp_numpy(
        self,
        x: np.ndarray,
        p: np.ndarray,
        *,
        sine_fit: Optional[Dict[str, Any]] = None,
        n_norm: Optional[int] = None,
    ) -> np.ndarray:
        pm = self.path_module
        if self.sample == "array":
            return soft_warp_numpy(x, p, path_mode=pm.path_mode, path_anchor=pm.path_anchor)
        if self.sample == "analytic_sine":
            if sine_fit is None:
                raise ValueError("analytic_sine sample requires sine_fit")
            n_ref = int(n_norm if n_norm is not None else len(x))
            return soft_warp_sine_numpy(
                p,
                n_ref,
                float(sine_fit["omega"]),
                float(sine_fit["phase"]),
                time_scale=float(sine_fit.get("time_scale", 1.0)),
                t_shift=float(sine_fit.get("t_shift", 0.0)),
                path_anchor=pm.path_anchor,
            )
        if self.sample == "prefit_sine":
            if sine_fit is None:
                raise ValueError("prefit_sine sample requires sine_fit")
            return soft_warp_prefit_sine_numpy(p, sine_fit, path_mode=pm.path_mode)
        raise ValueError(f"unknown sample kind: {self.sample}")

    def sample_paths(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.path_module.sample_paths(*args, **kwargs)

    def warp_realisations(
        self,
        x: np.ndarray,
        paths: np.ndarray,
        *,
        sine_fit: Optional[Dict[str, Any]] = None,
        n_norm: Optional[int] = None,
    ) -> np.ndarray:
        """Warp covariate under each path row; returns (n_paths, n_time)."""
        out = np.zeros_like(paths, dtype=np.float64)
        for k in range(paths.shape[0]):
            out[k] = self.warp_numpy(x, paths[k], sine_fit=sine_fit, n_norm=n_norm)
        return out
