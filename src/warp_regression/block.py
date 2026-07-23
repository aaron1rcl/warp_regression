"""Portable warp path + single-series WarpRegression block."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .core.path import (
    DEFAULT_PATH_ANCHOR,
    PathAnchor,
    PinsLike,
    normalize_pins,
    path_from_B_torch,
    pins_parameter_size,
)
from .core.training import terror_likelihood_torch, terror_path_for_likelihood_torch
from .core.warp import (
    soft_warp_numpy,
    soft_warp_prefit_sine_numpy,
    soft_warp_sine_numpy,
    soft_warp_torch,
    soft_warp_prefit_sine_torch,
    soft_warp_sine_torch,
)

# How the covariate is obtained at path indices p:
#   array          — soft_warp of a discrete driver array
#   analytic_sine  — closed-form sin at t = (p+½)/n
#   prefit_sine    — closed-form sin on the prefit calendar t0 + dt·p
CovariateKind = Literal["array", "analytic_sine", "prefit_sine"]


class WarpPath(nn.Module):
    """Shared knot parameters B and terror scale σ_t for one or more warp blocks."""

    def __init__(
        self,
        n: int,
        n_knots: int = 8,
        *,
        path_mode: str = "identity",
        path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
        pins: PinsLike = None,
        B_init: Optional[Union[Tensor, np.ndarray]] = None,
        log_sigma_t_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.n_knots = int(n_knots)
        self.path_mode = path_mode
        self.path_anchor: PathAnchor = path_anchor
        self.pins = normalize_pins(pins, self.n)
        b_dim = pins_parameter_size(self.n, self.n_knots, self.pins)
        if B_init is None:
            B0 = torch.zeros(b_dim)
        else:
            B0 = torch.as_tensor(B_init, dtype=torch.float32).reshape(-1)
            if B0.numel() != b_dim:
                raise ValueError(f"B_init length {B0.numel()} != expected {b_dim}")
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
            pins=self.pins,
        )

    def terror_nll(
        self,
        p: Optional[Tensor] = None,
        *,
        mask: Optional[Union[Tensor, np.ndarray]] = None,
    ) -> Tensor:
        """Negative terror log-likelihood (−terror_ll) for dual loss."""
        if p is None:
            p = self.path()
        p_terror = terror_path_for_likelihood_torch(p, path_mode=self.path_mode)
        return -terror_likelihood_torch(
            p_terror, self.sigma_t, self.n_knots, mask=mask
        )

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

    ``covariate_kind`` selects how values are read at path indices (array
    interpolation vs analytic / prefit sine). Portable to other PyTorch
    problems; path-level forecast helpers live on the attached ``WarpPath``.
    """

    def __init__(
        self,
        path: WarpPath,
        *,
        covariate_kind: CovariateKind = "array",
        name: str = "x",
    ) -> None:
        super().__init__()
        self.path_module = path
        self.covariate_kind: CovariateKind = covariate_kind
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
        covariate_kind: Optional[CovariateKind] = None,
    ) -> Tensor:
        """Soft-warp covariate at path ``p`` (torch; used in training)."""
        if p is None:
            p = self.path_tensor(n=int(x.shape[0]))
        kind = covariate_kind if covariate_kind is not None else self.covariate_kind
        if kind == "array":
            return soft_warp_torch(x, p)
        if kind == "analytic_sine":
            if sine_fit is None:
                raise ValueError("analytic_sine requires sine_fit")
            n_ref = int(n_norm if n_norm is not None else x.shape[0])
            return soft_warp_sine_torch(
                p,
                n_ref,
                float(sine_fit["omega"]),
                float(sine_fit["phase"]),
                time_scale=float(sine_fit.get("time_scale", 1.0)),
                t_shift=float(sine_fit.get("t_shift", 0.0)),
            )
        if kind == "prefit_sine":
            if sine_fit is None:
                raise ValueError("prefit_sine requires sine_fit")
            return soft_warp_prefit_sine_torch(p, sine_fit)
        raise ValueError(f"unknown covariate_kind: {kind}")

    def warp_numpy(
        self,
        x: np.ndarray,
        p: np.ndarray,
        *,
        sine_fit: Optional[Dict[str, Any]] = None,
        n_norm: Optional[int] = None,
        covariate_kind: Optional[CovariateKind] = None,
    ) -> np.ndarray:
        """Soft-warp covariate at path ``p`` (numpy; used in forecast / plots)."""
        kind = covariate_kind if covariate_kind is not None else self.covariate_kind
        if kind == "array":
            return soft_warp_numpy(x, p)
        if kind == "analytic_sine":
            if sine_fit is None:
                raise ValueError("analytic_sine requires sine_fit")
            n_ref = int(n_norm if n_norm is not None else len(x))
            return soft_warp_sine_numpy(
                p,
                n_ref,
                float(sine_fit["omega"]),
                float(sine_fit["phase"]),
                time_scale=float(sine_fit.get("time_scale", 1.0)),
                t_shift=float(sine_fit.get("t_shift", 0.0)),
            )
        if kind == "prefit_sine":
            if sine_fit is None:
                raise ValueError("prefit_sine requires sine_fit")
            return soft_warp_prefit_sine_numpy(p, sine_fit)
        raise ValueError(f"unknown covariate_kind: {kind}")

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
