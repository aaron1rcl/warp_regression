"""Warp regression module."""
from __future__ import annotations


import math
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch import Tensor

from ..constants import DEFAULT_PATH_ANCHOR, PathAnchor
from .path import path_for_warp_numpy, path_for_warp_torch


def soft_warp_prefit_sine_numpy(
    p: np.ndarray,
    sine_fit: Dict[str, Any],
    *,
    reverse_path: Optional[bool] = None,
    path_mode: str = "identity",
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    """Evaluate the prefit sine at warped indices using its calendar ``t0``/``dt``.

    Unlike array soft-warp, this has no boundary clamp: expansion past the
    train window reads known future values of the same analytic sine.
    ``t = t0 + dt · p`` matches prefit when ``t[i] = t0 + dt·i``.
    """
    if reverse_path is None:
        reverse_path = False
    p_use = (
        path_for_warp_numpy(p, path_mode=path_mode)
        if reverse_path
        else np.asarray(p, dtype=np.float64)
    )
    dt = sine_fit.get("dt")
    t0 = sine_fit.get("t0")
    if dt is None or t0 is None:
        raise ValueError("soft_warp_prefit_sine_numpy requires sine_fit['t0'] and sine_fit['dt']")
    t = float(t0) + float(dt) * p_use
    from ..drivers.sine import eval_sine_driver

    return eval_sine_driver(
        t,
        float(sine_fit["omega"]),
        float(sine_fit["phase"]),
        time_scale=float(sine_fit.get("time_scale", 1.0)),
        t_shift=float(sine_fit.get("t_shift", 0.0)),
    )


def soft_warp_prefit_sine_torch(
    p: Tensor,
    sine_fit: Dict[str, Any],
    *,
    reverse_path: Optional[bool] = None,
    path_mode: str = "identity",
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> Tensor:
    """Torch version of :func:`soft_warp_prefit_sine_numpy`."""
    if reverse_path is None:
        reverse_path = False
    p_use = path_for_warp_torch(p, path_mode=path_mode) if reverse_path else p
    dt = sine_fit.get("dt")
    t0 = sine_fit.get("t0")
    if dt is None or t0 is None:
        raise ValueError("soft_warp_prefit_sine_torch requires sine_fit['t0'] and sine_fit['dt']")
    t = float(t0) + float(dt) * p_use
    omega = float(sine_fit["omega"])
    phase = float(sine_fit["phase"])
    time_scale = float(sine_fit.get("time_scale", 1.0))
    t_shift = float(sine_fit.get("t_shift", 0.0))
    return torch.sin(2.0 * math.pi * omega * time_scale * t + phase + t_shift)


def soft_warp_numpy(
    x: np.ndarray,
    p: np.ndarray,
    reverse_path: Optional[bool] = None,
    path_mode: str = "identity",
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # Default: stored path is applied as-is (no reverse). Start-pin leaves the
    # train end free; end-pin is also stored=applied.
    if reverse_path is None:
        reverse_path = False
    p_use = (
        path_for_warp_numpy(p, path_mode=path_mode)
        if reverse_path
        else np.asarray(p, dtype=np.float64)
    )
    n = len(x)
    # Keep indices in range for generic (non-periodic) drivers; see soft_warp_sine_* for sin.
    p_use = np.clip(p_use, 0.0, n - 1.001)
    i0 = np.floor(p_use).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = p_use - i0
    return (1.0 - w) * x[i0] + w * x[i1]


def soft_warp_sine_numpy(
    p: np.ndarray,
    n: int,
    omega: float,
    phase: float,
    time_scale: float = 1.0,
    t_shift: float = 0.0,
    reverse_path: Optional[bool] = None,
    path_mode: str = "identity",
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    """Sample sin(2πωt+φ) at warped index p; t=(p+½)/n with no boundary clamp."""
    if reverse_path is None:
        reverse_path = False
    p_use = (
        path_for_warp_numpy(p, path_mode=path_mode)
        if reverse_path
        else np.asarray(p, dtype=np.float64)
    )
    t_warp = (p_use + 0.5) / n
    return np.sin(2.0 * np.pi * omega * time_scale * t_warp + phase + t_shift)


def soft_warp_sine_torch(
    p: Tensor,
    n: int,
    omega: float,
    phase: float,
    time_scale: float = 1.0,
    t_shift: float = 0.0,
    reverse_path: Optional[bool] = None,
    path_mode: str = "identity",
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> Tensor:
    """Torch sin driver at warped index p; t=(p+½)/n with no boundary clamp."""
    if reverse_path is None:
        reverse_path = False
    p_use = path_for_warp_torch(p, path_mode=path_mode) if reverse_path else p
    t_warp = (p_use + 0.5) / n
    return torch.sin(
        2.0 * math.pi * omega * time_scale * t_warp + phase + t_shift
    )


def soft_warp_torch(
    x: Tensor,
    p: Tensor,
    reverse_path: Optional[bool] = None,
    path_mode: str = "identity",
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> Tensor:
    if reverse_path is None:
        reverse_path = False
    p_use = path_for_warp_torch(p, path_mode=path_mode) if reverse_path else p
    n = x.shape[0]
    p_use = p_use.clamp(0.0, float(n) - 1.001)
    i0 = torch.floor(p_use).long().clamp(0, n - 2)
    i1 = i0 + 1
    w = p_use - i0.to(p_use.dtype)
    return (1.0 - w) * x[i0] + w * x[i1]


def fill_nan_with_last_value(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    last = out[0]
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def discrete_warp_numpy(x: np.ndarray, p: np.ndarray, n: int, sr: int = 10) -> np.ndarray:
    """Discrete warp: Gaussian-shift path, column aggregate (O(n·sr) memory)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(p) != n:
        p = np.interp(np.arange(n), np.linspace(0, n - 1, len(p)), p)
    nrow = n * sr
    ncol = nrow + (int(np.max(p)) if p.size and np.max(p) > 0 else 0)
    cols: List[List[float]] = [[] for _ in range(ncol)]
    for i in range(n):
        r = i * sr
        if i == 0:
            if r < ncol:
                cols[r].append(float(x[i]))
        else:
            c = r + int(p[i])
            if 0 <= c < ncol:
                cols[c].append(float(x[i]))
    med = np.array([np.median(v) if v else np.nan for v in cols], dtype=np.float64)
    return fill_nan_with_last_value(med)[::sr][:n]

