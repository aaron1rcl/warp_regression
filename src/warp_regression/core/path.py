"""Warp regression module."""
from __future__ import annotations


import math
from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor

from ..constants import DEFAULT_PATH_ANCHOR, PathAnchor


def path_for_warp_numpy(p: np.ndarray, path_mode: str = "identity") -> np.ndarray:
    """Apply stored path in reverse: p[0] (offset=0) anchors the last time index."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    idx = np.arange(n, dtype=np.float64)
    if path_mode == "identity":
        offset = p - idx
        offset = offset[::-1]
        return idx + offset
    return p[::-1].copy()


def path_for_warp_torch(p: Tensor, path_mode: str = "identity") -> Tensor:
    """Apply stored path in reverse: p[0] (offset=0) anchors the last time index."""
    n = p.shape[0]
    idx = torch.arange(n, device=p.device, dtype=p.dtype)
    if path_mode == "identity":
        offset = p - idx
        offset = offset.flip(0)
        return idx + offset
    return p.flip(0)


def knot_positions(n: int, n_knots: int) -> Tuple[np.ndarray, float]:
    knot_x = np.linspace(0.0, n - 1.0, n_knots)
    rw_interval = float((n - 1) / max(n_knots - 1, 1))
    return knot_x, rw_interval


def zero_gradient_G_numpy(B: np.ndarray) -> np.ndarray:
    G = np.zeros_like(B, dtype=np.float64)
    G[0] = B[0]
    for i in range(1, len(B)):
        G[i] = -np.sum(G[:i]) + B[i]
    return G


def zero_gradient_G_torch(B: Tensor) -> Tensor:
    parts = [B[0]]
    cum = B[0]
    for i in range(1, B.shape[0]):
        gi = -cum + B[i]
        parts.append(gi)
        cum = cum + gi
    return torch.stack(parts)


def _offset_from_G_numpy(G: np.ndarray, n: int, n_knots: int) -> np.ndarray:
    knot_x, _ = knot_positions(n, n_knots)
    knot_x = np.append(knot_x, n - 1.0)
    G = np.append(G, G[-1])
    idx = np.arange(n, dtype=np.float64)
    return np.interp(idx, knot_x, G)


def _offset_from_G_torch(G: Tensor, n: int, n_knots: int) -> Tensor:
    knot_x = torch.linspace(0, n - 1, n_knots, device=G.device, dtype=G.dtype)
    idx = torch.arange(n, device=G.device, dtype=G.dtype)
    offset = torch.zeros(n, device=G.device, dtype=G.dtype)
    for j in range(n_knots - 1):
        x0, x1 = knot_x[j], knot_x[j + 1]
        g0, g1 = G[j], G[j + 1]
        mask = (idx >= x0) & (idx <= x1)
        if x1 > x0:
            w = (idx[mask] - x0) / (x1 - x0)
            offset[mask] = (1.0 - w) * g0 + w * g1
    return offset


def path_from_B_numpy(
    B: np.ndarray,
    n: int,
    n_knots: int,
    path_mode: str = "identity",
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    """Stored path p with offset pinned at train end (``path_anchor='end'``) or start."""
    G = zero_gradient_G_numpy(B)
    G[0] = 0.0
    if path_mode == "identity":
        offset = _offset_from_G_numpy(G, n, n_knots)
        if path_anchor == "end":
            offset = offset[::-1].copy()
            offset[-1] = 0.0
        else:
            offset[0] = 0.0
        p = np.arange(n, dtype=np.float64) + offset
    elif path_mode == "absolute":
        p = _offset_from_G_numpy(G, n, n_knots)
    else:
        raise ValueError(path_mode)
    if path_mode == "identity":
        if path_anchor == "end":
            p[-1] = float(n - 1)
        else:
            p[0] = 0.0
    return p


def path_from_B_torch(
    B: Tensor,
    n: int,
    n_knots: int,
    path_mode: str = "identity",
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> Tensor:
    G = zero_gradient_G_torch(B)
    G = G.clone()
    G[0] = 0.0
    if path_mode == "identity":
        offset = _offset_from_G_torch(G, n, n_knots)
        if path_anchor == "end":
            offset = offset.flip(0).clone()
            offset[-1] = 0.0
        else:
            offset = offset.clone()
            offset[0] = 0.0
        p = torch.arange(n, device=B.device, dtype=B.dtype) + offset
    elif path_mode == "absolute":
        p = _offset_from_G_torch(G, n, n_knots)
    else:
        raise ValueError(path_mode)
    p = p.clone()
    if path_mode == "identity":
        if path_anchor == "end":
            p[-1] = float(n - 1)
        else:
            p[0] = 0.0
    return p
def stored_path_offset_numpy(p: np.ndarray, path_mode: str = "identity") -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if path_mode == "identity":
        return p - np.arange(len(p), dtype=np.float64)
    return p.copy()


def stored_path_offset_torch(p: Tensor, path_mode: str = "identity") -> Tensor:
    if path_mode == "identity":
        idx = torch.arange(p.shape[0], device=p.device, dtype=p.dtype)
        return p - idx
    return p


def applied_path_offset_numpy(p: np.ndarray, path_mode: str = "identity") -> np.ndarray:
    p_apply = path_for_warp_numpy(p, path_mode=path_mode)
    return p_apply - np.arange(len(p_apply), dtype=np.float64)

def point_forecast_path(
    p_train: np.ndarray,
    n_total: int,
    path_mode: str = "identity",
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    """Deterministic path: train offsets fixed; future offset frozen at train end."""
    n_train = len(p_train)
    off = stored_path_offset_numpy(p_train, path_mode=path_mode)
    off_ext = np.zeros(n_total, dtype=np.float64)
    off_ext[:n_train] = off
    off_ext[n_train:] = off[n_train - 1]
    idx = np.arange(n_total, dtype=np.float64)
    p_det = idx + off_ext
    if path_mode == "identity":
        if path_anchor == "end":
            p_det[n_train - 1] = float(n_train - 1)
        else:
            p_det[0] = 0.0
    return p_det
def extended_knot_positions(n_train: int, n_total: int, n_knots: int) -> np.ndarray:
    """Knot grid: train ``linspace(0, n_train-1, n_knots)``, extended with same spacing."""
    train_knots = np.linspace(0.0, float(n_train - 1), n_knots)
    if n_total <= n_train:
        return train_knots
    step = (n_train - 1) / max(n_knots - 1, 1)
    xs = list(train_knots)
    x = float(n_train - 1)
    while x < n_total - 1 - 1e-9:
        x += step
        xs.append(min(x, float(n_total - 1)))
        if xs[-1] >= n_total - 1 - 1e-9:
            break
    if xs[-1] < n_total - 1 - 1e-9:
        xs.append(float(n_total - 1))
    return np.unique(np.asarray(xs, dtype=np.float64))


