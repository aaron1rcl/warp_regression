"""Warp path geometry: knot coefficients → stored path → offsets.

Glossary
--------
``B``
    Free knot coefficients (length ``n_knots``), the trainable path parameters.
``G``
    Zero-gradient transform of ``B`` (``zero_gradient_G_*``). Forces the knot
    increments to sum near zero so the path cannot drift purely linearly.
``p`` (stored path)
    Fractional covariate indices, length ``n``. Under ``path_mode='identity'``,
    ``p[i] = i + o[i]`` where ``o`` is the warp offset.
``path_anchor``
    Where the offset is pinned during training. Default ``'start'`` pins
    ``o[0] = 0`` and leaves the train end free so forecast RW can continue.
    ``'end'`` pins ``o[-1] = 0`` instead.
Soft-warp
    Applies the **stored** ``p`` as-is. There is no reverse-path remapping in
    the current training / forecast path.

Legacy helpers ``path_for_warp_*`` / ``applied_path_offset_numpy`` reverse the
offset sequence (old end-anchored convention). Soft-warp does not use them;
they remain for notebook diagnostics only.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

PathAnchor = Literal["start", "end"]
DEFAULT_PATH_ANCHOR: PathAnchor = "start"
PinsLike = Optional[Union[Sequence[int], np.ndarray, Tensor]]


# ---------------------------------------------------------------------------
# Legacy reverse mapping (diagnostics only — soft-warp does not use these)
# ---------------------------------------------------------------------------


def path_for_warp_numpy(p: np.ndarray, path_mode: str = "identity") -> np.ndarray:
    """Reverse offset order: stored start-pin → applied end-pin coordinates."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    idx = np.arange(n, dtype=np.float64)
    if path_mode == "identity":
        offset = p - idx
        offset = offset[::-1]
        return idx + offset
    return p[::-1].copy()


def path_for_warp_torch(p: Tensor, path_mode: str = "identity") -> Tensor:
    """Torch version of :func:`path_for_warp_numpy`."""
    n = p.shape[0]
    idx = torch.arange(n, device=p.device, dtype=p.dtype)
    if path_mode == "identity":
        offset = p - idx
        offset = offset.flip(0)
        return idx + offset
    return p.flip(0)


def applied_path_offset_numpy(p: np.ndarray, path_mode: str = "identity") -> np.ndarray:
    """Offsets after legacy reverse mapping (notebook diagnostics)."""
    p_apply = path_for_warp_numpy(p, path_mode=path_mode)
    return p_apply - np.arange(len(p_apply), dtype=np.float64)


# ---------------------------------------------------------------------------
# Knot grid and B → G → path
# ---------------------------------------------------------------------------


def knot_positions(n: int, n_knots: int) -> Tuple[np.ndarray, float]:
    """Uniform knot locations on ``[0, n-1]`` and the RW segment width."""
    knot_x = np.linspace(0.0, n - 1.0, n_knots)
    rw_interval = float((n - 1) / max(n_knots - 1, 1))
    return knot_x, rw_interval


def zero_gradient_G_torch(B: Tensor) -> Tensor:
    """Map free knot coeffs ``B`` → offsets ``G`` with near-zero cumulative drift.

    Recursively ``G[i] = B[i] - sum(G[:i])``, so increments roughly cancel and
    the path cannot absorb a pure linear trend into the knots alone.
    """
    parts = [B[0]]
    cum = B[0]
    for i in range(1, B.shape[0]):
        gi = -cum + B[i]
        parts.append(gi)
        cum = cum + gi
    return torch.stack(parts)


def _offset_from_G_torch(G: Tensor, n: int, n_knots: int) -> Tensor:
    """Piecewise-linear interpolate knot offsets ``G`` onto every time index."""
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


def normalize_pins(pins: PinsLike, n: int) -> Optional[np.ndarray]:
    """Sorted unique pin indices in ``[0, n-1]``, or ``None`` if empty/absent."""
    if pins is None:
        return None
    if isinstance(pins, Tensor):
        arr = pins.detach().cpu().numpy()
    else:
        arr = np.asarray(pins)
    if arr.size == 0:
        return None
    arr = np.unique(arr.astype(np.int64))
    if arr[0] < 0 or arr[-1] >= int(n):
        raise ValueError(f"pins must lie in [0, {n - 1}], got {arr.tolist()}")
    return arr


def allocate_interval_knots(lengths: Sequence[int], n_knots: int) -> List[int]:
    """Split a knot budget across intervals (at least 2 knots each when possible)."""
    lengths = [max(int(L), 1) for L in lengths]
    m = len(lengths)
    if m == 0:
        return []
    budget = max(int(n_knots), 2 * m)
    total = float(sum(lengths))
    raw = [max(2, int(round(budget * (L / total)))) for L in lengths]
    # Fix rounding so sum == budget
    while sum(raw) > budget:
        j = int(np.argmax(raw))
        if raw[j] > 2:
            raw[j] -= 1
        else:
            break
    while sum(raw) < budget:
        j = int(np.argmax(lengths))
        raw[j] += 1
    return raw


def _segment_offset_torch(
    B_seg: Tensor,
    n_seg: int,
    n_knots_seg: int,
    *,
    pin_start: bool,
    pin_end: bool,
) -> Tensor:
    """PWL offset on a segment of length ``n_seg`` with optional endpoint pins."""
    if n_seg <= 0:
        return B_seg.new_zeros(0)
    if n_seg == 1:
        return B_seg.new_zeros(1)
    k = max(int(n_knots_seg), 2)
    if B_seg.numel() < k:
        pad = B_seg.new_zeros(k - B_seg.numel())
        B_use = torch.cat([B_seg, pad])
    else:
        B_use = B_seg[:k]
    G = zero_gradient_G_torch(B_use)
    G = G.clone()
    G[0] = 0.0
    offset = _offset_from_G_torch(G, n_seg, k).clone()
    if pin_start:
        offset = offset - offset[0]
    if pin_end:
        # Remove residual end drift with a linear ramp (keeps start pin if set).
        end = offset[-1]
        if pin_start:
            ramp = torch.linspace(0.0, 1.0, n_seg, device=offset.device, dtype=offset.dtype)
            offset = offset - end * ramp
        else:
            offset = offset - offset[-1]
    return offset


def path_from_B_torch(
    B: Tensor,
    n: int,
    n_knots: int,
    path_mode: str = "identity",
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
    pins: PinsLike = None,
) -> Tensor:
    """Build stored path ``p`` from knot coefficients ``B``.

    Without ``pins``, forces ``G[0] = 0``, then pins either ``p[0]``
    (``path_anchor='start'``) or ``p[-1]`` (``'end'``).

    With ``pins``, builds concatenated bridges so ``p[i] = i`` exactly at every
    pin (identity mode only). Segments between consecutive pins are both-ends
    pinned; the trailing segment after the last pin (if any) is start-pinned
    and end-free.
    """
    pin_arr = normalize_pins(pins, n)
    if pin_arr is not None and path_mode != "identity":
        raise ValueError("pins are only supported with path_mode='identity'")

    if pin_arr is None:
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

    pin_set = set(int(i) for i in pin_arr.tolist())
    # Cover [0, n-1] with cuts at 0, pins, and n-1
    cuts = [0] + [i for i in pin_arr.tolist() if i not in (0, n - 1)] + [n - 1]
    cuts = sorted(set(cuts))

    intervals: List[Tuple[int, int, bool]] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        # Both-ends pinned unless this is the open trailing segment past last pin
        trailing_open = b == n - 1 and (n - 1) not in pin_set
        intervals.append((a, b, not trailing_open))

    lengths = [b - a + 1 for a, b, _ in intervals]
    knot_counts = allocate_interval_knots(lengths, n_knots)

    offset = torch.zeros(n, device=B.device, dtype=B.dtype)
    cursor = 0
    for (a, b, pin_end), k_i in zip(intervals, knot_counts):
        n_seg = b - a + 1
        B_seg = B[cursor : cursor + k_i]
        if B_seg.numel() < k_i:
            B_seg = torch.cat([B_seg, B.new_zeros(k_i - B_seg.numel())])
        cursor += k_i
        seg_off = _segment_offset_torch(
            B_seg, n_seg, k_i, pin_start=True, pin_end=pin_end
        )
        offset[a : a + n_seg] = seg_off

    idx = torch.arange(n, device=B.device, dtype=B.dtype)
    p = idx + offset
    # Exact identity at user pins (and start anchor at 0)
    for i in pin_set:
        p[i] = float(i)
    if path_anchor == "start":
        p[0] = 0.0
    return p


def pins_parameter_size(n: int, n_knots: int, pins: PinsLike) -> int:
    """Number of free ``B`` entries needed for ``path_from_B_torch`` with ``pins``."""
    pin_arr = normalize_pins(pins, n)
    if pin_arr is None:
        return int(n_knots)
    pin_set = set(int(i) for i in pin_arr.tolist())
    cuts = [0] + [i for i in pin_arr.tolist() if i not in (0, n - 1)] + [n - 1]
    cuts = sorted(set(cuts))
    lengths = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            lengths.append(b - a + 1)
    return int(sum(allocate_interval_knots(lengths, n_knots)))


# ---------------------------------------------------------------------------
# Offsets and deterministic forecast extension
# ---------------------------------------------------------------------------


def stored_path_offset_numpy(p: np.ndarray, path_mode: str = "identity") -> np.ndarray:
    """``o[i] = p[i] - i`` under identity mode (else ``p`` itself)."""
    p = np.asarray(p, dtype=np.float64)
    if path_mode == "identity":
        return p - np.arange(len(p), dtype=np.float64)
    return p.copy()


def stored_path_offset_torch(p: Tensor, path_mode: str = "identity") -> Tensor:
    if path_mode == "identity":
        idx = torch.arange(p.shape[0], device=p.device, dtype=p.dtype)
        return p - idx
    return p


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
