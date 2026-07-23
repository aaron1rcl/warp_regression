"""Sparse / MMM covariate helpers (adstock, pulse starts, terror masks).

These are convenience utilities. Callers pass results into ``WarpPath(pins=...)``
and ``compute_dual_loss(..., terror_mask=...)`` — nothing here auto-wires into
soft-warp.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, list]


def pulse_start_indices(x: ArrayLike) -> np.ndarray:
    """First index of each contiguous run where ``x`` is on (nonzero).

    NaN is treated as off. Returns a 1-d int array, possibly empty.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    on = np.isfinite(x) & (x != 0.0)
    if not on.any():
        return np.zeros(0, dtype=np.int64)
    prev = np.concatenate([[False], on[:-1]])
    starts = np.flatnonzero(on & ~prev)
    return starts.astype(np.int64)


def geometric_adstock(
    x: ArrayLike,
    alpha: float,
    *,
    l_max: Optional[int] = None,
) -> np.ndarray:
    """Classic geometric adstock: ``s_t = x_t + alpha * s_{t-1}``.

    Parameters
    ----------
    x :
        Spend (or other impulse) series.
    alpha :
        Retention in ``[0, 1)``. ``alpha=0`` is no carry-over.
    l_max :
        If set, use a finite lag truncation of length ``l_max`` instead of the
        infinite IIR recurrence (still causal).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    a = float(alpha)
    if not (0.0 <= a < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    n = len(x)
    if l_max is not None:
        L = int(l_max)
        if L < 1:
            raise ValueError("l_max must be >= 1")
        out = np.zeros(n, dtype=np.float64)
        for lag in range(L):
            w = a**lag
            if w == 0.0:
                break
            out[lag:] += w * x[: n - lag]
        return out
    out = np.zeros(n, dtype=np.float64)
    s = 0.0
    for t in range(n):
        xt = x[t]
        if not np.isfinite(xt):
            xt = 0.0
        s = xt + a * s
        out[t] = s
    return out


def terror_mask_from_values(x: ArrayLike) -> np.ndarray:
    """Boolean terror mask: ``True`` where ``x`` is on (nonzero and finite)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.isfinite(x) & (x != 0.0)
