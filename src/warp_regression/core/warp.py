"""Soft-warp: resample a signal (or analytic sine) at fractional path indices.

Three operations, each in numpy (forecast / plots) and torch (training):

* ``soft_warp`` — linear interpolate an array ``x`` at indices ``p``
* ``soft_warp_sine`` — evaluate ``sin(2π ω t + φ)`` at ``t = (p+½)/n``
* ``soft_warp_prefit_sine`` — same sine on the prefit calendar ``t = t0 + dt·p``

The stored path ``p`` is applied as-is (start-pinned training leaves the
train end free). There is no reverse-path mapping anymore.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Array soft-warp (linear interpolation of a discrete driver)
# ---------------------------------------------------------------------------


def soft_warp_numpy(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """``x`` sampled at fractional indices ``p``.

    Clamp to ``[0, n-1.001]`` so floor(p) never hits the last index alone
    (avoids a zero-width interpolation segment at the right edge).
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = len(x)
    p = np.clip(p, 0.0, n - 1.001)
    i0 = np.floor(p).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = p - i0
    return (1.0 - w) * x[i0] + w * x[i1]


def soft_warp_torch(x: Tensor, p: Tensor) -> Tensor:
    """Torch version of :func:`soft_warp_numpy`."""
    n = x.shape[0]
    p = p.clamp(0.0, float(n) - 1.001)
    i0 = torch.floor(p).long().clamp(0, n - 2)
    i1 = i0 + 1
    w = p - i0.to(p.dtype)
    return (1.0 - w) * x[i0] + w * x[i1]


# ---------------------------------------------------------------------------
# Analytic sine at warped index  (t = (p + ½) / n)
# ---------------------------------------------------------------------------


def soft_warp_sine_numpy(
    p: np.ndarray,
    n: int,
    omega: float,
    phase: float,
    time_scale: float = 1.0,
    t_shift: float = 0.0,
) -> np.ndarray:
    """``sin(2π ω t + φ)`` at mid-bin calendar ``t = (p+½)/n`` (no clamp).

    The +½ centres each index in its unit bin. Prefit sines use
    ``t0 + dt·p`` instead — see :func:`soft_warp_prefit_sine_numpy`.
    """
    p = np.asarray(p, dtype=np.float64)
    t = (p + 0.5) / float(n)
    return np.sin(2.0 * np.pi * omega * time_scale * t + phase + t_shift)


def soft_warp_sine_torch(
    p: Tensor,
    n: int,
    omega: float,
    phase: float,
    time_scale: float = 1.0,
    t_shift: float = 0.0,
) -> Tensor:
    """Torch version of :func:`soft_warp_sine_numpy`."""
    t = (p + 0.5) / float(n)
    return torch.sin(2.0 * math.pi * omega * time_scale * t + phase + t_shift)


# ---------------------------------------------------------------------------
# Prefit sine on its own calendar  (t = t0 + dt · p)
# ---------------------------------------------------------------------------


def soft_warp_prefit_sine_numpy(p: np.ndarray, sine_fit: Dict[str, Any]) -> np.ndarray:
    """Evaluate the prefit sine at warped indices using ``sine_fit['t0']``/``dt``.

    Expansion past the train window reads known future values of the same
    analytic sine (no clamp).
    """
    dt = sine_fit.get("dt")
    t0 = sine_fit.get("t0")
    if dt is None or t0 is None:
        raise ValueError("soft_warp_prefit_sine requires sine_fit['t0'] and sine_fit['dt']")
    p = np.asarray(p, dtype=np.float64)
    t = float(t0) + float(dt) * p
    return np.sin(
        2.0
        * np.pi
        * float(sine_fit["omega"])
        * float(sine_fit.get("time_scale", 1.0))
        * t
        + float(sine_fit["phase"])
        + float(sine_fit.get("t_shift", 0.0))
    )


def soft_warp_prefit_sine_torch(p: Tensor, sine_fit: Dict[str, Any]) -> Tensor:
    """Torch version of :func:`soft_warp_prefit_sine_numpy`."""
    dt = sine_fit.get("dt")
    t0 = sine_fit.get("t0")
    if dt is None or t0 is None:
        raise ValueError("soft_warp_prefit_sine requires sine_fit['t0'] and sine_fit['dt']")
    t = float(t0) + float(dt) * p
    return torch.sin(
        2.0
        * math.pi
        * float(sine_fit["omega"])
        * float(sine_fit.get("time_scale", 1.0))
        * t
        + float(sine_fit["phase"])
        + float(sine_fit.get("t_shift", 0.0))
    )
