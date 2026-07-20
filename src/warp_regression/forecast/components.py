"""Observation mean decomposition under a warp path."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..core.warp import soft_warp_numpy, soft_warp_sine_numpy


def observation_components(
    observation: Any,
    calendar: Dict[str, np.ndarray],
    covariate: np.ndarray,
    p: np.ndarray,
    *,
    covariate_name: str = "z",
    sine_fit: Optional[Dict[str, Any]] = None,
    n_norm: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(trend, cyclic, total)`` for a calendar observation model."""
    t_idx = np.asarray(calendar.get("t_idx", np.arange(1, len(p) + 1, dtype=np.float64)), dtype=np.float64)
    t_norm = np.asarray(
        calendar.get("t_norm", (np.arange(len(p), dtype=np.float64) + 0.5) / max(len(p), 1)),
        dtype=np.float64,
    )
    z = np.asarray(covariate, dtype=np.float64)
    if sine_fit is not None:
        n_ref = n_norm if n_norm is not None else len(t_norm)
        z_w = soft_warp_sine_numpy(
            p,
            n_ref,
            sine_fit["omega"],
            sine_fit["phase"],
            time_scale=sine_fit.get("time_scale", 1.0),
            t_shift=sine_fit.get("t_shift", 0.0),
        )
    else:
        z_w = soft_warp_numpy(z, p)
    with torch.no_grad():
        ti = torch.tensor(t_idx, dtype=torch.float32)
        tn = torch.tensor(t_norm, dtype=torch.float32)
        zw = torch.tensor(z_w, dtype=torch.float32)
        if hasattr(observation, "trend_forward") and hasattr(observation, "cyclic_forward"):
            trend = observation.trend_forward(ti).detach().numpy()
            cyclic = observation.cyclic_forward(tn, zw).detach().numpy()
            return trend, cyclic, trend + cyclic
        y = observation({covariate_name: zw}, {"t_idx": ti, "t_norm": tn}).detach().numpy()
        return y * 0.0, y, y


def predict_components(
    observation: Any,
    t_idx: np.ndarray,
    t_norm: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    sine_fit: Optional[Dict[str, Any]] = None,
    n_norm: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return observation_components(
        observation,
        {"t_idx": t_idx, "t_norm": t_norm},
        z,
        p,
        sine_fit=sine_fit,
        n_norm=n_norm,
    )
