from __future__ import annotations


from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from ..constants import LYNX_CSV, DATA_ROOT, BITCOIN_CSV, SUNSPOTS_CSV


def _load_series_csv(
    csv_path: Path | str,
    transform: str,
    *,
    year_dtype: Any = np.float64,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    years = df["time"].to_numpy(dtype=year_dtype)
    counts = df["value"].to_numpy(dtype=np.float64)
    n = len(years)
    t = np.linspace(0.0, 1.0, n)
    if transform == "log1p":
        y_log = np.log1p(counts)
    elif transform == "log":
        y_log = np.log(np.maximum(counts, 1e-8))
    elif transform == "identity":
        y_log = counts.copy()
    else:
        raise ValueError(transform)
    return {"years": years, "counts": counts, "t": t, "y_log": y_log, "n": n, "transform": transform}


def prepare_lynx_log(csv_path: Path | str = LYNX_CSV, transform: str = "log1p") -> Dict[str, Any]:
    return _load_series_csv(csv_path, transform, year_dtype=np.int32)


def prepare_sunspots(csv_path: Path | str = SUNSPOTS_CSV, transform: str = "identity") -> Dict[str, Any]:
    """Monthly total sunspot number (SILSO, 1749-present). ``years`` is decimal-year."""
    return _load_series_csv(csv_path, transform, year_dtype=np.float64)


def matrix_decomp(x: np.ndarray, n: int, sr: int) -> np.ndarray:
    if x.ndim == 2:
        x = x[:, 0]
    x_i = np.full((n * sr,), np.nan, dtype=np.float32)
    pos = np.arange(0, n) * sr
    x_i[pos] = x
    ones = np.ones((1, x_i.shape[0]), dtype=np.float32)
    x_i = x_i.reshape(-1, 1)
    X = np.dot(x_i, ones)
    I = np.eye(n * sr)
    I[I == 0] = np.nan
    return X * I


def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    if k > 0:
        y = np.concatenate([np.full((k,), np.nan), x])
        y = y[: len(x)]
    elif k < 0:
        k = abs(k)
        y = np.concatenate([x, np.full((k,), np.nan)])
        y = y[k:]
    else:
        y = x
    return y


def X_shift(X: np.ndarray, sr: int, p: np.ndarray) -> np.ndarray:
    if np.max(p) > 0:
        X_out = np.pad(X.copy(), ((0, 0), (0, X.shape[0])), constant_values=np.nan)
    else:
        X_out = X.copy()
    for i in range(1, int(X.shape[0] / sr)):
        x_r = X_out[i * sr, :]
        x_shift = shift(x_r, int(p[i]))
        X_out[i * sr, :] = x_shift
    return X_out


def fill_nan_with_last_value_inplace(x: np.ndarray) -> np.ndarray:
    mask = np.isnan(x)
    last = x[0]
    for i in range(1, len(x)):
        if mask[i]:
            x[i] = last
        else:
            last = x[i]
    return x


def generate_example_waves(n: int, seed: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    phase = np.arange(1, n + 1) / 100.0
    phase = 2.0 * np.pi * phase
    y1 = np.sin(phase)
    y2 = y1 + np.random.normal(0.0, 0.5, size=n)
    return phase.astype(np.float64), y1.astype(np.float64), y2.astype(np.float64)


def generate_sample_perturbations(
    scale: float,
    n: int,
    truncate: bool = False,
    sr: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    perturbations = np.random.normal(loc=0.0, scale=(scale - 1), size=n).astype(int)
    if truncate:
        perturbations[perturbations <= -sr] = -sr
    return perturbations, np.cumsum(perturbations, dtype=np.float64)


def build_synthetic_dataset(
    n: int = 300,
    sr: int = 10,
    scale: float = 10.0,
    noise_std: float = 0.2,
    truncate_perturbations: bool = False,
    seed: int = 12,
) -> Dict[str, Any]:
    """Run WarpPureNumpy.ipynb cells 4–7: warp sin(phase), not raw phase."""
    _, x, y_clean = generate_example_waves(n, seed=seed)
    p_step, p_true = generate_sample_perturbations(scale, n, truncate=truncate_perturbations, sr=sr)
    X = matrix_decomp(x, n, sr)
    X_s = X_shift(X, sr, p_true)
    x_s = np.nanmedian(X_s, axis=0)
    x_out = fill_nan_with_last_value_inplace(x_s.copy())
    x_warped = x_out[::sr][:n].astype(np.float64)
    y = x_warped + np.random.normal(0.0, noise_std, size=n)
    return {
        "x": x.astype(np.float64),
        "phase": _,
        "y": y.astype(np.float64),
        "y_clean": y_clean,
        "x_warped": x_warped,
        "p_true": p_true.astype(np.float64),
        "p_step": p_step,
        "n": n,
        "sr": sr,
        "scale": scale,
        "noise_std": noise_std,
    }



