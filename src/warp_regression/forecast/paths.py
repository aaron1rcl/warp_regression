"""Future warp-path samplers for forecast terror."""

from __future__ import annotations

import numpy as np

from ..core.path import DEFAULT_PATH_ANCHOR, PathAnchor, knot_positions


def per_index_rw_sigma(sigma_t: float, n: int, n_knots: int) -> float:
    """Per-index RW std consistent with knot-segment terror scale sigma_t."""
    _, rw = knot_positions(n, n_knots)
    return float(sigma_t * np.sqrt(max(rw, 1e-8)))


def sample_warp_paths_future(
    p_train: np.ndarray,
    n_total: int,
    n_train: int,
    sigma_t: float,
    n_knots: int,
    n_paths: int = 500,
    seed: int = 0,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    """Per-index future RW, with a terror-scale kick at the free train end.

    Under ``path_anchor='start'`` the MAP train path is kept through
    ``n_train-2``, the terminal offset gets one N(0, sigma) kick, and the
    future RW continues from that kicked end (last index not frozen).
    """
    rng = np.random.default_rng(seed)
    sigma_step = per_index_rw_sigma(sigma_t, n_train, n_knots)
    offset_train = p_train - np.arange(n_train, dtype=np.float64)
    paths = np.zeros((n_paths, n_total), dtype=np.float64)
    idx = np.arange(n_total, dtype=np.float64)
    for k in range(n_paths):
        off = np.zeros(n_total)
        off[:n_train] = offset_train
        if path_anchor == "start" and n_train >= 1:
            off[n_train - 1] = offset_train[n_train - 1] + rng.normal(0.0, sigma_step)
        for i in range(n_train, n_total):
            off[i] = off[i - 1] + rng.normal(0.0, sigma_step)
        paths[k] = idx + off
        if path_anchor == "end":
            paths[k, :n_train] = p_train
            paths[k, n_train - 1] = float(n_train - 1)
        else:
            if n_train > 1:
                paths[k, : n_train - 1] = p_train[: n_train - 1]
            paths[k, 0] = 0.0
    return paths


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


def sample_warp_paths_future_knots(
    p_train: np.ndarray,
    n_total: int,
    n_train: int,
    sigma_t: float,
    n_knots: int,
    n_paths: int = 500,
    seed: int = 0,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    """Future warp paths via knot-level RW + piecewise-linear offset."""
    rng = np.random.default_rng(seed)
    _, rw_interval = knot_positions(n_train, n_knots)
    sigma_knot = float(sigma_t * rw_interval)
    knot_x = extended_knot_positions(n_train, n_total, n_knots)
    idx_train = np.arange(n_train, dtype=np.float64)
    off_train = p_train - idx_train
    off_anchor = np.interp(knot_x, idx_train, off_train)
    n_train_knots = len(np.linspace(0.0, float(n_train - 1), n_knots))
    idx_full = np.arange(n_total, dtype=np.float64)
    freeze_n = int(np.floor(knot_x[max(n_train_knots - 2, 0)])) + 1 if n_train_knots >= 2 else 1
    freeze_n = min(max(freeze_n, 1), n_train)
    paths = np.zeros((n_paths, n_total), dtype=np.float64)
    for k in range(n_paths):
        off_k = off_anchor.copy()
        if path_anchor == "start" and n_train_knots >= 1:
            i_last = n_train_knots - 1
            off_k[i_last] = off_anchor[i_last] + rng.normal(0.0, sigma_knot)
            for j in range(n_train_knots, len(knot_x)):
                off_k[j] = off_k[j - 1] + rng.normal(0.0, sigma_knot)
        else:
            for j in range(n_train_knots, len(knot_x)):
                off_k[j] = off_k[j - 1] + rng.normal(0.0, sigma_knot)
        off_full = np.interp(idx_full, knot_x, off_k)
        paths[k] = idx_full + off_full
        if path_anchor == "end":
            paths[k, :n_train] = p_train
            paths[k, n_train - 1] = float(n_train - 1)
        else:
            paths[k, :freeze_n] = p_train[:freeze_n]
            paths[k, 0] = 0.0
    return paths
