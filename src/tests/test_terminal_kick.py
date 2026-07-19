"""Terminal train-knot kick + future RW (start-pin)."""

from __future__ import annotations

import numpy as np

from warp_regression import sample_warp_paths_future, sample_warp_paths_future_knots


def test_knot_sampler_kicks_last_train_knot():
    n_train, n_future = 40, 20
    p_train = np.arange(n_train, dtype=np.float64)
    p_train[-1] += 2.5
    paths = sample_warp_paths_future_knots(
        p_train, n_train + n_future, n_train, sigma_t=0.05, n_knots=6, n_paths=200, seed=3
    )
    # Start pin + early MAP freeze.
    assert np.allclose(paths[:, 0], 0.0)
    assert np.allclose(paths[:, 1:10], p_train[None, 1:10], atol=1e-9)
    # Last train index moves; future spreads further.
    assert float(paths[:, n_train - 1].std()) > 0.2
    assert float(paths[:, -1].std()) > float(paths[:, n_train - 1].std())


def test_dense_sampler_kicks_terminal_offset():
    n_train, n_future = 30, 15
    p_train = np.arange(n_train, dtype=np.float64)
    paths = sample_warp_paths_future(
        p_train, n_train + n_future, n_train, sigma_t=0.04, n_knots=5, n_paths=150, seed=4
    )
    assert np.allclose(paths[:, : n_train - 1], p_train[None, : n_train - 1])
    assert float(paths[:, n_train - 1].std()) > 0.1
