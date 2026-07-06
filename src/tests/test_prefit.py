from __future__ import annotations

import numpy as np

from warp_regression import (
    DEFAULT_PATH_ANCHOR,
    analyze_log_trend,
    build_synthetic_dataset,
    cumsum_path_to_stored_path,
    prefit,
    prefit_synthetic_path,
    soft_warp_numpy,
    split_synthetic_holdout,
    stored_path_offset_numpy,
)


def test_synthetic_prefit_end_anchor():
    data = build_synthetic_dataset()
    split = split_synthetic_holdout(data["n"], n_train=200)
    n_train = split["n_train"]
    y_train = data["y"][:n_train]

    pf = prefit_synthetic_path(data, n_train, path_anchor=DEFAULT_PATH_ANCHOR)
    p_ref = pf.path_ref
    assert p_ref is not None
    off = stored_path_offset_numpy(p_ref)
    assert abs(off[-1]) < 1e-9
    assert abs(off[0]) > 1.0

    warp_ref = soft_warp_numpy(pf.drivers["x"], p_ref, reverse_path=False)
    corr = float(np.corrcoef(warp_ref, y_train)[0, 1])
    assert corr > 0.75

    p_start = cumsum_path_to_stored_path(data["p_true"], n_train, data["sr"], path_anchor="start")
    warp_start = soft_warp_numpy(pf.drivers["x"], p_start, reverse_path=False)
    corr_start = float(np.corrcoef(warp_start, y_train)[0, 1])
    assert corr > corr_start + 0.4


def test_analyze_log_trend_residual():
    y = np.linspace(4.0, 8.0, 100) + 0.05 * np.sin(np.linspace(0, 6 * np.pi, 100))
    t_idx = np.arange(1, 101, dtype=float)
    prep = analyze_log_trend(y, t_idx)
    assert prep.residual.shape == y.shape
    assert np.allclose(prep.trend + prep.residual, y)


def test_prefit_n_sines_one():
    y = np.sin(np.linspace(0, 4 * np.pi, 80))
    t = np.linspace(0, 1, 80)
    pf = prefit(y, t, n_sines=1, omega=2.0)
    assert pf.n_sines == 1
    assert "z" in pf.drivers
