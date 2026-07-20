from __future__ import annotations

from utils import build_synthetic_dataset, split_synthetic_holdout
import numpy as np

from warp_regression import (
    DEFAULT_PATH_ANCHOR,
    analyze_log_trend,
    prefit,
    prefit_synthetic_path,
    soft_warp_numpy,
    stored_path_offset_numpy,
)


def test_synthetic_prefit_start_anchor():
    data = build_synthetic_dataset()
    split = split_synthetic_holdout(data["n"], n_train=200)
    n_train = split["n_train"]
    y_train = data["y"][:n_train]

    pf = prefit_synthetic_path(data, n_train, path_anchor=DEFAULT_PATH_ANCHOR)
    p_ref = pf.path_ref
    assert p_ref is not None
    off = stored_path_offset_numpy(p_ref)
    assert abs(off[0]) < 1e-9
    assert "x" in pf.covariates
    warped = soft_warp_numpy(pf.covariates["x"], p_ref)
    assert warped.shape == y_train.shape
    assert np.isfinite(warped).all()


def test_analyze_log_trend_residual():
    y = np.linspace(4.0, 8.0, 100) + 0.05 * np.sin(np.linspace(0, 6 * np.pi, 100))
    t_idx = np.arange(1, 101, dtype=float)
    prep = analyze_log_trend(y, t_idx)
    assert prep.residual.shape == y.shape
    assert np.allclose(prep.trend + prep.residual, y)
    assert prep.gamma >= 0.0


def test_analyze_log_trend_saturating_nests_log():
    """γ=0 recovers plain log when z is known; free γ still fits pure-log well."""
    t_idx = np.arange(1, 201, dtype=float)
    z_true = 0.5
    y = 2.0 + 1.5 * np.log(t_idx - z_true)
    prep0 = analyze_log_trend(
        y, t_idx, z_grid=np.array([z_true]), gamma_grid=np.array([0.0])
    )
    assert prep0.gamma == 0.0
    assert abs(prep0.B - 1.5) < 1e-6
    assert abs(prep0.C - 2.0) < 1e-6
    prep = analyze_log_trend(y, t_idx, z_grid=np.array([z_true]))
    assert prep.gamma < 0.25
    assert np.mean((prep.trend - y) ** 2) < 1e-3


def test_prefit_n_sines_one():
    y = np.sin(np.linspace(0, 4 * np.pi, 80))
    t = np.linspace(0, 1, 80)
    pf = prefit(y, t, n_sines=1, omega=2.0)
    assert pf.n_sines == 1
    assert "z" in pf.covariates
