from __future__ import annotations

import numpy as np

from utils import (
    fit_dual_sine_shared_warp_nonlinear,
    prepare_lynx_log,
    split_lynx_holdout,
)
from warp_regression import (
    analyze_cycle_lengths,
    fit_dual_sine_log,
    nominal_cycle_length,
)


def test_nominal_cycle_length():
    assert nominal_cycle_length(1000, 5.0) == 200.0
    assert nominal_cycle_length(1000, 5.0, time_scale=2.0) == 100.0
    assert abs(nominal_cycle_length(200, 1.0, dt=0.01) - 100.0) < 1e-12


def test_analyze_cycle_lengths_synthetic_fit():
    n_obs = 400
    omega = 4.0
    n_cal = n_obs
    p_train = np.arange(n_obs, dtype=np.float64)
    sigma_t = 0.05
    n_knots = 8
    sine_fit = {
        "omega": omega,
        "phase": 0.0,
        "time_scale": 1.0,
        "t_shift": 0.0,
    }
    fit = {
        "warp": {"p": p_train, "sigma_t": sigma_t},
        "n_knots": n_knots,
        "n_calendar": n_cal,
        "sine_fit": sine_fit,
    }
    out = analyze_cycle_lengths(fit=fit, n_paths=80, seed=1, unit="steps")
    assert out.lengths.size > 10
    assert abs(out.mean_cycle_length - n_cal / omega) < 1e-6
    assert np.median(out.lengths) > 0.5 * out.mean_cycle_length
    assert np.min(out.lengths) >= 0.5 * out.mean_cycle_length
    assert len(np.unique(out.lengths)) > 1
    # Knot-smooth sampler stays near nominal (dense RW used to downward-bias).
    assert abs(np.median(out.lengths) - out.mean_cycle_length) < 0.15 * out.mean_cycle_length


def test_analyze_cycle_lengths_respects_prefit_dt():
    """Synthetic-style t=(i+1)/100 must yield nominal period 100, not n_train/ω."""
    n_obs = 200
    omega = 1.0
    dt = 0.01
    t0 = dt
    p_train = np.arange(n_obs, dtype=np.float64)
    sine_fit = {
        "omega": omega,
        "phase": 0.0,
        "time_scale": 1.0,
        "t_shift": 0.0,
        "t0": t0,
        "dt": dt,
    }
    fit = {
        "warp": {"p": p_train, "sigma_t": 0.05},
        "n_knots": 8,
        "n_calendar": n_obs,
        "sine_fit": sine_fit,
    }
    out = analyze_cycle_lengths(fit=fit, n_paths=200, seed=2, unit="steps")
    assert abs(out.mean_cycle_length - 100.0) < 1e-9
    assert out.lengths.size > 50
    assert abs(np.median(out.lengths) - 100.0) < 8.0


def test_dense_sampler_biases_short_vs_knots():
    n_obs = 200
    omega = 1.0
    p_train = np.arange(n_obs, dtype=np.float64)
    sine_fit = {
        "omega": omega,
        "phase": 0.0,
        "time_scale": 1.0,
        "t_shift": 0.0,
        "t0": 0.01,
        "dt": 0.01,
    }
    fit = {
        "warp": {"p": p_train, "sigma_t": 0.2},
        "n_knots": 8,
        "sine_fit": sine_fit,
    }
    knots = analyze_cycle_lengths(fit=fit, n_paths=300, seed=3, path_sampler="knots")
    dense = analyze_cycle_lengths(fit=fit, n_paths=300, seed=3, path_sampler="dense")
    assert knots.lengths.size > 50 and dense.lengths.size > 50
    assert np.median(dense.lengths) < np.median(knots.lengths) - 5.0


def test_analyze_cycle_lengths_lynx_has_spread():
    data = prepare_lynx_log()
    split = split_lynx_holdout(data, train_end_year=1910)
    train_idx = split["train_idx"]
    n_train = len(train_idx)
    y_tr = data["y_log"][train_idx]
    t_tr = data["t"][train_idx]
    years_tr = data["years"][train_idx]

    sine_fit = fit_dual_sine_log(y_tr, t_tr, years=years_tr)
    fit_nl = fit_dual_sine_shared_warp_nonlinear(
        y_tr,
        t_tr,
        years=years_tr,
        n_knots=14,
        epochs=400,
        fit_lambda=0.5,
        sine_fit=sine_fit,
    )
    out = analyze_cycle_lengths(fit=fit_nl, n_calendar=n_train, n_paths=200, seed=7)
    assert out.lengths.size > 50
    assert len(np.unique(out.lengths)) > 1
    assert np.std(out.lengths) > 0.1
    # Prefit dt makes nominal match 1/(ω·dt), not the old n_train/ω shortcut.
    assert "dt" in sine_fit
    assert abs(out.mean_cycle_length - 1.0 / (sine_fit["sine1"]["omega"] * sine_fit["dt"])) < 1e-9
    assert abs(np.median(out.lengths) - out.mean_cycle_length) < 0.35 * out.mean_cycle_length
