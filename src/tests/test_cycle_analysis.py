from __future__ import annotations

import numpy as np

from warp_regression import (
    analyze_cycle_lengths,
    fit_dual_sine_log,
    fit_dual_sine_shared_warp_nonlinear,
    nominal_cycle_length,
    prepare_lynx_log,
    split_lynx_holdout,
)


def test_nominal_cycle_length():
    assert nominal_cycle_length(1000, 5.0) == 200.0
    assert nominal_cycle_length(1000, 5.0, time_scale=2.0) == 100.0


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
