from __future__ import annotations

import numpy as np

from warp_regression.forecast import build_forecast_bands, ci_band_params, interval_coverage


def test_ci_band_params_95():
    pct_lo, pct_hi, z = ci_band_params(0.95)
    assert abs(pct_lo - 2.5) < 1e-9
    assert abs(pct_hi - 97.5) < 1e-9
    assert abs(z - 1.959963984540054) < 1e-6


def test_interval_coverage():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo = np.array([-1.0, 0.5, 1.5, 2.5])
    hi = np.array([0.5, 1.5, 2.5, 2.9])
    assert abs(interval_coverage(y, lo, hi) - 0.75) < 1e-12


def test_build_forecast_bands_combined_from_joint_draws():
    rng = np.random.default_rng(0)
    n_paths, n_time = 400, 50
    paths_terror = rng.normal(0.0, 1.0, size=(n_paths, n_time))
    y_point = np.zeros(n_time)
    sigma_y = 0.2

    bands = build_forecast_bands(
        paths_terror,
        y_point,
        sigma_y,
        ci=0.95,
        noise_seed=1,
    )
    assert bands["ci"] == 0.95

    c_lo, c_hi = bands["c_q_lo"], bands["c_q_hi"]
    drawn = bands["paths_combined"]
    assert np.all(c_lo <= np.percentile(drawn, 2.5, axis=0) + 1e-12)
    assert np.all(c_hi >= np.percentile(drawn, 97.5, axis=0) - 1e-12)
    assert np.all(c_lo <= bands["c_q50"])
    assert np.all(c_hi >= bands["c_q50"])

    # iid e per time step: combined − terror varies within a realisation.
    delta = drawn - paths_terror
    assert bands["err_draws"].shape == (n_paths, n_time)
    assert np.allclose(delta, bands["err_draws"])
    assert not np.allclose(delta, delta[:, :1])

    # Combined should not equal the old Minkowski envelope in general.
    mink_lo = bands["t_q_lo"] - bands["z_err"] * sigma_y
    mink_hi = bands["t_q_hi"] + bands["z_err"] * sigma_y
    assert not np.allclose(c_lo, mink_lo)
    assert not np.allclose(c_hi, mink_hi)

    err_margin = np.linspace(0.2, 0.6, n_time)
    bands_hetero = build_forecast_bands(
        paths_terror,
        y_point,
        sigma_y,
        ci=0.95,
        noise_seed=2,
        err_margin=err_margin,
    )
    assert bands_hetero["err_lo"].shape == (n_time,)
    assert np.allclose(bands_hetero["err_hi"] - bands_hetero["err_lo"], 2.0 * err_margin)
    delta_h = bands_hetero["paths_combined"] - paths_terror
    sigma_draw = err_margin / bands_hetero["z_err"]
    assert np.allclose(delta_h.std(axis=0), sigma_draw, rtol=0.15, atol=0.05)


def test_build_forecast_bands_iid_error_widens_vs_terror_only():
    rng = np.random.default_rng(3)
    paths_terror = rng.normal(0.0, 0.05, size=(200, 30))
    y_point = np.zeros(30)
    bands = build_forecast_bands(paths_terror, y_point, sigma_y=0.5, ci=0.9, noise_seed=4)
    assert bands["ci"] == 0.9
    assert abs(bands["pct_lo"] - 5.0) < 1e-9
    assert float(bands["paths_combined"].std()) > 0.35
