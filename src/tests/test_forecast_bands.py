from __future__ import annotations

import numpy as np

from warp_regression.forecast import build_forecast_bands


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
        noise_seed=1,
    )

    c_lo, c_hi = bands["c_q_lo"], bands["c_q_hi"]
    drawn = bands["paths_combined"]
    assert np.all(c_lo <= np.percentile(drawn, 2.5, axis=0) + 1e-12)
    assert np.all(c_hi >= np.percentile(drawn, 97.5, axis=0) - 1e-12)
    assert np.all(c_lo <= bands["c_q50"])
    assert np.all(c_hi >= bands["c_q50"])

    # Combined should not equal the old Minkowski envelope in general.
    mink_lo = bands["t_q_lo"] - 1.96 * sigma_y
    mink_hi = bands["t_q_hi"] + 1.96 * sigma_y
    assert not np.allclose(c_lo, mink_lo)
    assert not np.allclose(c_hi, mink_hi)

    err_margin = np.linspace(0.2, 0.6, n_time)
    bands_hetero = build_forecast_bands(
        paths_terror,
        y_point,
        sigma_y,
        noise_seed=2,
        err_margin=err_margin,
    )
    assert bands_hetero["err_lo"].shape == (n_time,)
    assert np.allclose(bands_hetero["err_hi"] - bands_hetero["err_lo"], 2.0 * err_margin)
