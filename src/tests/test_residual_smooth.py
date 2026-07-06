from __future__ import annotations

import numpy as np

from warp_regression import fit_residual_smooth, forecast_residual


def test_residual_forecast_shape_and_anchor():
    rng = np.random.default_rng(0)
    residual = 0.02 * np.arange(50) + rng.normal(0, 0.01, 50)
    res_fit = fit_residual_smooth(residual, kind="local_trend")
    fut = forecast_residual(res_fit, 10)
    assert fut.shape == (10,)
    assert np.isfinite(fut).all()
    assert fut[0] == res_fit.level + res_fit.trend


def test_local_level_constant_residual():
    residual = np.ones(30) * 0.5
    res_fit = fit_residual_smooth(residual, kind="local_level")
    fut = forecast_residual(res_fit, 5)
    assert np.allclose(fut, fut[0])
    assert fut[0] > 0.4
