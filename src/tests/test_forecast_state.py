from __future__ import annotations

import numpy as np
import torch

from warp_regression import (
    DEFAULT_PATH_ANCHOR,
    WarpParametricModel,
    as_forecast_state,
    build_synthetic_dataset,
    forecast_from_state,
    prefit,
    predict_forecast_realisations_torch,
    soft_warp_numpy,
    split_synthetic_holdout,
)
from warp_regression.core.warp import soft_warp_prefit_sine_numpy


def test_default_path_anchor_is_start():
    assert DEFAULT_PATH_ANCHOR == "start"


def test_soft_warp_does_not_reverse_by_default():
    x = np.linspace(0.0, 1.0, 50)
    p = np.arange(50, dtype=np.float64)
    p[10:] += 3.0
    a = soft_warp_numpy(x, p)
    b = soft_warp_numpy(x, p, reverse_path=False)
    assert np.allclose(a, b)


def test_forecast_from_state_frees_terminal_under_start_pin():
    data = build_synthetic_dataset()
    split = split_synthetic_holdout(data["n"], n_train=200)
    n_train = split["n_train"]
    t_tr = (np.arange(n_train) + 1) / 100.0
    t_full = (np.arange(data["n"]) + 1) / 100.0
    pf = prefit(data["y"][:n_train], t_tr, n_sines=1, t_full=t_full)
    sine_fit = pf.sine_fit
    # Extended driver so fit-time expansion can read future sine.
    x_ext = soft_warp_prefit_sine_numpy(np.arange(n_train + 80, dtype=float), sine_fit)
    # Actually pass analytic-length array for soft_warp during fit:
    from warp_regression.drivers.sine import eval_sine_driver

    x_long = eval_sine_driver(
        (np.arange(n_train + 80) + 1) / 100.0,
        sine_fit["omega"],
        sine_fit["phase"],
        time_scale=sine_fit.get("time_scale", 1.0),
        t_shift=sine_fit.get("t_shift", 0.0),
    )
    model = WarpParametricModel(n=n_train, n_knots=8, sr=data["sr"], path_anchor="start")
    model.fit(
        torch.tensor(x_long, dtype=torch.float32),
        torch.tensor(data["y"][:n_train], dtype=torch.float32),
        epochs=400,
        lr=0.03,
        seed=0,
        fit_lambda=0.5,
    )
    with torch.no_grad():
        p = model.path().numpy()
    assert abs(p[0]) < 1e-5

    state = as_forecast_state(model, sine_fit=sine_fit)
    assert state.path_anchor == "start"
    fc = forecast_from_state(state, n_future=30, n_draws=40, n_paths_ci=50, seed=1)
    assert fc["preds"].shape == (40, n_train + 30)
    # Early train stays at MAP; terminal train knot is kicked (not frozen).
    assert np.allclose(fc["paths_p"][:, 0], 0.0, atol=1e-5)
    assert np.std(fc["paths_p"][:, n_train - 1]) > 0.01
    assert np.std(fc["paths_p"][:, -1]) > np.std(fc["paths_p"][:, n_train - 1]) * 0.5


def test_predict_forecast_with_residual_horizon():
    data = build_synthetic_dataset()
    n_train = 200
    t_tr = (np.arange(n_train) + 1) / 100.0
    pf = prefit(data["y"][:n_train], t_tr, n_sines=1, t_full=(np.arange(data["n"]) + 1) / 100.0)
    sine_fit = pf.sine_fit
    from warp_regression.drivers.sine import eval_sine_driver

    x_long = eval_sine_driver((np.arange(n_train + 100) + 1) / 100.0, sine_fit["omega"], sine_fit["phase"])
    model = WarpParametricModel(n=n_train, n_knots=8, sr=10, path_anchor="start")
    y = data["y"][:n_train]
    model.fit(torch.tensor(x_long, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), epochs=300, seed=0)
    fc = predict_forecast_realisations_torch(
        model,
        n_future=40,
        n_draws=30,
        sine_fit=sine_fit,
        y_train=y,
        residual_horizon=20,
        seed=2,
    )
    assert "residual_smooth" in fc
    assert fc["residual_add"].shape == (n_train + 40,)
    # Future residual correction is the flat MA level.
    assert np.allclose(fc["residual_add"][n_train:], fc["residual_smooth"].level)
