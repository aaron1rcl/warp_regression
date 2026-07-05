from __future__ import annotations

import numpy as np

from warp_regression import (
    WarpModel,
    WarpModelConfig,
    build_dual_sines_from_fit,
    fit_dual_sine_log,
    prepare_lynx_log,
    split_lynx_holdout,
)


def test_lynx_holdout(baselines):
    cfg_b = baselines["lynx"]
    data = prepare_lynx_log()
    split = split_lynx_holdout(data, train_end_year=1910)
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    y_log, t, years = data["y_log"], data["t"], data["years"]

    sine_fit = fit_dual_sine_log(y_log[train_idx], t[train_idx], years=years[train_idx])
    z1, z2 = build_dual_sines_from_fit(t, sine_fit)

    model = WarpModel(
        WarpModelConfig(
            readout="dual_mlp",
            n_knots=14,
            epochs=cfg_b["epochs"],
            fit_lambda=1.0,
            seed=cfg_b["seed"],
        )
    )
    fit_res = model.fit(
        y_log[train_idx],
        drivers={"t": t[train_idx], "z1": z1, "z2": z2},
        sine_fit=sine_fit,
        years=years[train_idx],
    )
    model._drivers = {"t": t, "z1": z1, "z2": z2}
    model._fit["_y_log_train"] = y_log[train_idx]
    model._fit["_y_log_test"] = y_log[test_idx]
    model._calendar["split"] = split

    assert fit_res.r2 >= cfg_b["train_r2"] - cfg_b["train_r2_tol"], f"train R²={fit_res.r2:.3f}"

    fc = model.forecast(split["n_test"], n_draws=20, seed=cfg_b["seed"])
    corr = float(np.corrcoef(fc.y_point[test_idx], y_log[test_idx])[0, 1])
    assert abs(corr - cfg_b["test_corr"]) <= cfg_b["test_corr_tol"], f"test corr={corr:.3f}"
