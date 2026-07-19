from __future__ import annotations

import numpy as np

from warp_regression import (
    WarpModel,
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
    cov_tr = {"z1": z1[train_idx], "z2": z2[train_idx]}

    linear = WarpModel.from_yaml(
        "lynx_linear.yaml",
        overrides={"train": {"epochs": max(400, cfg_b["epochs"] // 4), "seed": cfg_b["seed"]}},
    )
    linear.fit(y_log[train_idx], covariates=cov_tr, sine_fit=sine_fit)
    B = linear.path_B()

    model = WarpModel.from_yaml(
        "lynx.yaml",
        overrides={
            "train": {"epochs": cfg_b["epochs"], "seed": cfg_b["seed"]},
            "observation": {
                "terms": [
                    {
                        "kind": "mlp_on_warped",
                        "inputs": ["z1", "z2"],
                        "hidden": 32,
                        "layers": 2,
                        "act": "gelu",
                    }
                ]
            },
        },
    )
    fit_res = model.fit(y_log[train_idx], covariates=cov_tr, sine_fit=sine_fit, B_init=B)
    model._covariates = {"z1": z1, "z2": z2}
    model._fit["_y_log_train"] = y_log[train_idx]
    model._fit["_y_log_test"] = y_log[test_idx]

    assert fit_res.r2 >= cfg_b["train_r2"] - cfg_b["train_r2_tol"], f"train R²={fit_res.r2:.3f}"

    fc = model.forecast(
        split["n_test"],
        n_draws=20,
        seed=cfg_b["seed"],
        covariates_full={"z1": z1, "z2": z2},
    )
    corr = float(np.corrcoef(fc.y_point[test_idx], y_log[test_idx])[0, 1])
    assert abs(corr - cfg_b["test_corr"]) <= cfg_b["test_corr_tol"], f"test corr={corr:.3f}"
