from __future__ import annotations

import numpy as np
import torch

from warp_regression import (
    WarpModel,
    WarpModelConfig,
    build_synthetic_dataset,
    split_synthetic_holdout,
)


def test_synthetic_holdout(baselines):
    cfg_b = baselines["synthetic"]
    data = build_synthetic_dataset(n=300, seed=cfg_b.get("data_seed", 12))
    split = split_synthetic_holdout(data["n"], n_train=200)
    train_idx, test_idx = split["train_idx"], split["test_idx"]

    model = WarpModel(
        WarpModelConfig(
            readout="parametric",
            n_knots=8,
            epochs=cfg_b["epochs"],
            fit_lambda=cfg_b.get("fit_lambda", 0.5),
            A_init=cfg_b.get("A_init", 1.0),
            C_init=cfg_b.get("C_init", 0.0),
            seed=cfg_b["seed"],
        )
    )
    model.fit(data["y"][train_idx], drivers={"x": data["x"][train_idx]})

    x_train_t = torch.tensor(data["x"], dtype=torch.float32)
    fc = model.forecast(
        len(test_idx),
        n_draws=20,
        seed=cfg_b["seed"],
        x_train=x_train_t,
    )
    y_point = fc.y_point
    y_test = data["y"][test_idx]
    corr = float(np.corrcoef(y_point[test_idx], y_test)[0, 1])
    rmse = float(np.sqrt(np.mean((y_point[test_idx] - y_test) ** 2)))

    assert abs(corr - cfg_b["test_corr"]) <= cfg_b["test_corr_tol"], f"corr={corr:.3f}"
    assert abs(rmse - cfg_b["test_rmse"]) <= cfg_b["test_rmse_tol"], f"rmse={rmse:.3f}"
