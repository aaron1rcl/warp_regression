from __future__ import annotations

import numpy as np
import torch

from warp_regression import (
    WarpModel,
    build_synthetic_dataset,
    split_synthetic_holdout,
)


def test_synthetic_holdout(baselines):
    cfg_b = baselines["synthetic"]
    data = build_synthetic_dataset(n=300, seed=cfg_b.get("data_seed", 12))
    split = split_synthetic_holdout(data["n"], n_train=200)
    train_idx, test_idx = split["train_idx"], split["test_idx"]

    model = WarpModel.from_yaml(
        "synthetic.yaml",
        overrides={
            "train": {
                "epochs": cfg_b["epochs"],
                "fit_lambda": cfg_b.get("fit_lambda", 0.5),
                "seed": cfg_b["seed"],
            },
            "observation": {
                "terms": [
                    {
                        "kind": "affine",
                        "inputs": ["x"],
                        "A_init": cfg_b.get("A_init", 1.0),
                        "C_init": cfg_b.get("C_init", 0.0),
                    }
                ]
            },
        },
    )
    model.fit(data["y"][train_idx], covariates={"x": data["x"][train_idx]})

    x_full = data["x"]
    fc = model.forecast(
        len(test_idx),
        n_draws=20,
        seed=cfg_b["seed"],
        covariates_full={"x": x_full},
        x_train=torch.tensor(x_full, dtype=torch.float32),
    )
    y_point = fc.y_point
    y_test = data["y"][test_idx]
    corr = float(np.corrcoef(y_point[test_idx], y_test)[0, 1])
    rmse = float(np.sqrt(np.mean((y_point[test_idx] - y_test) ** 2)))

    assert abs(corr - cfg_b["test_corr"]) <= cfg_b["test_corr_tol"], f"corr={corr:.3f}"
    assert abs(rmse - cfg_b["test_rmse"]) <= cfg_b["test_rmse_tol"], f"rmse={rmse:.3f}"
