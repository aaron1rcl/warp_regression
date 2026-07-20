"""Train-path realisations respect start-pin (no reverse RW mapping)."""

from __future__ import annotations

import numpy as np

from utils import build_synthetic_dataset
from warp_regression import WarpModel, predict_realisations_torch


def test_train_realisations_grow_from_start_pin():
    data = build_synthetic_dataset()
    n_train = 80
    x = np.asarray(data["x"][:n_train], dtype=np.float64)
    y = np.asarray(data["y"][:n_train], dtype=np.float64)
    model = WarpModel.from_yaml("synthetic.yaml", n=n_train)
    model.fit(y, covariates={"x": x}, epochs=80, lr=0.05, fit_lambda=0.5, seed=0)

    # Inflate timing noise so path-space spread is obvious.
    model.primary_path.log_sigma_t.data.fill_(0.5)

    paths = predict_realisations_torch(model, x=x, n_draws=200, horizon=n_train, seed=3)
    assert paths.shape == (200, n_train)
    assert float(np.std(paths[:, 0])) < 1e-8
    assert float(np.std(paths[:, -1])) > float(np.std(paths[:, n_train // 4]))
