"""Train-path realisations respect start-pin (no reverse RW mapping)."""

from __future__ import annotations

import numpy as np
import torch

from warp_regression import WarpParametricModel, build_synthetic_dataset, predict_realisations_torch


def test_train_realisations_grow_from_start_pin():
    data = build_synthetic_dataset()
    n_train = 80
    x = torch.tensor(data["x"][:n_train], dtype=torch.float32)
    y = torch.tensor(data["y"][:n_train], dtype=torch.float32)
    model = WarpParametricModel(n=n_train, n_knots=8, sr=data["sr"], path_anchor="start")
    model.fit(x, y, epochs=80, lr=0.05, fit_lambda=0.5, seed=0)

    # Inflate timing noise so path-space spread is obvious.
    with torch.no_grad():
        model.log_sigma_t.fill_(np.log(0.05))

    captured: list[np.ndarray] = []
    orig_predict = model.predict

    def _capture(x_in, p=None):
        if p is not None:
            captured.append(np.asarray(p.detach().cpu().numpy(), dtype=np.float64).copy())
        return orig_predict(x_in, p)

    model.predict = _capture  # type: ignore[method-assign]
    predict_realisations_torch(model, x, n_draws=200, horizon=n_train, seed=3)
    model.predict = orig_predict  # type: ignore[method-assign]

    P = np.stack(captured, axis=0)
    assert float(P[:, 0].std()) < 1e-9
    assert float(P[:, n_train // 3].std()) > 0.5
    assert float(P[:, -1].std()) > float(P[:, n_train // 3].std())
