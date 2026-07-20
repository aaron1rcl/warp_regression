from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils import (
    day_index,
    fetch_bitcoin_daily,
    fit_log_trend,
    fit_sine_peak_presize,
    log_price,
    normalized_time,
    sine_from_fit,
    split_bitcoin_holdout,
)
from warp_regression import WarpModel


def _build_z_full(t: np.ndarray, sine_fit: dict) -> np.ndarray:
    return sine_from_fit(t, sine_fit)


@pytest.mark.slow
def test_bitcoin_holdout_full(baselines):
    _run_bitcoin_holdout(baselines["bitcoin"])


def test_bitcoin_holdout_smoke(baselines):
    cfg = {**baselines["bitcoin"], "epochs": 500}
    corr, rmse = _run_bitcoin_holdout(cfg, return_metrics=True)
    assert corr > 0.3, f"smoke test corr too low: {corr:.3f}"
    assert rmse < 1.5, f"smoke test rmse too high: {rmse:.3f}"


def _run_bitcoin_holdout(cfg, corr_tol=None, rmse_tol=None, return_metrics=False):
    corr_tol = corr_tol if corr_tol is not None else cfg["test_corr_tol"]
    rmse_tol = rmse_tol if rmse_tol is not None else cfg["test_rmse_tol"]

    df = fetch_bitcoin_daily()
    dates = pd.to_datetime(df["date"])
    y = log_price(df["price_usd"].to_numpy())
    n = len(y)
    t_idx = day_index(n)
    t = normalized_time(n)
    split = split_bitcoin_holdout(n, test_days=365, dates=dates)
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    n_train = split["n_train"]

    y_tr = y[train_idx]
    t_idx_tr = t_idx[train_idx]
    t_tr = t[train_idx]
    dates_tr = dates[train_idx]

    _, _, _, trend, _ = fit_log_trend(y_tr, t_idx_tr)
    residual = y_tr - trend
    sine_fit = fit_sine_peak_presize(residual, y_tr, t_tr, dates_tr)
    z_full = _build_z_full(t, sine_fit)

    model = WarpModel.from_yaml(
        "bitcoin.yaml",
        overrides={
            "train": {
                "epochs": cfg["epochs"],
                "fit_lambda": cfg["fit_lambda"],
                "lr": 0.2,
                "seed": cfg["seed"],
            },
            "n_knots": cfg["n_knots"],
        },
    )
    model.fit(
        y_tr,
        covariates={"z": z_full[:n_train]},
        calendar={"t_idx": t_idx_tr, "t_norm": t_tr},
        sine_fit={**sine_fit, "z": z_full[:n_train]},
    )

    fc = model.forecast(
        split["n_test"],
        n_draws=20,
        seed=cfg["seed"],
        t_idx=t_idx,
        t_norm=t,
        z_full=z_full,
        n_obs=n_train,
        covariates_full={"z": z_full},
        calendar_full={"t_idx": t_idx, "t_norm": t},
        sine_fit=sine_fit,
    )
    y_hat = fc.y_point[test_idx]
    y_te = y[test_idx]
    corr = float(np.corrcoef(y_hat, y_te)[0, 1])
    rmse = float(np.sqrt(np.mean((y_hat - y_te) ** 2)))
    if return_metrics:
        return corr, rmse
    assert abs(corr - cfg["test_corr"]) <= corr_tol, f"corr={corr:.3f}"
    assert abs(rmse - cfg["test_rmse"]) <= rmse_tol, f"rmse={rmse:.3f}"
