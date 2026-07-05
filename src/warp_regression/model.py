"""Unified WarpModel facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from torch import Tensor

from .constants import DEFAULT_PATH_ANCHOR, PathAnchor
from .forecast import predict_forecast_realisations_torch
from .readouts.dual import fit_dual_sine_shared_warp, fit_dual_sine_shared_warp_nonlinear
from .readouts.log_trend_sine import fit_warp_model, forecast_future
from .readouts.parametric import WarpParametricModel
from .utilities.metrics import _r2_rmse


ReadoutKind = Literal["parametric", "dual_linear", "dual_mlp", "log_trend_sine"]


@dataclass
class WarpModelConfig:
    readout: ReadoutKind
    n_knots: int = 8
    fit_lambda: float = 0.5
    epochs: int = 1000
    lr: float = 0.03
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR
    path_mode: str = "identity"
    seed: int = 0
    mlp_hidden: int = 32
    mlp_layers: int = 2
    sr: int = 10
    A_init: float = 2.7
    C_init: float = 0.5


@dataclass
class FitResult:
    y_hat: np.ndarray
    r2: float
    rmse: float
    fit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastResult:
    preds: np.ndarray
    y_point: np.ndarray
    bands: Dict[str, np.ndarray]
    sigma_t: float
    sigma_y: float
    n_train: int
    n_future: int
    raw: Dict[str, Any] = field(default_factory=dict)


class WarpModel:
    """Unified warp regression model for synthetic, Lynx, and Bitcoin workflows."""

    def __init__(self, config: WarpModelConfig) -> None:
        self.config = config
        self._fit: Optional[Dict[str, Any]] = None
        self._parametric: Optional[WarpParametricModel] = None
        self._drivers: Dict[str, Any] = {}
        self._calendar: Dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self._fit is not None or self._parametric is not None

    def fit(
        self,
        y: np.ndarray,
        *,
        drivers: Optional[Dict[str, np.ndarray]] = None,
        t_idx: Optional[np.ndarray] = None,
        t_norm: Optional[np.ndarray] = None,
        sine_fit: Optional[Dict[str, Any]] = None,
        years: Optional[np.ndarray] = None,
    ) -> FitResult:
        cfg = self.config
        y = np.asarray(y, dtype=np.float64)
        drivers = drivers or {}

        if cfg.readout == "parametric":
            x = drivers.get("x")
            if x is None:
                raise ValueError("parametric readout requires drivers['x']")
            x_t = torch.tensor(x, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            model = WarpParametricModel(
                n=len(y),
                n_knots=cfg.n_knots,
                sr=cfg.sr,
                A_init=cfg.A_init,
                C_init=cfg.C_init,
                path_mode=cfg.path_mode,
            )
            model.fit(x_t, y_t, epochs=cfg.epochs, lr=cfg.lr, seed=cfg.seed, fit_lambda=cfg.fit_lambda)
            self._parametric = model
            with torch.no_grad():
                y_hat = model.predict(x_t).detach().cpu().numpy()
            r2, rmse = _r2_rmse(y, y_hat)
            self._fit = {"kind": "parametric", "model": model}
            return FitResult(y_hat=y_hat, r2=r2, rmse=rmse, fit=self._fit)

        if cfg.readout in ("dual_linear", "dual_mlp"):
            t = drivers.get("t")
            z1 = drivers.get("z1")
            z2 = drivers.get("z2")
            if t is None or z1 is None or z2 is None:
                raise ValueError("dual readout requires drivers t, z1, z2")
            sine_fit = sine_fit or {}
            if cfg.readout == "dual_linear":
                fit = fit_dual_sine_shared_warp(
                    y, t, years=years, n_knots=cfg.n_knots, epochs=cfg.epochs,
                    lr=cfg.lr, fit_lambda=cfg.fit_lambda, seed=cfg.seed, sine_fit=sine_fit,
                )
            else:
                fit = fit_dual_sine_shared_warp_nonlinear(
                    y, t, years=years, n_knots=cfg.n_knots, epochs=cfg.epochs,
                    hidden=cfg.mlp_hidden, n_hidden_layers=cfg.mlp_layers,
                    lr=cfg.lr, fit_lambda=cfg.fit_lambda, seed=cfg.seed, sine_fit=sine_fit,
                )
            self._fit = {**fit, "kind": cfg.readout}
            self._drivers = {"t": t, "z1": z1, "z2": z2}
            self._calendar = {"n_train": len(y)}
            y_hat = fit["y_hat_log"]
            return FitResult(y_hat=y_hat, r2=fit["r2_log"], rmse=fit["rmse_log"], fit=self._fit)

        if cfg.readout == "log_trend_sine":
            if t_idx is None or t_norm is None or sine_fit is None:
                raise ValueError("log_trend_sine requires t_idx, t_norm, sine_fit")
            fit = fit_warp_model(
                y, t_idx, t_norm, sine_fit,
                n_knots=cfg.n_knots, epochs=cfg.epochs, lr=cfg.lr,
                fit_lambda=cfg.fit_lambda, mlp_hidden=cfg.mlp_hidden,
                mlp_layers=cfg.mlp_layers, seed=cfg.seed,
            )
            self._fit = {**fit, "kind": "log_trend_sine"}
            self._drivers = {"z": np.asarray(sine_fit["z"], dtype=np.float64)[: len(y)]}
            self._calendar = {"t_idx": t_idx, "t_norm": t_norm, "sine_fit": sine_fit}
            return FitResult(y_hat=fit["y_hat"], r2=fit["r2"], rmse=fit["rmse"], fit=self._fit)

        raise ValueError(f"unknown readout: {cfg.readout}")

    def predict(self, drivers: Optional[Dict[str, np.ndarray]] = None, p: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("model not fitted")
        if self._parametric is not None:
            x = (drivers or self._drivers).get("x")
            if x is None:
                raise ValueError("drivers['x'] required")
            x_t = torch.tensor(x, dtype=torch.float32)
            p_t = torch.tensor(p, dtype=torch.float32) if p is not None else None
            with torch.no_grad():
                return self._parametric.predict(x_t, p_t).detach().cpu().numpy()
        return self._fit["y_hat"]  # type: ignore[index]

    def forecast(
        self,
        n_future: int,
        *,
        n_draws: int = 20,
        n_paths_ci: Optional[int] = None,
        seed: int = 0,
        noise_seed: int = 2,
        t_idx: Optional[np.ndarray] = None,
        t_norm: Optional[np.ndarray] = None,
        z_full: Optional[np.ndarray] = None,
        n_obs: Optional[int] = None,
        x_train: Optional[Tensor] = None,
    ) -> ForecastResult:
        if not self.is_fitted:
            raise RuntimeError("model not fitted")
        cfg = self.config

        if self._parametric is not None:
            fc = predict_forecast_realisations_torch(
                model=self._parametric,
                x_train=x_train,
                n_future=n_future,
                n_draws=n_draws,
                n_paths_ci=n_paths_ci,
                seed=seed,
                noise_seed=noise_seed,
                path_anchor=cfg.path_anchor,
            )
        elif cfg.readout == "log_trend_sine":
            if t_idx is None or t_norm is None or z_full is None or n_obs is None:
                raise ValueError("bitcoin forecast needs t_idx, t_norm, z_full, n_obs")
            fc = predict_forecast_realisations_torch(
                fit=self._fit,
                t_idx=t_idx,
                t_norm=t_norm,
                z_full=z_full,
                n_obs=n_obs,
                n_future=n_future,
                n_draws=n_draws,
                n_paths_ci=n_paths_ci,
                seed=seed,
                noise_seed=noise_seed,
            )
        elif cfg.readout in ("dual_linear", "dual_mlp"):
            from .forecast import forecast_lynx_holdout_paths

            z1 = self._drivers["z1"]
            z2 = self._drivers["z2"]
            n_train = len(self._fit["warp"]["p"])
            n_total = n_train + int(n_future)
            split = self._calendar.get("split") or {
                "n_train": n_train,
                "test_idx": np.arange(n_train, n_total, dtype=int),
                "train_idx": np.arange(n_train, dtype=int),
            }
            p_full = np.arange(n_total, dtype=np.float64)
            p_full[:n_train] = self._fit["warp"]["p"]
            fit_ext = {**self._fit, "warp": {**self._fit["warp"], "p": p_full, "n_knots": self.config.n_knots}}
            fc = forecast_lynx_holdout_paths(
                fit_ext,
                z1[:n_total],
                z2[:n_total],
                split,
                n_paths=n_draws,
                seed=seed,
                noise_seed=noise_seed,
            )
        else:
            raise ValueError(f"forecast not supported for {cfg.readout}")

        return ForecastResult(
            preds=fc["preds"],
            y_point=fc["y_point"],
            bands=fc["bands"],
            sigma_t=float(fc.get("sigma_t", self._fit.get("warp", {}).get("sigma_t", 0))),
            sigma_y=float(fc.get("sigma_y", self._fit.get("warp", {}).get("sigma_y", 0))),
            n_train=int(fc.get("n_train", fc.get("n_obs", n_obs or 0))),
            n_future=int(fc.get("n_future", n_future)),
            raw=fc,
        )
