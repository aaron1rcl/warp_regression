"""Unified WarpModel facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from torch import Tensor

from .constants import DEFAULT_PATH_ANCHOR, PathAnchor
from .forecast import (
    ForecastState,
    as_forecast_state,
    predict_forecast_realisations_torch,
)
from .prefit import LogTrendAnalysis, PrefitResult, analyze_log_trend, prefit, prefit_synthetic_path
from .readouts.dual import fit_dual_sine_shared_warp, fit_dual_sine_shared_warp_nonlinear
from .residual_smooth import ResidualSmoothFit, fit_residual_smooth
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
    residual_smooth_horizon: Optional[int] = None


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
        self._prefit: Optional[PrefitResult] = None
        self._parametric: Optional[WarpParametricModel] = None
        self._drivers: Dict[str, Any] = {}
        self._calendar: Dict[str, Any] = {}
        self._y: Optional[np.ndarray] = None

    @property
    def prefit_result(self) -> Optional[PrefitResult]:
        return self._prefit

    def prefit(
        self,
        y: np.ndarray,
        *,
        data: Optional[Dict[str, Any]] = None,
        t: Optional[np.ndarray] = None,
        years: Optional[np.ndarray] = None,
        t_idx: Optional[np.ndarray] = None,
        t_norm: Optional[np.ndarray] = None,
        dates: Optional[Any] = None,
        t_full: Optional[np.ndarray] = None,
        peak_idx: Optional[np.ndarray] = None,
        omega: Optional[float] = None,
        **kwargs: Any,
    ) -> PrefitResult:
        """Presize warp driver(s) before joint fit."""
        cfg = self.config
        y = np.asarray(y, dtype=np.float64)

        if cfg.readout == "parametric":
            if data is None:
                raise ValueError("parametric prefit requires data= from build_synthetic_dataset()")
            result = prefit_synthetic_path(
                data, len(y), sr=cfg.sr, path_anchor=cfg.path_anchor
            )
        elif cfg.readout in ("dual_linear", "dual_mlp"):
            if t is None:
                raise ValueError("dual readout prefit requires t")
            result = prefit(y, t, n_sines=2, years=years, **kwargs)
        elif cfg.readout == "log_trend_sine":
            t_use = t if t is not None else t_norm
            if t_use is None:
                raise ValueError("log_trend_sine prefit requires t (pass detrended residual as y)")
            if peak_idx is None:
                raise ValueError("log_trend_sine prefit requires peak_idx=")
            result = prefit(
                y,
                t_use,
                n_sines=1,
                peak_idx=peak_idx,
                omega=omega if omega is not None else 5.0,
                envelope_init=True,
                t_full=t_full,
                **kwargs,
            )
        else:
            raise ValueError(f"unknown readout: {cfg.readout}")

        self._prefit = result
        self._drivers = dict(result.drivers)
        if t_norm is not None:
            self._calendar["t_norm"] = t_norm
        if t_idx is not None:
            self._calendar["t_idx"] = t_idx
        return result

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
        self._y = y
        self._prepared = None
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
                path_anchor=cfg.path_anchor,
            )
            model.fit(x_t, y_t, epochs=cfg.epochs, lr=cfg.lr, seed=cfg.seed, fit_lambda=cfg.fit_lambda)
            self._parametric = model
            self._drivers = {"x": np.asarray(x, dtype=np.float64)}
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
                if self._prefit is not None:
                    t = self._prefit.drivers.get("t")
                    z1 = self._prefit.drivers.get("z1")
                    z2 = self._prefit.drivers.get("z2")
                if t is None or z1 is None or z2 is None:
                    raise ValueError("dual readout requires drivers t, z1, z2 (or call prefit() first)")
            sine_fit = sine_fit or (self._prefit.sine_fit if self._prefit else {})
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
            self._fit = {**fit, "kind": cfg.readout, "_y_log_train": y, "n_knots": cfg.n_knots}
            self._fit["warp"] = {**self._fit["warp"], "n_knots": cfg.n_knots}
            self._drivers = {"t": t, "z1": z1, "z2": z2}
            self._calendar = {"n_train": len(y)}
            y_hat = fit["y_hat_log"]
            return FitResult(y_hat=y_hat, r2=fit["r2_log"], rmse=fit["rmse_log"], fit=self._fit)

        if cfg.readout == "log_trend_sine":
            if sine_fit is None and self._prefit is not None:
                sine_fit = self._prefit.sine_fit
            if t_idx is None or t_norm is None or sine_fit is None:
                raise ValueError("log_trend_sine requires t_idx, t_norm, sine_fit (or call prefit() first)")
            fit = fit_warp_model(
                y, t_idx, t_norm, sine_fit,
                n_knots=cfg.n_knots, epochs=cfg.epochs, lr=cfg.lr,
                fit_lambda=cfg.fit_lambda, mlp_hidden=cfg.mlp_hidden,
                mlp_layers=cfg.mlp_layers, seed=cfg.seed,
            )
            self._fit = {**fit, "kind": "log_trend_sine", "_y_train": y}
            if cfg.residual_smooth_horizon is not None:
                res_fit = fit_residual_smooth(y - fit["y_hat"], horizon=cfg.residual_smooth_horizon)
                self._fit["residual_smooth"] = res_fit
            self._drivers = {"z": np.asarray(sine_fit["z"], dtype=np.float64)[: len(y)]}
            self._calendar = {"t_idx": t_idx, "t_norm": t_norm, "sine_fit": sine_fit}
            return FitResult(y_hat=fit["y_hat"], r2=fit["r2"], rmse=fit["rmse"], fit=self._fit)

        raise ValueError(f"unknown readout: {cfg.readout}")

    def as_forecast_state(
        self,
        *,
        z1_full: Optional[np.ndarray] = None,
        z2_full: Optional[np.ndarray] = None,
        x_full: Optional[np.ndarray] = None,
        t_idx_full: Optional[np.ndarray] = None,
        t_norm_full: Optional[np.ndarray] = None,
        z_full: Optional[np.ndarray] = None,
        n_obs: Optional[int] = None,
        sine_fit: Optional[Dict[str, Any]] = None,
    ) -> ForecastState:
        if not self.is_fitted:
            raise RuntimeError("model not fitted")
        if self._parametric is not None:
            return as_forecast_state(
                self._parametric,
                x_full=x_full if x_full is not None else self._drivers.get("x"),
                sine_fit=sine_fit,
            )
        cfg = self.config
        if cfg.readout in ("dual_linear", "dual_mlp"):
            z1 = z1_full if z1_full is not None else self._drivers.get("z1")
            z2 = z2_full if z2_full is not None else self._drivers.get("z2")
            if z1 is None or z2 is None:
                raise ValueError("dual prepare/forecast needs z1_full/z2_full")
            n_tr = len(self._y) if self._y is not None else len(z1)
            return as_forecast_state(self._fit, z1_full=z1, z2_full=z2, n_train=n_tr)
        if cfg.readout == "log_trend_sine":
            if t_idx_full is None or t_norm_full is None or z_full is None:
                raise ValueError("log_trend_sine prepare/forecast needs t_idx_full, t_norm_full, z_full")
            n_tr = len(self._y) if self._y is not None else int(n_obs or 0)
            return as_forecast_state(
                self._fit,
                t_idx_full=t_idx_full,
                t_norm_full=t_norm_full,
                z_full=z_full,
                n_obs=n_obs if n_obs is not None else n_tr,
            )
        raise ValueError(f"as_forecast_state not supported for {cfg.readout}")

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
        n_future: Optional[int] = None,
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
        z1_full: Optional[np.ndarray] = None,
        z2_full: Optional[np.ndarray] = None,
        sine_fit: Optional[Dict[str, Any]] = None,
        residual_horizon: Optional[int] = None,
    ) -> ForecastResult:
        if not self.is_fitted:
            raise RuntimeError("model not fitted")
        if n_future is None:
            raise ValueError("n_future required")
        cfg = self.config
        res_h = residual_horizon if residual_horizon is not None else cfg.residual_smooth_horizon

        if self._parametric is not None:
            fc = predict_forecast_realisations_torch(
                model=self._parametric,
                x_train=x_train,
                n_future=n_future,
                n_draws=n_draws,
                n_paths_ci=n_paths_ci,
                seed=seed,
                noise_seed=noise_seed,
                sine_fit=sine_fit if sine_fit is not None else self._prefit.sine_fit if self._prefit else None,
                y_train=self._y,
                residual_horizon=res_h,
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
                n_future=n_future or 0,
                n_draws=n_draws,
                n_paths_ci=n_paths_ci,
                seed=seed,
                noise_seed=noise_seed,
                y_train=self._y,
                residual_horizon=res_h,
            )
        elif cfg.readout in ("dual_linear", "dual_mlp"):
            from .forecast import forecast_lynx_holdout_paths

            z1 = z1_full if z1_full is not None else self._drivers["z1"]
            z2 = z2_full if z2_full is not None else self._drivers["z2"]
            n_train = len(self._fit["warp"]["p"])
            n_fut = int(n_future or 0)
            n_total = n_train + n_fut
            split = self._calendar.get("split") or {
                "n_train": n_train,
                "test_idx": np.arange(n_train, n_total, dtype=int),
                "train_idx": np.arange(n_train, dtype=int),
            }
            fc = forecast_lynx_holdout_paths(
                self._fit,
                z1[:n_total],
                z2[:n_total],
                split,
                n_paths=n_draws,
                seed=seed,
                noise_seed=noise_seed,
                y_train=self._y,
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
            n_future=int(fc.get("n_future", n_future or 0)),
            raw=fc,
        )
