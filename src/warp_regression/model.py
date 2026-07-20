"""YAML-configured WarpModel composing WarpRegression blocks + observation model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import Tensor

from .block import CovariateKind, WarpPath, WarpRegression
from .core.path import DEFAULT_PATH_ANCHOR, PathAnchor
from .core.training import DualLossValues, _lambda_schedule, compute_dual_loss
from .forecast import ForecastState, forecast_from_state
from .observation import ObservationModel
from .utilities.metrics import _r2_rmse


def _repo_configs_dir() -> Path:
    # examples/models/*.yaml (repo root / examples / models)
    return Path(__file__).resolve().parents[2] / "examples" / "models"


def _block_covariate_kind(bspec: Dict[str, Any]) -> CovariateKind:
    """Read ``covariate_kind`` from a YAML block; accept legacy ``sample`` key."""
    kind = bspec.get("covariate_kind", bspec.get("sample", "array"))
    return kind  # type: ignore[return-value]


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


class WarpModel(nn.Module):
    """Composed warp regression problem: path groups, blocks, observation model, σ_y."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        n: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = dict(config or {})
        self.n = int(n) if n is not None else 0
        self.path_anchor: PathAnchor = self.config.get("path_anchor", DEFAULT_PATH_ANCHOR)
        self.path_mode: str = self.config.get("path_mode", "identity")
        self.n_knots: int = int(self.config.get("n_knots", 8))
        train = self.config.get("train", {})
        self.epochs = int(train.get("epochs", 1000))
        self.lr = float(train.get("lr", 0.03))
        self.fit_lambda = train.get("fit_lambda", 0.5)
        self.seed = int(train.get("seed", 0))

        self.path_groups = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        self.observation: Optional[ObservationModel] = None
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self._covariates: Dict[str, np.ndarray] = {}
        self._calendar: Dict[str, np.ndarray] = {}
        self._sine_fit: Optional[Dict[str, Any]] = None
        self._y: Optional[np.ndarray] = None
        self._y_hat: Optional[np.ndarray] = None
        self._fit_meta: Dict[str, Any] = {}
        self._built = False

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        *,
        n: Optional[int] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "WarpModel":
        path = Path(path)
        if not path.is_file():
            alt = _repo_configs_dir() / path.name
            if alt.is_file():
                path = alt
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if overrides:
            cfg = _deep_merge(cfg, overrides)
        return cls(cfg, n=n)

    @classmethod
    def from_dict(cls, config: Dict[str, Any], *, n: Optional[int] = None) -> "WarpModel":
        return cls(config, n=n)

    def build(self, n: int, *, B_init: Optional[Dict[str, Any]] = None) -> None:
        """Instantiate path groups, blocks, and observation model for length ``n``."""
        self.n = int(n)
        self.path_groups = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        pg_cfg = self.config.get("path_groups") or {"shared": {}}
        B_init = B_init or {}
        for name, spec in pg_cfg.items():
            spec = spec or {}
            knots = int(spec.get("n_knots", self.n_knots))
            b0 = B_init.get(name, B_init.get("B"))
            self.path_groups[name] = WarpPath(
                self.n,
                knots,
                path_mode=self.path_mode,
                path_anchor=self.path_anchor,
                B_init=b0,
            )
        for bspec in self.config.get("blocks", []):
            bid = str(bspec["id"])
            pg_name = str(bspec.get("path_group", "shared"))
            if pg_name not in self.path_groups:
                raise ValueError(f"block {bid} references unknown path_group {pg_name}")
            self.blocks[bid] = WarpRegression(
                self.path_groups[pg_name],
                covariate_kind=_block_covariate_kind(bspec),
                name=str(bspec.get("input", bid)),
            )
        obs_cfg = self.config.get("observation", {})
        terms = obs_cfg.get("terms", [])
        self.observation = ObservationModel(terms)
        self._built = True

    @property
    def primary_path(self) -> WarpPath:
        if not self.path_groups:
            raise RuntimeError("model not built")
        return next(iter(self.path_groups.values()))

    @property
    def sigma_y(self) -> Tensor:
        return torch.exp(self.log_sigma)

    @property
    def sigma_t(self) -> Tensor:
        return self.primary_path.sigma_t

    @property
    def is_fitted(self) -> bool:
        return self._y_hat is not None

    def _resolve_input_name(self, block: WarpRegression, bid: str) -> str:
        return block.name if block.name else bid

    def _warp_all(
        self,
        covariates: Dict[str, Tensor],
        p: Tensor,
        *,
        sine_fit: Optional[Dict[str, Any]] = None,
        n_norm: Optional[int] = None,
        force_kind: Optional[CovariateKind] = None,
    ) -> Dict[str, Tensor]:
        assert self.observation is not None
        warped: Dict[str, Tensor] = {}
        for bid, block in self.blocks.items():
            key = self._resolve_input_name(block, bid)
            x = covariates.get(key, covariates.get(bid))
            kind = force_kind if force_kind is not None else block.covariate_kind
            if x is None and kind == "array":
                raise ValueError(f"missing covariate '{key}' for block '{bid}'")
            if x is None:
                x = torch.zeros(int(p.shape[0]), dtype=torch.float32)
            warped[key] = block.warp(
                x, p, sine_fit=sine_fit, n_norm=n_norm, covariate_kind=kind
            )
            warped[bid] = warped[key]
        return warped

    def predict_torch(
        self,
        covariates: Dict[str, Tensor],
        calendar: Dict[str, Tensor],
        p: Optional[Tensor] = None,
        *,
        B: Optional[Tensor] = None,
        sine_fit: Optional[Dict[str, Any]] = None,
        n_norm: Optional[int] = None,
        y_for_ls: Optional[Tensor] = None,
        force_kind: Optional[CovariateKind] = None,
    ) -> tuple[Tensor, Tensor]:
        assert self.observation is not None
        path = self.primary_path
        if p is None:
            # Path length follows the model window (len(y)), not covariate length.
            # Covariates may be longer so expansion can read future driver values.
            n_path = int(self.n) if self.n else (
                int(y_for_ls.shape[0]) if y_for_ls is not None
                else int(next(iter(covariates.values())).shape[0])
            )
            p = path.path(B=B, n=n_path)
        warped = self._warp_all(
            covariates, p, sine_fit=sine_fit, n_norm=n_norm, force_kind=force_kind
        )
        if y_for_ls is not None:
            self.observation.closed_form_step(warped, y_for_ls)
        y_hat = self.observation(warped, calendar)
        return y_hat, p

    def _prepare_fit_inputs(
        self,
        y: np.ndarray,
        covariates: Optional[Dict[str, np.ndarray]],
        calendar: Optional[Dict[str, np.ndarray]],
        t_idx: Optional[np.ndarray],
        t_norm: Optional[np.ndarray],
        sine_fit: Optional[Dict[str, Any]],
        B_init: Optional[Union[Tensor, np.ndarray, Dict[str, Any]]],
    ) -> tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
        """Normalize arrays, build the model, and init observation from sine/log-trend."""
        y = np.asarray(y, dtype=np.float64)
        covariates = {k: np.asarray(v, dtype=np.float64) for k, v in (covariates or {}).items()}
        calendar = {k: np.asarray(v, dtype=np.float64) for k, v in (calendar or {}).items()}
        if t_idx is not None:
            calendar["t_idx"] = np.asarray(t_idx, dtype=np.float64)
        if t_norm is not None:
            calendar["t_norm"] = np.asarray(t_norm, dtype=np.float64)
        if sine_fit is not None and "z" not in covariates and "z" in sine_fit:
            covariates["z"] = np.asarray(sine_fit["z"], dtype=np.float64)[: len(y)]
        n = len(y)

        b_init_map: Dict[str, Any] = {}
        if isinstance(B_init, dict):
            b_init_map = B_init
        elif B_init is not None:
            b_init_map = {"B": B_init}

        if not self._built or self.n != n:
            self.build(n, B_init=b_init_map)
        elif b_init_map:
            b0 = b_init_map.get("shared", b_init_map.get("B"))
            if b0 is not None:
                with torch.no_grad():
                    self.primary_path.B.copy_(torch.as_tensor(b0, dtype=torch.float32).reshape(-1))

        assert self.observation is not None
        if sine_fit is not None and any(t["kind"] == "log_trend" for t in self.observation.term_specs):
            from .prefit import analyze_log_trend

            t_idx_arr = calendar.get("t_idx")
            if t_idx_arr is not None:
                prep = analyze_log_trend(y, t_idx_arr)
                self.observation.init_log_trend_from_fit(
                    prep.B, prep.C, prep.z, gamma=float(prep.gamma)
                )
            self.observation.init_envelope_from_sine(
                float(sine_fit.get("f_init", 0.1)),
                a_floor_init=float(
                    sine_fit.get("a_floor_init", sine_fit.get("a_inf_init", 1.0))
                ),
                kappa_init=float(
                    sine_fit.get("kappa_init", sine_fit.get("lam_init", 0.0))
                ),
            )

        self._y = y
        self._covariates = covariates
        self._calendar = calendar
        self._sine_fit = sine_fit
        return y, covariates, calendar, n

    def _calendar_n_norm(self, calendar: Dict[str, np.ndarray], sine_fit: Optional[Dict[str, Any]], n: int) -> int:
        """Reference length for analytic sine mid-bin calendar ``t = (p+½)/n``."""
        n_norm = int(sine_fit.get("n_calendar", n)) if sine_fit else n
        if calendar.get("t_norm") is not None and len(calendar["t_norm"]):
            t0 = float(calendar["t_norm"][0])
            if t0 > 0:
                n_norm = int(round(0.5 / t0))
        return n_norm

    def _run_dual_training(
        self,
        y: np.ndarray,
        covariates: Dict[str, np.ndarray],
        calendar: Dict[str, np.ndarray],
        sine_fit: Optional[Dict[str, Any]],
        n: int,
        epochs: int,
        lr: float,
        fit_lambda: Any,
        seed: int,
    ) -> tuple[Dict[str, Any], List[DualLossValues], int]:
        """Adam loop on dual loss; returns best checkpoint, history, and n_norm."""
        torch.manual_seed(seed)
        cov_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in covariates.items()}
        cal_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in calendar.items()}
        y_t = torch.tensor(y, dtype=torch.float32)
        n_norm = self._calendar_n_norm(calendar, sine_fit, n)

        path = self.primary_path
        assert self.observation is not None
        params: List[Tensor] = [path.B, self.log_sigma, path.log_sigma_t]
        seen = {id(path)}
        for pg in self.path_groups.values():
            if id(pg) not in seen:
                params.extend([pg.B, pg.log_sigma_t])
                seen.add(id(pg))
        params.extend(self.observation.extra_params())
        opt = torch.optim.Adam(params, lr=lr)

        history: List[DualLossValues] = []
        best: Dict[str, Any] = {}
        best_loss = float("inf")
        kinds = {t["kind"] for t in self.observation.term_specs}
        use_ls = bool(kinds & {"linear", "affine", "linear_multi"})

        for ep in range(epochs):
            opt.zero_grad()
            lam = _lambda_schedule(ep, epochs, float(fit_lambda) if fit_lambda is not None else None)
            y_hat, p = self.predict_torch(
                cov_t,
                cal_t,
                B=path.B,
                sine_fit=sine_fit,
                n_norm=n_norm,
                y_for_ls=y_t if use_ls else None,
            )
            loss, vals = compute_dual_loss(
                y_t,
                y_hat,
                p,
                self.log_sigma,
                path.log_sigma_t,
                path.n_knots,
                lam,
                path_mode=self.path_mode,
            )
            for pg in self.path_groups.values():
                if pg is path:
                    continue
                loss = loss + (1.0 - lam) * pg.terror_nll(pg.path(n=n))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            history.append(vals)
            if float(loss.detach()) < best_loss:
                best_loss = float(loss.detach())
                with torch.no_grad():
                    y_hat_b, p_b = self.predict_torch(
                        cov_t,
                        cal_t,
                        B=path.B,
                        sine_fit=sine_fit,
                        n_norm=n_norm,
                        y_for_ls=y_t if use_ls else None,
                    )
                best = {
                    "B": path.B.detach().cpu().clone(),
                    "p": p_b.detach().cpu().numpy(),
                    "y_hat": y_hat_b.detach().cpu().numpy(),
                    "log_sigma": self.log_sigma.detach().cpu().clone(),
                    "log_sigma_t": path.log_sigma_t.detach().cpu().clone(),
                    **vals.__dict__,
                }

        with torch.no_grad():
            path.B.copy_(best["B"].to(path.B.device))
            self.log_sigma.copy_(best["log_sigma"].to(self.log_sigma.device))
            path.log_sigma_t.copy_(best["log_sigma_t"].to(path.log_sigma_t.device))
            y_hat_f, p_f = self.predict_torch(
                cov_t,
                cal_t,
                sine_fit=sine_fit,
                n_norm=n_norm,
                y_for_ls=y_t if use_ls else None,
            )
            best["y_hat_final"] = y_hat_f.detach().cpu().numpy()
            best["p_final"] = p_f.detach().cpu().numpy()
            best["kinds"] = kinds
            best["use_ls"] = use_ls

        return best, history, n_norm

    def _pack_fit_meta(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        p_np: np.ndarray,
        best: Dict[str, Any],
        history: List[DualLossValues],
        n_norm: int,
        sine_fit: Optional[Dict[str, Any]],
    ) -> FitResult:
        assert self.observation is not None
        path = self.primary_path
        n = len(y)
        r2, rmse = _r2_rmse(y, y_hat)
        self._y_hat = y_hat
        kinds = best["kinds"]
        obs_kind = next(iter(kinds)) if len(kinds) == 1 else "composed"
        if "mlp_on_warped" in kinds:
            kind = "dual_mlp"
        elif kinds & {"linear", "linear_multi", "affine"}:
            multi = any(
                t["kind"] in ("linear", "linear_multi", "affine") and len(t.get("inputs", ["x"])) > 1
                for t in self.observation.term_specs
            )
            kind = "dual_linear" if multi else "affine"
        elif "log_trend" in kinds or "envelope_sine" in kinds:
            kind = "log_trend_sine"
        else:
            kind = obs_kind

        self._fit_meta = {
            "kind": kind,
            "history": history,
            "n_calendar": n_norm,
            "warp": {
                "p": p_np,
                "B": best["B"].numpy(),
                "n_knots": path.n_knots,
                "path_anchor": self.path_anchor,
                "obj_err": best["obj_err"],
                "obj_time": best["obj_time"],
                "sigma_y": best["sigma_y"],
                "sigma_t": best["sigma_t"],
                "max_abs_offset": float(np.max(np.abs(p_np - np.arange(n)))),
            },
            "observation": self.observation,
            "sine_fit": sine_fit,
            "n_knots": path.n_knots,
            "_y_train": y,
        }
        return FitResult(y_hat=y_hat, r2=r2, rmse=rmse, fit=self._fit_meta)

    def fit(
        self,
        y: np.ndarray,
        covariates: Optional[Dict[str, np.ndarray]] = None,
        *,
        calendar: Optional[Dict[str, np.ndarray]] = None,
        t_idx: Optional[np.ndarray] = None,
        t_norm: Optional[np.ndarray] = None,
        sine_fit: Optional[Dict[str, Any]] = None,
        B_init: Optional[Union[Tensor, np.ndarray, Dict[str, Any]]] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        fit_lambda: Optional[float] = None,
        seed: Optional[int] = None,
        years: Optional[np.ndarray] = None,
    ) -> FitResult:
        del years  # prefit responsibility
        y, covariates, calendar, n = self._prepare_fit_inputs(
            y, covariates, calendar, t_idx, t_norm, sine_fit, B_init
        )
        epochs = int(epochs if epochs is not None else self.epochs)
        lr = float(lr if lr is not None else self.lr)
        fit_lambda = self.fit_lambda if fit_lambda is None else fit_lambda
        seed = int(seed if seed is not None else self.seed)

        best, history, n_norm = self._run_dual_training(
            y, covariates, calendar, sine_fit, n, epochs, lr, fit_lambda, seed
        )
        return self._pack_fit_meta(
            y,
            best["y_hat_final"],
            best["p_final"],
            best,
            history,
            n_norm,
            sine_fit,
        )

    @property
    def _fit(self) -> Optional[Dict[str, Any]]:
        return self._fit_meta if self._fit_meta else None

    @_fit.setter
    def _fit(self, value: Dict[str, Any]) -> None:
        self._fit_meta = value

    def path_B(self) -> Tensor:
        return self.primary_path.B.detach().clone()

    def _forecast_force_kind(
        self, sine_fit: Optional[Dict[str, Any]]
    ) -> Optional[CovariateKind]:
        """Override covariate kind for horizon expansion (analytic/prefit sine)."""
        if sine_fit is None:
            return None
        kind_meta = self._fit_meta.get("kind")
        if kind_meta == "log_trend_sine":
            return "analytic_sine"
        if (
            kind_meta == "affine"
            and sine_fit.get("dt") is not None
            and sine_fit.get("t0") is not None
        ):
            return "prefit_sine"
        if any(b.covariate_kind in ("analytic_sine", "prefit_sine") for b in self.blocks.values()):
            return None  # already configured on the block
        return None

    def as_forecast_state(
        self,
        *,
        covariates_full: Optional[Dict[str, np.ndarray]] = None,
        calendar_full: Optional[Dict[str, np.ndarray]] = None,
        sine_fit: Optional[Dict[str, Any]] = None,
        n_train: Optional[int] = None,
        z1_full: Optional[np.ndarray] = None,
        z2_full: Optional[np.ndarray] = None,
        x_full: Optional[np.ndarray] = None,
        t_idx_full: Optional[np.ndarray] = None,
        t_norm_full: Optional[np.ndarray] = None,
        z_full: Optional[np.ndarray] = None,
        n_obs: Optional[int] = None,
    ) -> ForecastState:
        if not self.is_fitted:
            raise RuntimeError("model not fitted")
        cov = dict(covariates_full or {})
        if x_full is not None:
            cov.setdefault("x", x_full)
        if z1_full is not None:
            cov.setdefault("z1", z1_full)
        if z2_full is not None:
            cov.setdefault("z2", z2_full)
        if z_full is not None:
            cov.setdefault("z", z_full)
        for k, v in self._covariates.items():
            cov.setdefault(k, v)
        cal = dict(calendar_full or {})
        if t_idx_full is not None:
            cal["t_idx"] = t_idx_full
        if t_norm_full is not None:
            cal["t_norm"] = t_norm_full
        for k, v in self._calendar.items():
            cal.setdefault(k, v)
        sf = sine_fit if sine_fit is not None else self._sine_fit
        n_tr = int(n_obs if n_obs is not None else (n_train if n_train is not None else self.n))
        return self._build_forecast_state(cov, cal, n_tr, sf)

    def _build_forecast_state(
        self,
        cov_full: Dict[str, np.ndarray],
        cal_full: Dict[str, np.ndarray],
        n_train: int,
        sine_fit: Optional[Dict[str, Any]],
    ) -> ForecastState:
        assert self.observation is not None
        path = self.primary_path
        with torch.no_grad():
            p = path.path(n=n_train).detach().cpu().numpy()
            sigma_y = float(self.sigma_y.detach())
            sigma_t = float(self.sigma_t.detach())
            B = path.B.detach().cpu().numpy().copy()
        n_norm = int(self._fit_meta.get("n_calendar", n_train))
        block_kind = (
            next(iter(self.blocks.values())).covariate_kind if self.blocks else "array"
        )

        def predict_train(p_tr: np.ndarray) -> np.ndarray:
            return self._predict_numpy(cov_full, cal_full, p_tr, n_train, sine_fit, n_norm)

        def extend_predict(p_tr: np.ndarray, n_future: int):
            n_total = n_train + int(n_future)

            def predict_full(p_full: np.ndarray) -> np.ndarray:
                return self._predict_numpy(cov_full, cal_full, p_full, n_total, sine_fit, n_norm)

            return predict_full

        return ForecastState(
            p=np.asarray(p, dtype=np.float64),
            sigma_t=sigma_t,
            sigma_y=sigma_y,
            n_knots=path.n_knots,
            n_train=n_train,
            path_mode=self.path_mode,
            path_anchor=self.path_anchor,
            predict_train=predict_train,
            extend_predict=extend_predict,
            B=B,
            meta={
                "kind": self._fit_meta.get("kind", "warp_model"),
                "sine_fit": sine_fit,
                "covariate_kind": block_kind,
            },
        )

    def _predict_numpy(
        self,
        cov_full: Dict[str, np.ndarray],
        cal_full: Dict[str, np.ndarray],
        p: np.ndarray,
        n: int,
        sine_fit: Optional[Dict[str, Any]],
        n_norm: int,
    ) -> np.ndarray:
        """Numpy predict for forecast paths; may force analytic/prefit sine for expansion."""
        assert self.observation is not None
        p_t = torch.tensor(p[:n], dtype=torch.float32)
        cov_t: Dict[str, Tensor] = {}
        for bid, block in self.blocks.items():
            key = self._resolve_input_name(block, bid)
            arr = cov_full.get(key, cov_full.get(bid))
            if arr is None:
                arr = np.zeros(n, dtype=np.float64)
            cov_t[key] = torch.tensor(np.asarray(arr[:n], dtype=np.float64), dtype=torch.float32)
        cal_t = {
            k: torch.tensor(np.asarray(v[:n], dtype=np.float64), dtype=torch.float32)
            for k, v in cal_full.items()
            if len(v) >= n
        }
        force_kind = self._forecast_force_kind(sine_fit)
        use_sine = force_kind is not None or (
            sine_fit is not None
            and any(b.covariate_kind in ("analytic_sine", "prefit_sine") for b in self.blocks.values())
        )
        with torch.no_grad():
            warped = self._warp_all(
                cov_t,
                p_t,
                sine_fit=sine_fit if use_sine else None,
                n_norm=n_norm,
                force_kind=force_kind,
            )
            y_hat = self.observation(warped, cal_t)
        return y_hat.detach().cpu().numpy()

    def forecast(
        self,
        n_future: int,
        *,
        n_draws: int = 20,
        n_paths_ci: Optional[int] = None,
        seed: int = 0,
        noise_seed: int = 2,
        ci: float = 0.95,
        covariates_full: Optional[Dict[str, np.ndarray]] = None,
        calendar_full: Optional[Dict[str, np.ndarray]] = None,
        sine_fit: Optional[Dict[str, Any]] = None,
        x_train: Optional[Tensor] = None,
        z1_full: Optional[np.ndarray] = None,
        z2_full: Optional[np.ndarray] = None,
        t_idx: Optional[np.ndarray] = None,
        t_norm: Optional[np.ndarray] = None,
        z_full: Optional[np.ndarray] = None,
        n_obs: Optional[int] = None,
        residual_horizon: Optional[int] = None,
    ) -> ForecastResult:
        del residual_horizon  # applied by caller via residual_smooth helpers
        cov = dict(covariates_full or {})
        if x_train is not None:
            cov.setdefault(
                "x",
                x_train.detach().cpu().numpy() if isinstance(x_train, Tensor) else x_train,
            )
        if z1_full is not None:
            cov["z1"] = z1_full
        if z2_full is not None:
            cov["z2"] = z2_full
        if z_full is not None:
            cov["z"] = z_full
        for key in ("z1", "z2", "z", "x"):
            if key in self._covariates and key not in cov:
                cov[key] = self._covariates[key]
        cal = dict(calendar_full or {})
        if t_idx is not None:
            cal["t_idx"] = t_idx
        if t_norm is not None:
            cal["t_norm"] = t_norm

        n_tr = int(n_obs if n_obs is not None else self.n)
        n_total = n_tr + int(n_future)
        sf = sine_fit if sine_fit is not None else self._sine_fit
        cal_use = cal if cal else self._calendar
        if any(len(v) >= n_total for v in cov.values()) or cov:
            state = self._build_forecast_state(cov, cal_use, n_tr, sf)
        else:
            state = self.as_forecast_state(
                covariates_full=cov or None,
                calendar_full=cal or None,
                sine_fit=sf,
                n_obs=n_obs,
            )

        fc = forecast_from_state(
            state,
            n_future=int(n_future),
            n_draws=n_draws,
            n_paths_ci=n_paths_ci,
            seed=seed,
            noise_seed=noise_seed,
            ci=ci,
        )
        return ForecastResult(
            preds=fc["preds"],
            y_point=fc["y_point"],
            bands=fc["bands"],
            sigma_t=float(fc["sigma_t"]),
            sigma_y=float(fc["sigma_y"]),
            n_train=int(fc["n_train"]),
            n_future=int(fc["n_future"]),
            raw=fc,
        )


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
