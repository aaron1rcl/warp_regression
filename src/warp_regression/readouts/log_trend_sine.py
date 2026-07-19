"""Log-trend + warped sine readout (Bitcoin)."""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from ..constants import BITCOIN_CSV, DATA_ROOT, DEFAULT_PATH_ANCHOR, PathAnchor
from ..core.path import path_from_B_torch, point_forecast_path, stored_path_offset_numpy
from ..core.training import WarpTrainConfig, train_dual_warp
from ..core.warp import soft_warp_numpy, soft_warp_sine_numpy, soft_warp_torch
from ..drivers.sine import align_sine_to_macro_peaks, refine_param_endpoint
from ..forecast import (
    build_forecast_bands,
    per_index_rw_sigma,
    sample_warp_paths_future,
    sample_warp_paths_future_knots,
)
from ..utilities.metrics import _r2_rmse
from ..plotting import format_date_axis

DATA_PATH = BITCOIN_CSV

N_CYCLE_PEAKS = 5
CYCLE_PEAK_YEARS = (2011, 2013, 2017, 2021, 2024)
CYCLE_PEAK_HALF_WINDOW_DAYS = 540
N_PATHS = 400
N_SINGLE_PATHS = 20
FORECAST_DAYS = 365
FORECAST_5YR_DAYS = int(5 * 365.25)
TEST_DAYS = 365
EPOCHS = 24000
N_KNOTS = 8
MLP_HIDDEN = 32
MLP_LAYERS = 2
FIT_LAMBDA = 0.5
WARP_LR = 0.2


def fetch_bitcoin_daily(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"])
        if len(df) > 100:
            return df

    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=json&sampled=false"
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    rows = [
        {
            "date": datetime.fromtimestamp(pt["x"], tz=timezone.utc).date().isoformat(),
            "price_usd": float(pt["y"]),
        }
        for pt in payload["values"]
        if pt.get("y") and float(pt["y"]) > 0.0
    ]
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def log_price(price: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(price, 1e-12))


def price_from_log(y: np.ndarray) -> np.ndarray:
    return np.exp(y)


def normalized_time(n: int) -> np.ndarray:
    """Unit-interval calendar time for the sine driver: (i+½)/n, i=0…n−1."""
    return (np.arange(n, dtype=np.float64) + 0.5) / n


def split_bitcoin_holdout(
    n: int,
    test_days: int = TEST_DAYS,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, Any]:
    """Hold out the last `test_days` observations for out-of-sample forecast."""
    n_test = int(min(max(test_days, 1), n - 30))
    n_train = n - n_test
    train_idx = np.arange(n_train, dtype=int)
    test_idx = np.arange(n_train, n, dtype=int)
    out: Dict[str, Any] = {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_train": n_train,
        "n_test": n_test,
        "test_days": n_test,
    }
    if dates is not None:
        out["train_end_date"] = dates[n_train - 1]
        out["test_start_date"] = dates[n_train]
        out["dates_train"] = dates[train_idx]
        out["dates_test"] = dates[test_idx]
    return out


def day_index(n: int, start: int = 1) -> np.ndarray:
    """Day counter for the log trend: start, start+1, …, start+n−1."""
    return np.arange(start, start + n, dtype=np.float64)


def log_day_index(n: int, start: int = 1) -> np.ndarray:
    """Natural log of day index; start=1 keeps log(t) finite."""
    return np.log(day_index(n, start))


def log_shifted_day_index(t_idx: np.ndarray, z: float) -> np.ndarray:
    """log(t_idx − z) with t_idx ≥ 1 and z < 1 so the argument stays positive."""
    return np.log(t_idx - z)


def _sine(
    t: np.ndarray,
    omega: float,
    phase: float,
    time_scale: float = 1.0,
    t_shift: float = 0.0,
) -> np.ndarray:
    return np.sin(2.0 * np.pi * omega * time_scale * t + phase + t_shift)


def sine_from_fit(t: np.ndarray, sine_fit: Dict[str, Any]) -> np.ndarray:
    return _sine(
        t,
        sine_fit["omega"],
        sine_fit["phase"],
        time_scale=sine_fit.get("time_scale", 1.0),
        t_shift=sine_fit.get("t_shift", 0.0),
    )


_sine_from_fit = sine_from_fit


def _peak_center(year: int) -> pd.Timestamp:
    if year == 2011:
        return pd.Timestamp("2011-06-01")
    if year == 2013:
        return pd.Timestamp("2013-12-01")
    return pd.Timestamp(f"{year}-06-15")


def detect_btc_cycle_peaks(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    cycle_years: Tuple[int, ...] = CYCLE_PEAK_YEARS,
    half_window_days: int = CYCLE_PEAK_HALF_WINDOW_DAYS,
) -> np.ndarray:
    peak_idx: List[int] = []
    for year in cycle_years:
        center = _peak_center(year)
        lo = center - pd.Timedelta(days=half_window_days)
        hi = center + pd.Timedelta(days=half_window_days)
        mask = (dates >= lo) & (dates <= hi)
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        peak_idx.append(int(idx[np.argmax(y[idx])]))
    if len(peak_idx) < 2:
        return np.linspace(0, len(y) - 1, N_CYCLE_PEAKS, dtype=int)
    return np.array(sorted(set(peak_idx)), dtype=int)


def omega_from_peak_times(t: np.ndarray, peak_idx: np.ndarray) -> float:
    t_peaks = t[peak_idx]
    span = float(t_peaks[-1] - t_peaks[0])
    if span < 1e-6:
        return 1.0
    return float(len(peak_idx) - 1) / span


def _t_z_candidates() -> np.ndarray:
    """Grid for presizing z in log(t_idx − z); z < 1 keeps t_idx − z > 0."""
    near_one = 1.0 - np.exp(np.linspace(-8.0, -0.01, 40))
    negative = -np.exp(np.linspace(0.0, np.log(2000.0), 40))
    return np.concatenate([near_one, negative])


def fit_log_trend(
    y: np.ndarray, t_idx: np.ndarray, z_grid: Optional[np.ndarray] = None
) -> Tuple[float, float, float, np.ndarray, Dict[str, Any]]:
    """Fit C + B·log(t_idx − z), searching z to avoid the log(1)=0 corner."""
    from ..prefit import analyze_log_trend

    out = analyze_log_trend(y, t_idx, z_grid=z_grid)
    return out.B, out.C, out.z, out.trend, out.pin


def fit_sine_peak_presize(
    y_residual: np.ndarray,
    y_prefit: np.ndarray,
    t: np.ndarray,
    dates: pd.DatetimeIndex,
    n_phase: int = 360,
) -> Dict[str, Any]:
    """Presize sine to N_CYCLE_PEAKS full cycles; endpoint-pin φ to macro peaks."""
    from ..prefit import prefit

    peak_idx = detect_btc_cycle_peaks(y_prefit, dates)
    pf = prefit(
        y_residual,
        t,
        n_sines=1,
        peak_idx=peak_idx,
        omega=float(N_CYCLE_PEAKS),
        n_phase=n_phase,
        envelope_init=True,
    )
    sine_fit = dict(pf.sine_fit)
    sine_fit["n_cycles"] = N_CYCLE_PEAKS
    sine_fit["omega_from_peaks"] = omega_from_peak_times(t, peak_idx)
    sine_fit["period_series_units"] = 1.0 / float(N_CYCLE_PEAKS)
    sine_fit["peak_dates"] = [str(dates[i].date()) for i in peak_idx]
    last_idx = len(t) - 1
    cyclic_end = sine_fit["f_init"] * (1.0 + sine_fit["d_init"] * (1.0 - t[last_idx])) * sine_fit["z"][last_idx]
    sine_fit["endpoint_pin"]["endpoint_residual"] = float(abs(y_residual[last_idx] - cyclic_end))
    return sine_fit


class TimeMLP(nn.Module):
    """Scalar readout f(t) from normalized calendar time."""

    def __init__(self, hidden: int = MLP_HIDDEN, n_layers: int = MLP_LAYERS, out_init: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        d = 1
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self._init_near_constant(out_init)

    def _init_near_constant(self, value: float) -> None:
        with torch.no_grad():
            for module in self.net:
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    module.bias.zero_()
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.bias.fill_(value)

    def forward(self, t_norm: Tensor) -> Tensor:
        return self.net(t_norm.unsqueeze(-1)).squeeze(-1)


class WarpReadout(nn.Module):
    """y = C + B·log(t_idx − z) + f(t)·(1+d(1−t))·z_w with learnable z < 1."""

    def __init__(
        self,
        f_init: float = 0.1,
        d_init: float = 0.0,
        B_init: float = 1.0,
        C_init: float = 0.0,
        z_init: float = 0.0,
        mlp_hidden: int = MLP_HIDDEN,
        mlp_layers: int = MLP_LAYERS,
    ):
        super().__init__()
        self.f_net = TimeMLP(mlp_hidden, mlp_layers, out_init=f_init)
        self.B = nn.Parameter(torch.tensor(B_init, dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor(C_init, dtype=torch.float32))
        z_init = float(min(z_init, 1.0 - 1e-4))
        zeta_init = float(np.log(max(1.0 - z_init, 1e-4)))
        self.zeta = nn.Parameter(torch.tensor(zeta_init, dtype=torch.float32))
        d_init = max(float(d_init), 1e-6)
        self.d_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(d_init))), dtype=torch.float32))

    @property
    def d(self) -> Tensor:
        return torch.nn.functional.softplus(self.d_raw)

    @property
    def z_shift(self) -> Tensor:
        return 1.0 - torch.exp(self.zeta)

    def trend_log(self, t_idx: Tensor) -> Tensor:
        return torch.log(t_idx - self.z_shift)

    def trend_forward(self, t_idx: Tensor) -> Tensor:
        return self.C + self.B * self.trend_log(t_idx)

    def atten(self, t_norm: Tensor) -> Tensor:
        return 1.0 + self.d * (1.0 - t_norm)

    def cyclic_forward(self, t_norm: Tensor, z_w: Tensor) -> Tensor:
        return self.f_net(t_norm) * self.atten(t_norm) * z_w

    def forward(self, t_idx: Tensor, t_norm: Tensor, z_w: Tensor) -> Tensor:
        return self.trend_forward(t_idx) + self.cyclic_forward(t_norm, z_w)


def warped_sine_driver(
    p: np.ndarray,
    sine_fit: Dict[str, Any],
    n_norm: int,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    """Apply fitted soft_warp to the presized sine (analytic, no array clamp)."""
    return soft_warp_sine_numpy(
        p,
        n_norm,
        sine_fit["omega"],
        sine_fit["phase"],
        time_scale=sine_fit.get("time_scale", 1.0),
        t_shift=sine_fit.get("t_shift", 0.0),
        path_anchor=path_anchor,
    )


def predict_components(
    readout: WarpReadout,
    t_idx: np.ndarray,
    t_norm: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    sine_fit: Optional[Dict[str, Any]] = None,
    n_norm: Optional[int] = None,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sine_fit is not None:
        n_ref = n_norm if n_norm is not None else len(t_norm)
        z_w = warped_sine_driver(p, sine_fit, n_ref, path_anchor=path_anchor)
    else:
        z_w = soft_warp_numpy(z, p, path_anchor=path_anchor)
    with torch.no_grad():
        ti = torch.tensor(t_idx, dtype=torch.float32)
        tn = torch.tensor(t_norm, dtype=torch.float32)
        zw = torch.tensor(z_w, dtype=torch.float32)
        trend = readout.trend_forward(ti).numpy()
        cyclic = readout.cyclic_forward(tn, zw).numpy()
    return trend, cyclic, trend + cyclic


def fit_warp_model(
    y: np.ndarray,
    t_idx: np.ndarray,
    t: np.ndarray,
    sine_fit: Dict[str, Any],
    n_knots: int = N_KNOTS,
    epochs: int = EPOCHS,
    lr: float = WARP_LR,
    fit_lambda: float = FIT_LAMBDA,
    mlp_hidden: int = MLP_HIDDEN,
    mlp_layers: int = MLP_LAYERS,
    seed: int = 0,
) -> Dict[str, Any]:
    n = len(y)
    z = np.asarray(sine_fit["z"], dtype=np.float64)[:n]
    z_t = torch.tensor(z, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    t_idx_t = torch.tensor(t_idx, dtype=torch.float32)
    t_norm_t = torch.tensor(t, dtype=torch.float32)
    n_calendar = int(round(0.5 / float(t[0]))) if len(t) else n
    B0, C0, z0, _, _ = fit_log_trend(y, t_idx)
    readout = WarpReadout(
        f_init=sine_fit.get("f_init", 0.1),
        d_init=sine_fit.get("d_init", 0.0),
        B_init=B0,
        C_init=C0,
        z_init=z0,
        mlp_hidden=mlp_hidden,
        mlp_layers=mlp_layers,
    )

    def forward_fn(B: Tensor, _lsy: Tensor, _lst: Tensor) -> Tuple[Tensor, Tensor]:
        p = path_from_B_torch(B, n, n_knots, "identity", path_anchor=DEFAULT_PATH_ANCHOR)
        zw = soft_warp_torch(z_t, p, path_anchor=DEFAULT_PATH_ANCHOR)
        return readout(t_idx_t, t_norm_t, zw), p

    result = train_dual_warp(
        WarpTrainConfig(n=n, n_knots=n_knots, epochs=epochs, lr=lr, fit_lambda=fit_lambda, seed=seed),
        y_t,
        forward_fn,
        extra_params=[readout.B, readout.C, readout.zeta, readout.d_raw]
        + list(readout.f_net.parameters()),
    )
    p = result["warp"]["p"]
    z_w = soft_warp_numpy(z, p, path_anchor=DEFAULT_PATH_ANCHOR)
    trend, cyclic, y_hat = predict_components(readout, t_idx, t, z, p)
    with torch.no_grad():
        f_t = readout.f_net(torch.tensor(t, dtype=torch.float32)).numpy()
        atten_t = readout.atten(torch.tensor(t, dtype=torch.float32)).numpy()
    r2, rmse = _r2_rmse(y, y_hat)
    w = result["warp"]
    return {
        "sine_fit": sine_fit,
        "y_hat": y_hat,
        "trend": trend,
        "cyclic": cyclic,
        "f_t": f_t,
        "atten_t": atten_t,
        "z_warped": z_w,
        "r2": r2,
        "rmse": rmse,
        "B": float(readout.B.detach()),
        "C": float(readout.C.detach()),
        "d": float(readout.d.detach()),
        "t_z": float(readout.z_shift.detach()),
        "readout": readout,
        "mlp_hidden": mlp_hidden,
        "mlp_layers": mlp_layers,
        "warp": {
            "p": p,
            "B_spline": result["B"].numpy(),
            "path_anchor": DEFAULT_PATH_ANCHOR,
            "obj_err": w["obj_err"],
            "obj_time": w["obj_time"],
            "sigma_y": w["sigma_y"],
            "sigma_t": w["sigma_t"],
            "max_abs_offset": float(np.max(np.abs(p - np.arange(n)))),
        },
        "n_knots": n_knots,
        "epochs": epochs,
        "n_calendar": n_calendar,
        "_y_train": np.asarray(y, dtype=np.float64),
    }


def sine_fit_from_checkpoint(
    sine_fit: Dict[str, Any],
    ckpt: Dict[str, Any],
) -> Dict[str, Any]:
    """Align presized sine parameters with a saved checkpoint."""
    return {
        **sine_fit,
        "omega": float(ckpt.get("omega", sine_fit["omega"])),
        "phase": float(ckpt.get("phase", sine_fit["phase"])),
        "time_scale": float(ckpt.get("time_scale", sine_fit.get("time_scale", 1.0))),
        "t_shift": float(ckpt.get("t_shift", sine_fit.get("t_shift", 0.0))),
    }


def build_forecast_extension(
    n_current: int,
    n_future_extra: int,
    sine_fit: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Calendar arrays for ``n_current + n_future_extra`` days (t_norm base = n_current)."""
    n_total = n_current + n_future_extra
    t_ext = (np.arange(n_total, dtype=np.float64) + 0.5) / n_current
    t_idx_ext = day_index(n_total, start=1)
    z_ext = _sine_from_fit(t_ext, sine_fit)
    return t_ext, t_idx_ext, z_ext, n_total


def forecast_future(
    fit: Dict[str, Any],
    t_idx_full: np.ndarray,
    t_norm_full: np.ndarray,
    z_full: np.ndarray,
    n_obs: int,
    n_paths: int = N_PATHS,
    n_paths_ci: Optional[int] = None,
    seed: int = 1,
) -> Dict[str, Any]:
    n_total = len(z_full)
    p_train = fit["warp"]["p"]
    readout: WarpReadout = fit["readout"]
    sigma_t = fit["warp"]["sigma_t"]
    sigma_y = fit["warp"]["sigma_y"]
    n_knots = fit["n_knots"]
    n_ci = n_paths_ci if n_paths_ci is not None else N_PATHS

    with torch.no_grad():
        trend_full = readout.trend_forward(
            torch.tensor(t_idx_full, dtype=torch.float32)
        ).numpy()

    paths_p = sample_warp_paths_future_knots(
        p_train, n_total, n_obs, sigma_t, n_knots, n_paths, seed
    )
    paths_p_ci = sample_warp_paths_future(
        p_train, n_total, n_obs, sigma_t, n_knots, n_ci, seed + 1000
    )
    p_det = point_forecast_path(p_train, n_total)
    sine_fit = fit.get("sine_fit")
    n_cal = fit.get("n_calendar", n_obs)
    pred_kw = {"sine_fit": sine_fit, "n_norm": n_cal} if sine_fit else {}
    cyclic_paths = np.stack([
        predict_components(
            readout, t_idx_full, t_norm_full, z_full, paths_p[k], **pred_kw
        )[1]
        for k in range(n_paths)
    ])
    cyclic_ci = np.stack([
        predict_components(
            readout, t_idx_full, t_norm_full, z_full, paths_p_ci[k], **pred_kw
        )[1]
        for k in range(n_ci)
    ])
    _, cyclic_point, y_point = predict_components(
        readout, t_idx_full, t_norm_full, z_full, p_det, **pred_kw
    )
    preds = trend_full[np.newaxis, :] + cyclic_paths
    preds_ci = trend_full[np.newaxis, :] + cyclic_ci

    z_warped_paths = z_warped_point = None
    if sine_fit is not None:
        z_warped_paths = np.stack([
            warped_sine_driver(paths_p[k], sine_fit, n_cal) for k in range(n_paths)
        ])
        z_warped_point = warped_sine_driver(p_det, sine_fit, n_cal)

    bands = build_forecast_bands(preds_ci, y_point, sigma_y, noise_seed=seed + 1)
    bands["c_q50"] = y_point
    fut_sl = slice(n_obs, n_total)
    path_std = float(np.mean(np.std(cyclic_paths[:, fut_sl], axis=0)))
    return {
        "preds": preds,
        "preds_ci": preds_ci,
        "y_point": y_point,
        "trend": trend_full,
        "bands": bands,
        "paths_p": paths_p,
        "paths_p_ci": paths_p_ci,
        "p_det": p_det,
        "z_warped_paths": z_warped_paths,
        "z_warped_point": z_warped_point,
        "n_obs": n_obs,
        "n_total": n_total,
        "n_future": n_total - n_obs,
        "path_std_mean_future": path_std,
    }


def load_production_fit(
    y: np.ndarray,
    t_idx: np.ndarray,
    t: np.ndarray,
    sine_fit: Dict[str, Any],
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reload warp + readout from disk (train z driver must match ``y`` length)."""
    if model_path is None:
        raise ValueError("model_path required")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    metrics_path = model_path.parent / "metrics.json"
    holdout_metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        holdout_metrics = json.loads(metrics_path.read_text()).get("holdout", {})
    n = len(y)
    n_knots = int(ckpt["n_knots"])
    z = np.asarray(sine_fit["z"], dtype=np.float64)[:n]
    B0, C0, z0, _, _ = fit_log_trend(y, t_idx)
    readout = WarpReadout(
        f_init=sine_fit.get("f_init", 0.1),
        d_init=sine_fit.get("d_init", 0.0),
        B_init=B0,
        C_init=C0,
        z_init=z0,
        mlp_hidden=int(ckpt.get("mlp_hidden", MLP_HIDDEN)),
        mlp_layers=int(ckpt.get("mlp_layers", MLP_LAYERS)),
    )
    readout.load_state_dict(ckpt["readout_state"])
    readout.eval()
    if "p_train" in ckpt and len(ckpt["p_train"]) == n:
        p = np.asarray(ckpt["p_train"], dtype=np.float64)
    else:
        p = path_from_B_torch(
            torch.tensor(ckpt["B_spline"], dtype=torch.float32), n, n_knots, "identity",
            path_anchor=DEFAULT_PATH_ANCHOR,
        ).numpy()
    trend, cyclic, y_hat = predict_components(readout, t_idx, t, z, p)
    with torch.no_grad():
        f_t = readout.f_net(torch.tensor(t, dtype=torch.float32)).numpy()
        atten_t = readout.atten(torch.tensor(t, dtype=torch.float32)).numpy()
    r2, rmse = _r2_rmse(y, y_hat)
    off = stored_path_offset_numpy(p)
    return {
        "sine_fit": sine_fit,
        "y_hat": y_hat,
        "trend": trend,
        "cyclic": cyclic,
        "f_t": f_t,
        "atten_t": atten_t,
        "z_warped": soft_warp_numpy(z, p, path_anchor=DEFAULT_PATH_ANCHOR),
        "r2": r2,
        "rmse": rmse,
        "B": float(readout.B.detach()),
        "C": float(readout.C.detach()),
        "d": float(readout.d.detach()),
        "t_z": float(readout.z_shift.detach()),
        "readout": readout,
        "mlp_hidden": MLP_HIDDEN,
        "mlp_layers": MLP_LAYERS,
        "warp": {
            "p": p,
            "B_spline": np.asarray(ckpt["B_spline"], dtype=np.float64),
            "path_anchor": DEFAULT_PATH_ANCHOR,
            "sigma_y": holdout_metrics.get("sigma_y", 0.0),
            "sigma_t": holdout_metrics.get("sigma_t", 0.0),
            "max_abs_offset": float(np.max(np.abs(off))),
        },
        "n_knots": n_knots,
        "epochs": EPOCHS,
    }


def forecast_bitcoin_holdout(
    fit: Dict[str, Any],
    t_idx: np.ndarray,
    t_norm: np.ndarray,
    z_full: np.ndarray,
    split: Dict[str, Any],
    y_test: Optional[np.ndarray] = None,
    n_paths: int = N_SINGLE_PATHS,
    seed: int = 43,
) -> Dict[str, Any]:
    """Holdout forecast via ``predict_forecast_realisations_torch`` (sampled warp paths)."""
    from ..forecast import predict_forecast_realisations_torch

    n_train = split["n_train"]
    test_idx = split["test_idx"]
    fc = predict_forecast_realisations_torch(
        fit=fit,
        t_idx=t_idx,
        t_norm=t_norm,
        z_full=z_full,
        n_obs=n_train,
        n_future=split["n_test"],
        n_draws=n_paths,
        seed=seed,
        noise_seed=seed + 1,
    )
    bands = fc["bands"]
    y_point = fc["y_point"]
    out: Dict[str, Any] = {
        **fc,
        "split": split,
        "sigma_t": fit["warp"]["sigma_t"],
        "sigma_y": fit["warp"]["sigma_y"],
    }
    y_hat_tr = fit.get("y_hat")
    if y_hat_tr is not None:
        out["corr_log_train"] = float(np.corrcoef(y_point[:n_train], y_hat_tr)[0, 1])
    if y_test is not None:
        out["corr_log_test"] = float(np.corrcoef(y_point[test_idx], y_test)[0, 1])
        out["rmse_log_test"] = float(np.sqrt(np.mean((y_point[test_idx] - y_test) ** 2)))
        in_combined = (y_test >= bands["c_q_lo"][test_idx]) & (y_test <= bands["c_q_hi"][test_idx])
        out["coverage_combined_95_test"] = float(in_combined.mean())
    return out

