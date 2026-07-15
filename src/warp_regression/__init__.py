"""Unified warp regression package."""

from .constants import DEFAULT_PATH_ANCHOR, PathAnchor, NOTEBOOK_LL_TARGET
from .forecast import (
    build_forecast_bands,
    count_bands_from_log1p_forecast,
    forecast_lynx_holdout_paths,
    log1p_error_band_counts,
    per_index_rw_sigma,
    predict_forecast_realisations_torch,
    sample_warp_paths_future,
    sample_warp_paths_future_knots,
)
from .model import FitResult, ForecastResult, WarpModel, WarpModelConfig
from .cycle_analysis import (
    CycleLengthAnalysis,
    analyze_cycle_lengths,
    nominal_cycle_length,
    plot_cycle_length_distribution,
    sample_next_cycle_lengths,
)
from .residual_smooth import (
    ResidualSmoothFit,
    apply_residual_forecast,
    fit_residual_smooth,
    forecast_residual,
)
from .prefit import (
    LogTrendAnalysis,
    PrefitResult,
    analyze_log_trend,
    prefit,
    prefit_synthetic_path,
)
from .readouts.parametric import EvalReport, WarpParametricModel, evaluate_model
from .readouts.dual import (
    fit_dual_sine_shared_warp,
    fit_dual_sine_shared_warp_nonlinear,
)
from .readouts.log_trend_sine import (
    WarpReadout,
    fetch_bitcoin_daily,
    fit_log_trend,
    fit_sine_peak_presize,
    fit_warp_model,
    forecast_future,
    build_forecast_extension,
    log_price,
    normalized_time,
    day_index,
    split_bitcoin_holdout,
    sine_from_fit,
)
from .drivers.sine import (
    align_sine_to_macro_peaks,
    build_dual_sines_from_fit,
    fit_dual_sine_log,
    build_sine_features,
    presize_dual_sine,
    presize_log_trend_sine,
    SineSpec,
    sine_wave,
)
from .utilities.datasets import build_synthetic_dataset, prepare_lynx_log, prepare_sunspots
from .utilities.splits import (
    cumsum_path_to_stored_path,
    split_holdout_by_year,
    split_lynx_holdout,
    split_synthetic_holdout,
)
from .utilities.metrics import _r2_rmse
from .plotting import (
    format_date_axis,
    plot_before_after_warp,
    plot_dual_objective_scatter,
    plot_fit_with_residual,
    plot_forecast_bands,
    plot_realisation_spaghetti,
    plot_warp_offset,
)
from .core.path import applied_path_offset_numpy, stored_path_offset_numpy, point_forecast_path
from .readouts.parametric import predict_realisations_torch
from .core.warp import soft_warp_numpy, soft_warp_sine_numpy

__all__ = [
    "DEFAULT_PATH_ANCHOR",
    "PathAnchor",
    "WarpModel",
    "WarpModelConfig",
    "FitResult",
    "ForecastResult",
    "ResidualSmoothFit",
    "fit_residual_smooth",
    "forecast_residual",
    "apply_residual_forecast",
    "CycleLengthAnalysis",
    "analyze_cycle_lengths",
    "nominal_cycle_length",
    "plot_cycle_length_distribution",
    "sample_next_cycle_lengths",
    "PrefitResult",
    "LogTrendAnalysis",
    "prefit",
    "prefit_synthetic_path",
    "analyze_log_trend",
    "WarpParametricModel",
    "WarpReadout",
    "EvalReport",
    "NOTEBOOK_LL_TARGET",
    "build_forecast_bands",
    "count_bands_from_log1p_forecast",
    "log1p_error_band_counts",
    "forecast_lynx_holdout_paths",
    "predict_forecast_realisations_torch",
    "predict_realisations_torch",
    "applied_path_offset_numpy",
    "stored_path_offset_numpy",
    "cumsum_path_to_stored_path",
    "evaluate_model",
    "fit_dual_sine_log",
    "fit_dual_sine_shared_warp",
    "fit_dual_sine_shared_warp_nonlinear",
    "fit_warp_model",
    "forecast_future",
    "build_forecast_extension",
    "build_synthetic_dataset",
    "prepare_lynx_log",
    "prepare_sunspots",
    "split_synthetic_holdout",
    "split_holdout_by_year",
    "split_lynx_holdout",
    "split_bitcoin_holdout",
    "fetch_bitcoin_daily",
    "fit_log_trend",
    "fit_sine_peak_presize",
    "sine_from_fit",
    "build_sine_features",
    "SineSpec",
    "sine_wave",
    "presize_dual_sine",
    "presize_log_trend_sine",
    "plot_realisation_spaghetti",
    "plot_forecast_bands",
    "plot_warp_offset",
    "plot_fit_with_residual",
    "plot_before_after_warp",
    "plot_dual_objective_scatter",
    "format_date_axis",
]
