"""Likelihood-based time warping for regression and forecasting.

Typical imports
---------------
::

    from warp_regression import WarpModel, soft_warp_numpy, DEFAULT_PATH_ANCHOR
    from warp_regression import as_forecast_state, forecast_from_state, build_forecast_bands
    from warp_regression import prefit, analyze_cycle_lengths
"""

from .block import WarpPath, WarpRegression
from .core.path import (
    DEFAULT_PATH_ANCHOR,
    PathAnchor,
    applied_path_offset_numpy,
    stored_path_offset_numpy,
)
from .core.warp import soft_warp_numpy, soft_warp_sine_numpy
from .forecast import (
    ForecastState,
    ResidualSmoothFit,
    apply_residual_forecast,
    as_forecast_state,
    build_forecast_bands,
    ci_band_params,
    count_bands_from_log1p_forecast,
    fit_residual_smooth,
    forecast_from_state,
    forecast_residual,
    interval_coverage,
    log1p_error_band_counts,
    observation_components,
    per_index_rw_sigma,
    predict_components,
    predict_forecast_realisations_torch,
    predict_realisations_torch,
    sample_warp_paths_future,
    sample_warp_paths_future_knots,
)
from .model import FitResult, ForecastResult, WarpModel
from .observation import ObservationModel
from .cycle_analysis import (
    CycleLengthAnalysis,
    analyze_cycle_lengths,
    nominal_cycle_length,
    plot_cycle_length_distribution,
    sample_next_cycle_lengths,
)
from .prefit import (
    LogTrendAnalysis,
    PrefitResult,
    analyze_log_trend,
    prefit,
    prefit_synthetic_path,
)
from .covariates.sine import (
    align_sine_to_macro_peaks,
    build_dual_sines_from_fit,
    fit_dual_sine_log,
    sine_from_fit,
    eval_sine_driver,
)
from .utilities.splits import cumsum_path_to_stored_path, split_holdout
from .utilities.metrics import EvalReport

__all__ = [
    "DEFAULT_PATH_ANCHOR",
    "PathAnchor",
    "WarpPath",
    "WarpRegression",
    "WarpModel",
    "ObservationModel",
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
    "EvalReport",
    "ForecastState",
    "as_forecast_state",
    "forecast_from_state",
    "build_forecast_bands",
    "ci_band_params",
    "interval_coverage",
    "count_bands_from_log1p_forecast",
    "log1p_error_band_counts",
    "predict_forecast_realisations_torch",
    "predict_realisations_torch",
    "predict_components",
    "observation_components",
    "per_index_rw_sigma",
    "sample_warp_paths_future",
    "sample_warp_paths_future_knots",
    "applied_path_offset_numpy",
    "stored_path_offset_numpy",
    "cumsum_path_to_stored_path",
    "fit_dual_sine_log",
    "split_holdout",
    "sine_from_fit",
    "eval_sine_driver",
    "align_sine_to_macro_peaks",
    "build_dual_sines_from_fit",
    "soft_warp_numpy",
    "soft_warp_sine_numpy",
]
