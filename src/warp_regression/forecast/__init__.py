"""Series-agnostic forecasting: bands, path sampling, state, residual layer."""

from .api import predict_forecast_realisations_torch, predict_realisations_torch
from .bands import (
    build_forecast_bands,
    ci_band_params,
    count_bands_from_log1p_forecast,
    interval_coverage,
    log1p_error_band_counts,
)
from .components import observation_components, predict_components
from .paths import (
    extended_knot_positions,
    per_index_rw_sigma,
    sample_warp_paths_future,
    sample_warp_paths_future_knots,
)
from .residual import (
    ResidualSmoothFit,
    apply_residual_forecast,
    fit_residual_smooth,
    forecast_residual,
    residual_adjustment,
)
from .state import ForecastState, as_forecast_state, forecast_from_state

__all__ = [
    "ForecastState",
    "ResidualSmoothFit",
    "apply_residual_forecast",
    "as_forecast_state",
    "build_forecast_bands",
    "ci_band_params",
    "count_bands_from_log1p_forecast",
    "extended_knot_positions",
    "fit_residual_smooth",
    "forecast_from_state",
    "forecast_residual",
    "interval_coverage",
    "log1p_error_band_counts",
    "observation_components",
    "per_index_rw_sigma",
    "predict_components",
    "predict_forecast_realisations_torch",
    "predict_realisations_torch",
    "residual_adjustment",
    "sample_warp_paths_future",
    "sample_warp_paths_future_knots",
]
