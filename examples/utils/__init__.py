"""Helpers used only by the example notebooks (and their tests).

Import as::

    import utils
    from utils import prepare_lynx_log, plot_forecast_bands, ...
"""

from .data import (
    BITCOIN_CSV,
    CYCLE_PEAK_HALF_WINDOW_DAYS,
    CYCLE_PEAK_YEARS,
    DATA_PATH,
    DATA_ROOT,
    LYNX_CSV,
    N_CYCLE_PEAKS,
    TEST_DAYS,
    build_forecast_extension,
    build_synthetic_dataset,
    day_index,
    detect_btc_cycle_peaks,
    fetch_bitcoin_daily,
    fit_log_trend,
    fit_sine_peak_presize,
    log_price,
    normalized_time,
    prepare_lynx_log,
    price_from_log,
    sine_from_fit,
    split_bitcoin_holdout,
)
from .fits import (
    fit_dual_sine_shared_warp,
    fit_dual_sine_shared_warp_nonlinear,
)
from .eval import evaluate_model, roll_forward_synthetic
from .plotting import (
    format_date_axis,
    plot_before_after_warp,
    plot_dual_objective_scatter,
    plot_fit_with_residual,
    plot_forecast_bands,
    plot_realisation_spaghetti,
    plot_warp_offset,
)
from warp_regression.utilities.splits import (
    cumsum_path_to_stored_path,
    split_holdout,
    split_holdout_by_year,
    split_lynx_holdout,
    split_synthetic_holdout,
)

NOTEBOOK_LL_TARGET = (283.71669407633806, 506.8158229257141)

__all__ = [
    "BITCOIN_CSV",
    "CYCLE_PEAK_HALF_WINDOW_DAYS",
    "CYCLE_PEAK_YEARS",
    "DATA_PATH",
    "DATA_ROOT",
    "LYNX_CSV",
    "N_CYCLE_PEAKS",
    "NOTEBOOK_LL_TARGET",
    "TEST_DAYS",
    "build_forecast_extension",
    "build_synthetic_dataset",
    "cumsum_path_to_stored_path",
    "day_index",
    "detect_btc_cycle_peaks",
    "evaluate_model",
    "roll_forward_synthetic",
    "fetch_bitcoin_daily",
    "fit_dual_sine_shared_warp",
    "fit_dual_sine_shared_warp_nonlinear",
    "fit_log_trend",
    "fit_sine_peak_presize",
    "format_date_axis",
    "log_price",
    "normalized_time",
    "plot_before_after_warp",
    "plot_dual_objective_scatter",
    "plot_fit_with_residual",
    "plot_forecast_bands",
    "plot_realisation_spaghetti",
    "plot_warp_offset",
    "prepare_lynx_log",
    "price_from_log",
    "sine_from_fit",
    "split_bitcoin_holdout",
    "split_holdout",
    "split_holdout_by_year",
    "split_lynx_holdout",
    "split_synthetic_holdout",
]
