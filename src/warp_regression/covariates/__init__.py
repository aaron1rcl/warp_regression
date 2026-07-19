"""Covariate utilities (e.g. sine components for warp regression)."""

from .sine import (
    SineSpec,
    align_sine_to_macro_peaks,
    build_dual_sines_from_fit,
    build_sine_features,
    eval_sine_driver,
    fit_dual_sine_log,
    presize_dual_sine,
    presize_log_trend_sine,
    refine_param_endpoint,
    sine_wave,
)

__all__ = [
    "SineSpec",
    "align_sine_to_macro_peaks",
    "build_dual_sines_from_fit",
    "build_sine_features",
    "eval_sine_driver",
    "fit_dual_sine_log",
    "presize_dual_sine",
    "presize_log_trend_sine",
    "refine_param_endpoint",
    "sine_wave",
]
