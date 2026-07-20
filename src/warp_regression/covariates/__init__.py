"""Covariate utilities (e.g. sine components for warp regression)."""

from .sine import (
    align_sine_to_macro_peaks,
    build_dual_sines_from_fit,
    eval_sine_driver,
    fit_dual_sine_log,
    refine_param_endpoint,
    sine_from_fit,
)

__all__ = [
    "align_sine_to_macro_peaks",
    "build_dual_sines_from_fit",
    "eval_sine_driver",
    "fit_dual_sine_log",
    "refine_param_endpoint",
    "sine_from_fit",
]
