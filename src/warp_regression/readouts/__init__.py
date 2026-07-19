"""Legacy fit wrappers (prefer ``WarpModel.from_yaml``)."""

from .parametric import WarpParametricModel, evaluate_model, predict_realisations_torch

__all__ = [
    "WarpParametricModel",
    "evaluate_model",
    "predict_realisations_torch",
]
