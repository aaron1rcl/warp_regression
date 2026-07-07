from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _r2_rmse(y: np.ndarray, y_hat: np.ndarray) -> Tuple[float, float]:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-8
    return 1.0 - ss_res / ss_tot, float(np.sqrt(ss_res / len(y)))


@dataclass
class EvalReport:
    mse: float
    rmse: float
    corr: float
    obj_err: float
    obj_time: float
    err_nll: float
    time_ll: float
    ll_distance: float
    discrete_mse: Optional[float] = None
