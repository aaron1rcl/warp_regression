from __future__ import annotations


from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

def split_lynx_holdout(data: Dict[str, Any], train_end_year: int = 1910) -> Dict[str, Any]:
    years = data["years"]
    train_mask = years <= train_end_year
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]
    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_end_year": train_end_year,
        "years_train": years[train_mask],
        "years_test": years[~train_mask],
    }


def split_synthetic_holdout(n: int, n_train: int = 200) -> Dict[str, Any]:
    """Train/test index split for the synthetic holdout notebook."""
    n_train = int(min(n_train, n))
    train_idx = np.arange(n_train, dtype=int)
    test_idx = np.arange(n_train, n, dtype=int)
    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_train": n_train,
        "n_test": n - n_train,
    }


def cumsum_path_to_stored_path(p_cumsum: np.ndarray, n: int, sr: int = 10) -> np.ndarray:
    """Map discrete cumsum shifts to stored identity path ``p[i]=i+offset[i]``, ``offset[0]=0``."""
    p_cumsum = np.asarray(p_cumsum, dtype=np.float64)
    n_use = min(n, len(p_cumsum))
    off = (p_cumsum[:n_use] - p_cumsum[0]) / float(sr)
    idx = np.arange(n_use, dtype=np.float64)
    p = idx + off
    p[0] = 0.0
    return p


