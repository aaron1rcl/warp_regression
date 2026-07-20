"""Train/test holdout splits and path-index helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..core.path import DEFAULT_PATH_ANCHOR, PathAnchor


def split_holdout(
    n: Optional[int] = None,
    *,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    years: Optional[np.ndarray] = None,
    train_end_year: Optional[float] = None,
    dates: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
    min_train: int = 1,
) -> Dict[str, Any]:
    """Generic end-holdout split.

    Two ways to cut the series (exactly one must be used):

    1. **By index count** — pass ``n`` and either ``n_train`` or ``n_test``.
       Test is always the trailing block.
    2. **By time threshold** — pass ``years`` and ``train_end_year``; train is
       ``years <= train_end_year``.

    Optional ``dates`` (aligned with the series) adds date/year metadata to the
    result when the cut is by index.
    """
    if years is not None and train_end_year is not None:
        years = np.asarray(years)
        if n is not None and int(n) != len(years):
            raise ValueError(f"n={n} does not match len(years)={len(years)}")
        train_mask = years <= float(train_end_year)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(~train_mask)[0]
        out: Dict[str, Any] = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_end_year": float(train_end_year),
            "years_train": years[train_mask],
            "years_test": years[~train_mask],
        }
        if dates is not None:
            dates_arr = pd.DatetimeIndex(dates) if not isinstance(dates, pd.DatetimeIndex) else dates
            if len(dates_arr) != len(years):
                raise ValueError("dates length must match years")
            if len(train_idx):
                out["train_end_date"] = dates_arr[int(train_idx[-1])]
            if len(test_idx):
                out["test_start_date"] = dates_arr[int(test_idx[0])]
            out["dates_train"] = dates_arr[train_idx]
            out["dates_test"] = dates_arr[test_idx]
        return out

    if years is not None or train_end_year is not None:
        raise ValueError("year split requires both years= and train_end_year=")

    if n is None:
        raise ValueError("index split requires n=")
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    if (n_train is None) == (n_test is None):
        raise ValueError("pass exactly one of n_train= or n_test=")

    min_train = int(max(min_train, 1))
    if n_test is not None:
        n_test_i = int(min(max(int(n_test), 1), n - min_train))
        n_train_i = n - n_test_i
    else:
        n_train_i = int(min(max(int(n_train), min_train), n - 1))
        n_test_i = n - n_train_i

    train_idx = np.arange(n_train_i, dtype=int)
    test_idx = np.arange(n_train_i, n, dtype=int)
    out = {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_train": n_train_i,
        "n_test": n_test_i,
    }
    if dates is not None:
        dates_arr = pd.DatetimeIndex(dates) if not isinstance(dates, pd.DatetimeIndex) else dates
        if len(dates_arr) != n:
            raise ValueError(f"dates length {len(dates_arr)} != n={n}")
        out["train_end_date"] = dates_arr[n_train_i - 1]
        out["test_start_date"] = dates_arr[n_train_i]
        out["dates_train"] = dates_arr[train_idx]
        out["dates_test"] = dates_arr[test_idx]
    return out


# Thin aliases kept for existing notebooks/tests
def split_holdout_by_year(data: Dict[str, Any], train_end_year: float) -> Dict[str, Any]:
    return split_holdout(years=data["years"], train_end_year=train_end_year)


def split_lynx_holdout(data: Dict[str, Any], train_end_year: int = 1910) -> Dict[str, Any]:
    return split_holdout(years=data["years"], train_end_year=train_end_year)


def split_synthetic_holdout(n: int, n_train: int = 200) -> Dict[str, Any]:
    return split_holdout(n, n_train=n_train)


def split_bitcoin_holdout(
    n: int,
    test_days: int = 365,
    dates: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
) -> Dict[str, Any]:
    out = split_holdout(n, n_test=int(test_days), dates=dates, min_train=30)
    out["test_days"] = out["n_test"]
    return out


def cumsum_path_to_stored_path(
    p_cumsum: np.ndarray,
    n: int,
    sr: int = 10,
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> np.ndarray:
    """Map discrete cumsum shifts to stored path ``p[i]=i+offset[i]``.

    With ``path_anchor='start'`` (default), offset is pinned at the train start
    (``offset[0]=0``). With ``'end'``, offset is pinned at the train end
    (``offset[n-1]=0``), matching the old ``path_from_B_*`` convention.
    """
    p_cumsum = np.asarray(p_cumsum, dtype=np.float64)
    n_use = min(int(n), len(p_cumsum))
    off = (p_cumsum[:n_use] - p_cumsum[0]) / float(sr)
    idx = np.arange(n_use, dtype=np.float64)
    if path_anchor == "end":
        off = off - off[-1]
        p = idx + off
        p[-1] = float(n_use - 1)
    else:
        p = idx + off
        p[0] = 0.0
    return p
