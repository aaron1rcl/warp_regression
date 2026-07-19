"""Shared constants and types."""
from __future__ import annotations
from pathlib import Path
from typing import Literal

PathAnchor = Literal["start", "end"]
DEFAULT_PATH_ANCHOR: PathAnchor = "start"

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT.parent / "data"
LYNX_CSV = DATA_ROOT / "lynx.csv"
BITCOIN_CSV = DATA_ROOT / "bitcoin_daily.csv"
SUNSPOTS_CSV = DATA_ROOT / "sunspots_monthly.csv"
NOTEBOOK_LL_TARGET = (283.71669407633806, 506.8158229257141)
