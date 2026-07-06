"""Notebook shim: re-export Bitcoin helpers from the package."""

from warp_regression.readouts import log_trend_sine as _btc

for _name in dir(_btc):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_btc, _name)
