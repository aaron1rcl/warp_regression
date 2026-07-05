# warp_regression

Unified warp regression API for synthetic drivers, Lynx population forecasting, and Bitcoin log-price holdout evaluation.

## Layout

```
legacy/          # archived ShiftNetwork reference
src/
  warp_regression/   # importable package (WarpModel facade)
  data/              # lynx.csv, bitcoin_daily.csv
  tests/             # pytest + baseline metrics
  notebooks/         # slim holdout demos + HTML exports
```

## Install

```bash
pip install -e ".[dev]"
```

## Tests

```bash
pytest src/tests/ -v
pytest src/tests/ -v -m slow   # full Bitcoin reproduction (24k epochs)
```

## Notebooks

```bash
jupyter nbconvert --execute --to notebook --inplace src/notebooks/*.ipynb
jupyter nbconvert --to html src/notebooks/*.ipynb --output-dir src/notebooks/
```
