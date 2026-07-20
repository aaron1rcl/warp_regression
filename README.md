# warp_regression

Likelihood-based time warping for regression and forecasting on **cyclical** series — cycles whose timing drifts, not just their height.

---

## What is warp regression?

For a while the interesting idea has been to model errors in the **time** dimension explicitly, not only on the $y$-axis. If your parcel arrives a day late, you do not say you are one package short. The mistake is *when*, not *how many*. Most time-series tools still dump timing mistakes into the residual on $y$. Warp regression keeps them separate: a path $p$ says *when* a known shape arrives; an observation model $\mu$ says *what level* you see once it has arrived.

At each index $i$,

$$
\hat{y}_i = \mu\big(\mathrm{warp}(x, p)_i\big),
$$

where $x$ is a covariate (often a sine or other known shape), $p$ is a low-dimensional warp path (fractional indices into $x$), and $\mu$ is linear, an MLP, trend + cycle, …. Soft-warp interpolates $x$ at those indices so $p$ is differentiable.

### Terror (timing error)

Training uses a **dual likelihood**:

| Term | Scale | Meaning |
|------|-------|---------|
| **error** | $\sigma_y$ | Usual Gaussian fit of $y$ vs $\hat{y}$ |
| **terror** | $\sigma_t$ | Likelihood on path offsets $p(i) - i$ — how plausible is this timing? |

(*Terror* = **t**iming **error**.) The objective mixes them with weight $\lambda$ (`fit_lambda`):

$$
\mathcal{J} = \lambda \cdot (-\log p(y \mid \hat{y},\sigma_y)) - (1-\lambda)\cdot \log p(p \mid \sigma_t).
$$

| $\lambda$ | Interpretation |
|-----------|----------------|
| $0.5$ | Equal weight on both log-likelihoods — the natural joint (neither term preferred *a priori*) |
| $\to 1$ | Fit only: terror off; the path may warp freely to reduce residual error |
| $\to 0$ | Terror only: timing prior dominates; $y$-fit is barely scored |

**What is the terror likelihood?** Timing is scored as a Gaussian random walk on offsets, with scale $\sigma_t$: the clock is allowed to drift early/late from segment to segment. A literal free random-walk path would be *rough* (Brownian paths are almost surely nowhere differentiable; a discrete RW is jagged) and would need a free value at every index — too many degrees of freedom for a useful warp.

So the fitted path is **piecewise linear through $K \ll n$ knots**. Terror is the **expected log-likelihood of a Brownian bridge** on each knot segment: integrate out the unknown rough path between the two knot endpoints, given their displacement and $\sigma_t$. That closed form (`expected_likelihood`) is $O(K)$, autograd-friendly, and keeps the RW generative story (including forecast path sampling) without optimizing a jagged path.

Because timing has its own scale, forecasts can continue the warp as a random walk and build:

- **terror bands** — uncertainty from future timing alone  
- **error bands** — observation noise alone  
- **combined bands** — both together  

That is also how you get a distribution over *next cycle length*, not only next height.

---

## Cyclical ≠ seasonal

Macro and ecological series often look periodic without a fixed calendar phase: business cycles, commodity booms, predator–prey peaks, Bitcoin’s ~4-year halving rhythm. These are **cyclical, not seasonal**. Seasonality resets every January; a cycle can run long or short. The *shape* recurs; the *clock* slips.

In this repo that warping shows up as:

- **Synthetic** — a known sine pushed through a hidden path (grade recovery).  
- **Lynx** — two sines share one warp: both components speed up or slow down together.  
- **Bitcoin** — one macro cycle rides a strong log-trend; the path absorbs early/late peaks.

In practice those series are hard to (1) **analyse** without forcing a fixed period, (2) **forecast** when the next peak’s date is the question, and (3) attach honest **uncertainty** to cycle timing rather than only to level.

---

## Why use this — and when not to

**Benefits**

- Separates shape ($x$, $\mu$) from timing ($p$, $\sigma_t$).  
- One objective for fit and path plausibility; gradients end to end in PyTorch.  
- Forecast uncertainty includes *when*, via path sampling (terror / combined bands, cycle-length draws).  
- Works at small-to-medium $n$ with a hand-specified cycle shape (Lynx-scale series are in scope).

**Compared with nearby tools** (detail in [`legacy/info/comparable_methods.md`](legacy/info/comparable_methods.md)):

| Approach | Overlap | Gap vs warp regression |
|----------|---------|------------------------|
| **DTW / soft-DTW** | Aligns by stretching time | Optimisation path, not a generative timing law — weak for forecasts / $\sigma_t$ |
| **Structural TS / TBATS** | Cycles + probabilistic forecasts | Fixed frequency; phase wanders, period does not |
| **Neural warping (DTAN, …)** | Learned warps | Built for alignment / ensembles, not cycle-length forecasts |
| **GPs / deep sequence nets** | Flexible forecasts | Timing absorbed into the black box; no explicit path |

**Use it** when you have a plausible parametric shape, timing drifts off a fixed calendar, sample size is tens to a few thousand points, and “when does the next peak land?” matters as much as “how high?”.

**Skip it** when there is no shape to warp (use a GP or deep net); when you need mature SEs and textbook tooling on long high-frequency series (UCM / TBATS); when the task is aligning many signals, not forecasting one (DTW); when timestamps are irregular (GPs handle that natively; this package assumes a regular index); or when a research-only repo is disqualifying for production.

---

## Install

```bash
pip install -e ".[dev]"
# optional: fully Bayesian notebook (PyMC + JAX / NumPyro)
pip install -e ".[bayes]"
```

## Quick start

Minimal block — `WarpPath` + `WarpRegression` in ordinary PyTorch:

```python
import torch
import torch.nn as nn
from warp_regression import WarpPath, WarpRegression

n, n_knots = 100, 8
path = WarpPath(n, n_knots, path_anchor="start")
warp = WarpRegression(path, covariate_kind="array", name="x")

A = nn.Parameter(torch.tensor(1.0))
C = nn.Parameter(torch.tensor(0.0))

x = torch.sin(2 * torch.pi * torch.arange(n) / n)
p = path.path()                 # identity at init (B = 0)
y_hat = A * warp.warp(x, p) + C
```

For YAML models, dual loss, and forecast bands, see the notebooks and `WarpModel.from_yaml(...)`.

---

## Examples

| Notebook | Setting |
|----------|---------|
| [`0_Warp_Block_Basics.ipynb`](examples/notebooks/0_Warp_Block_Basics.ipynb) | Synthetic: warp layer, error-only vs dual loss |
| [`1_Introduction_to_Warp_Regression.ipynb`](examples/notebooks/1_Introduction_to_Warp_Regression.ipynb) | Synthetic sine + `WarpModel` + forecast bands |
| [`2_Adding_complexity_Lynx_Forecast.ipynb`](examples/notebooks/2_Adding_complexity_Lynx_Forecast.ipynb) | Hudson Bay lynx (dual sines, shared warp) |
| [`3_Bitcoin_Warp.ipynb`](examples/notebooks/3_Bitcoin_Warp.ipynb) | Daily BTC log-price |
| [`4_Fully_Bayesian.ipynb`](examples/notebooks/4_Fully_Bayesian.ipynb) | Full-series PyMC + JAX dual posterior (λ=0.5) |

HTML under [`examples/html/`](examples/html/). Configs in [`examples/models/`](examples/models/).

---

## Package

| Piece | Role |
|-------|------|
| `WarpPath` / `WarpRegression` | Portable warp block |
| `WarpModel` | Paths, blocks, observation $\mu$, dual fit, forecast |
| `observation` | Term kinds for $\mu$ |
| `forecast` | Path continuation and bands |
| `prefit` | Covariates (e.g. sine) before `fit` |
| `core` | Soft-warp, path geometry, dual / terror loss |

```
examples/notebooks/   tutorials
examples/html/        static HTML renders
examples/models/      YAML configs
src/warp_regression/  package
src/data/             lynx.csv, bitcoin_daily.csv
src/tests/            pytest
legacy/               archived notes (not imported)
```

```python
from warp_regression import WarpModel, WarpPath, WarpRegression
from warp_regression import as_forecast_state, forecast_from_state, build_forecast_bands
from warp_regression import prefit, analyze_cycle_lengths
```

---

## Develop

```bash
pytest src/tests/ -v
pytest src/tests/ -v -m slow   # full Bitcoin reproduction

jupyter nbconvert --execute --to notebook --inplace examples/notebooks/*.ipynb
jupyter nbconvert --to html examples/notebooks/*.ipynb --output-dir examples/html/
```
