# warp_regression

Likelihood-based time warping for regression and forecasting on **cyclical** and **warped** series.

Motivation and modelling basis: [`blog_post.md`](blog_post.md).

---

## What is warp regression?

Most ML sticks to **$y$-axis** error. Real timing mistakes are often about *when*: a train ten minutes late, a parcel a day after the promised window. Errors-in-variables already puts noise on inputs; warp regression asks whether **time** error can sit in a regression model explicitly, instead of being dumped into the residual on $y$.

That matters for **warped** or **cyclical** series (irregular phase — not the same as seasonality). Dynamic time warping is common, but it is mainly pattern matching: it does not easily give regression, inference, or forecasts of future timing. The bet here is a **generative** warp: an input series $x$ is deformed by slow expansions and contractions; we score that deformation with its own likelihood (**terror** = *t*iming *error*), and fit it jointly with ordinary observation error.

At each index $i$,

$$
\hat{y}_i = f\big(\mathrm{warp}(x, p)_i\big),
$$

where $x$ is a covariate (often a sine or other known shape), $p$ is a low-dimensional warp path (fractional indices into $x$), and $f$ is linear, an MLP, trend + cycle, …. Soft-warp interpolates $x$ at those indices so $p$ is differentiable.

### Terror (timing error)

Timing is modelled as a **Gaussian random walk** on path offsets — not the only option, but a simple generative way to let the series drift early and late, with scale $\sigma_t$. A free random walk at every index is infeasible (jagged, one parameter per observation). So the fitted path is **piecewise linear through $K \ll n$ knots**. Terror is the **expected log-likelihood of a Brownian bridge** on each knot segment (`expected_likelihood`): $O(K)$, autograd-friendly, and still tied to the RW story used for forecast path sampling.

Training uses a **dual likelihood**:

| Term | Scale | Meaning |
|------|-------|---------|
| **error** | $\sigma_y$ | Usual Gaussian fit of $y$ vs $\hat{y}$ |
| **terror** | $\sigma_t$ | Likelihood on path offsets $p(i) - i$ — how plausible is this timing? |

$$
\mathcal{J} = \lambda \cdot (-\log p(y \mid \hat{y},\sigma_y)) - (1-\lambda)\cdot \log p(p \mid \sigma_t).
$$

| $\lambda$ | Interpretation |
|-----------|----------------|
| $0.5$ | Equal weight on both log-likelihoods — the natural joint (neither term preferred *a priori*) |
| $\to 1$ | Fit only: terror off; the path may warp freely to reduce residual error |
| $\to 0$ | Terror only: timing prior dominates; $y$-fit is barely scored |

Because timing has its own scale, forecasts can continue the warp as a random walk and build:

- **terror bands** — uncertainty from future timing alone  
- **error bands** — observation noise alone  
- **combined bands** — both together  

That is also how you get a distribution over *next cycle length*, not only next height.

---

## Cyclical ≠ seasonal

Seasonality resets every January. A business cycle, predator–prey boom, or Bitcoin’s ~4-year rhythm can run long or short: the *shape* recurs; the *clock* slips. Packaged tooling rarely gives all of: time warping, regression/inference, uncertainty over cycle lengths, and forecasts that sample different warping paths (*what* and *when*). GPs, state-space models, DTW, and DTW-style nets cover pieces of that list; getting all of them usually means a custom research pipeline.

In this repo:

- **Synthetic** — a known sine pushed through a hidden path (grade recovery).  
- **Lynx** — two sines share one warp: both components speed up or slow down together.  
- **Bitcoin** — one macro cycle rides a strong log-trend; the path absorbs early/late peaks.  
- **Fully Bayesian** — same dual geometry with PyMC + JAX / NumPyro posteriors.

---

## Why use this — and when not to

**Benefits**

- Separates shape ($x$, $f$) from timing ($p$, $\sigma_t$).  
- One objective for fit and path plausibility; gradients end to end in PyTorch (JAX for Bayesian workflows).  
- Forecast uncertainty includes *when*, via path sampling (terror / combined bands, cycle-length draws).  
- Works at small-to-medium $n$ with a hand-specified cycle shape (Lynx-scale series are in scope).

**Compared with nearby tools.** Cyclical series (irregular phase, not ordinary seasonality) are hard to handle with packaged tooling. In particular, it is rare to get time warping, a proper regression likelihood, uncertainty over cycle lengths, and forecasts that sample future warp paths in one place. Neighbouring methods cover pieces of that list; getting all four usually means a custom research stack. More detail is in [`legacy/info/comparable_methods.md`](legacy/info/comparable_methods.md).

**DTW / soft-DTW** stretches time to align series, but it is fundamentally pattern matching (or a differentiable alignment loss). It does not give a generative timing law, a $\sigma_t$, or a natural way to sample future warps for regression and forecasting.

**Structural time series** (Harvey UCM, TBATS) do give cycles and Kalman-style probabilistic forecasts. Their cycle frequency is fixed at estimation time: phase can wander, but the period does not. On Lynx they are competitive on point forecasts while producing wider bands; series with two superimposed cycles need TBATS-style multi-seasonality or a hand-rolled state-space extension.

**Gaussian processes** define a prior over functions through a covariance kernel, which gives flexible fits and predictive bands — but that prior is over values of $y$, not over phase warps. Methods that add warping (for example BoTorch's Kumaraswamy `Warp`) are not a warp kernel in that sense: they apply an input map $w(t)$ and then run an ordinary GP in the warped coordinates. You still do not get a generative law over cycle lengths or terror-style forecasts that sample future timing paths.

**Custom GP or monotone-net warps** (latent speed fields, UMNN-style nets, and similar) can do phase-only timing in principle, but they are research DIY rather than one-liners. Warp uncertainty and cycle-length forecasts are extras you assemble on top.

**Neural warping** methods such as DTAN learn monotone warps, but they are built to align ensembles of signals, not to forecast when the next peak of one series will land.

**Deep sequence nets** forecast flexibly, yet they absorb timing into a black box, want a lot of data, and offer no explicit $\sigma_t$ or cycle-length law.

**Use it** when the series is cyclical in the sense above: you have a plausible parametric shape, the clock slips relative to a fixed calendar, and you care about *when* the next peak arrives as much as how high it is. The examples in this repo (synthetic sine, Lynx, Bitcoin) sit in that regime—tens to a few thousand regularly spaced points—where a hand-specified cycle plus a low-dimensional path is still practical.

**Skip it** when there is no shape worth warping (prefer a plain GP or a deep net); when you mainly need mature standard errors on long, high-frequency seasonal series (UCM or TBATS); when the task is aligning many signals rather than forecasting one (DTW or DTAN); when timestamps are irregular (GPs handle that natively; this package assumes a regular index); or when a research-only codebase is a non-starter for production.


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

| Notebook | What it covers |
|----------|----------------|
| [`0_Warp_Block_Basics.ipynb`](examples/notebooks/0_Warp_Block_Basics.ipynb) | Portable `WarpPath` / `WarpRegression` block; error-only vs dual loss on a known warp |
| [`1_Introduction_to_Warp_Regression.ipynb`](examples/notebooks/1_Introduction_to_Warp_Regression.ipynb) | End-to-end `WarpModel` on synthetic sine: prefit, dual fit, path recovery, forecast bands |
| [`2_Adding_complexity_Lynx_Forecast.ipynb`](examples/notebooks/2_Adding_complexity_Lynx_Forecast.ipynb) | Hudson Bay lynx: two sines, one shared warp, nonlinear readout, holdout forecast |
| [`3_Bitcoin_Warp.ipynb`](examples/notebooks/3_Bitcoin_Warp.ipynb) | Daily BTC log-price: log-trend + envelope sine, cycle timing, out-of-sample bands |
| [`4_Fully_Bayesian.ipynb`](examples/notebooks/4_Fully_Bayesian.ipynb) | Fully Bayesian dual model (PyMC + JAX / NumPyro): posteriors on $A$, $C$, path, and scales |

HTML under [`examples/html/`](examples/html/). Configs in [`examples/models/`](examples/models/).

---

## Package

| Piece | Role |
|-------|------|
| `WarpPath` / `WarpRegression` | Portable warp block |
| `WarpModel` | Paths, blocks, observation $f$, dual fit, forecast |
| `observation` | Term kinds for $f$ |
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
