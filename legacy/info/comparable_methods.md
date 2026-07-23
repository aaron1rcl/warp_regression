# Comparable Methods to Warp Regression

The README section **Why use this — and when not to** summarises the comparison. This note goes deeper: what each neighbour actually does, where the overlap is real, and where the gap shows up in this repo’s notebooks and in the modelling checklist from [`blog_post.md`](../../blog_post.md).

## The checklist (what “all of it” means)

For cyclical (not merely seasonal) series, the package aims at four things together:

1. **Time warping** — a monotone map that stretches/compresses a known shape  
2. **Regression / inference** — parameters in a likelihood, not a one-shot alignment  
3. **Uncertainty over cycle lengths** — a distribution for “how long until the next peak?”  
4. **Forecasts that sample future warps** — bands that answer *when* as well as *what*

Neighbouring methods cover **pieces** of that list. Packaged tools rarely cover all four without a custom research stack. Empirically in this repo:
---

## Sequence alignment: Dynamic Time Warping (DTW) and soft-DTW

### Core idea

DTW finds a monotone alignment between two series that minimises total pointwise cost. It does **not** model *why* timing differs; it searches a discrete path through an $n\times m$ grid via dynamic programming. Soft-DTW (Cuturi & Blondel, 2017) replaces the hard `min` with a soft-minimum so the cost is differentiable and usable as a training loss.

### Overlap and divergence

Both stretch the time axis. DTW’s warp is the *output* of an optimisation against a target sequence (or a per-example loss), not a *parameter with a prior*. There is no $\sigma_t$, nothing that regularises exotic alignments, and no generative story for an unseen future path. That is why DTW dominates classification / clustering / retrieval and is a poor substitute for regression + cycle-length forecasts. Soft-DTW improves trainability; it does not add terror-style path sampling.

### Tooling

`tslearn`, `dtaidistance`, `fastdtw`; `sdtw-cuda-torch` for soft-DTW in PyTorch.

---

## Structural time series: Harvey UCM and TBATS

### Core idea

Harvey’s unobserved-components model decomposes $y$ into level/trend, a **stochastic cycle**, and irregular noise, estimated by ML via the Kalman filter. The cycle is a damped rotating recursion at a **fixed frequency** $\lambda_c$: phase wanders through process noise; the period itself does not. TBATS stacks Box–Cox, ARMA errors, and multiple trigonometric seasonal terms for richer fixed-period seasonality.

### Formulation (stochastic cycle)

$$
\begin{pmatrix} \psi_t \\ \psi_t^* \end{pmatrix}
=
\rho
\begin{pmatrix} \cos\lambda_c & \sin\lambda_c \\ -\sin\lambda_c & \cos\lambda_c \end{pmatrix}
\begin{pmatrix} \psi_{t-1} \\ \psi_{t-1}^* \end{pmatrix}
+
\begin{pmatrix} \kappa_t \\ \kappa_t^* \end{pmatrix},
\qquad
y_t = \mu_t + \psi_t + \varepsilon_t.
$$

### What we saw on Lynx

In notebook 2, warp regression (two sines, shared path, optional nonlinear $f$) is compared to `statsmodels` `UnobservedComponents` (fixed and RW level). Rough pattern:

- **Point forecasts** can be in the same ballpark.  
- **UCM predictive intervals** tend to be wider at the same nominal coverage — the stochastic cycle (and RW level, if used) compounds uncertainty without the fixed-shape constraint.  
- Vanilla UCM is **one cycle frequency**; Lynx’s two components need TBATS-style multi-seasonality or a hand-rolled multi-cycle SSM.  
- Linear-Gaussian UCMs often want a `log1p` (or similar) transform on right-skewed counts; a nonlinear warp readout can stay on the raw scale.

### Overlap and divergence

Both give generative cyclical dynamics with propagated forecast uncertainty. Warp regression pins **shape** and lets **timing** move through $p$ and $\sigma_t$ (so cycle *length* is an explicit sampling target). UCM/TBATS pin **frequency** and let phase/amplitude wander inside the state; “how long is the next cycle?” is not a first-class path draw.

### Tooling

`statsmodels.tsa.statespace.structural.UnobservedComponents`; R `KFAS` / `bsts`; `tbats` / `sktime` for TBATS.

---

## Gaussian processes and input warping

### Core idea

A Gaussian process places a prior over functions through a mean and a covariance (kernel). That prior is over values of $y$ (or a latent function of calendar time) — **not** over phase warps. Some toolkits add warping on top: BoTorch's `Warp`, for example, learns a low-capacity Kumaraswamy CDF map $w(t)$ and then runs an ordinary GP in the warped input coordinates. That is input warping plus a standard kernel, not a “warp kernel,” and it does not give a generative law over cycle lengths or terror-style forecasts that sample future timing paths.

### Overlap and divergence

Shared instinct: change the time coordinate, then use a simpler model. GPs win on irregular timestamps and nonparametric flexibility. For this repo’s checklist they miss generative path uncertainty ($\sigma_t$, Brownian-bridge terror, cycle-length Monte Carlo) unless you leave “packaged” territory.

### Tooling

`GPyTorch`, `GPflow`, BoTorch `Warp` + `SingleTaskGP`, `scikit-learn` GPR.

---

## Custom GP / monotone-net / Neural-ODE warps

### Core idea

Research constructions go beyond Kumaraswamy: latent GP **speed fields** $g(t)=\int \mathrm{softplus}(f)$, monotonic neural nets (UMNN-style), Neural-ODE flows, etc. In principle these can implement rich phase-only timing $\mu(t)=A\sin(2\pi\omega g(t)+\psi)+C$.

### Overlap and divergence

Closest *mathematical* cousins to an expressive $p$. In practice they are **not off-the-shelf**: slower fits, easy overfitting, and warp uncertainty / cycle forecasts are something you assemble yourself (Laplace on a latent field, ensembles, …). That matches the blog’s point: pieces exist; the full checklist usually means a custom stack. This package’s bet is a **low-dimensional PWL path + terror** that stays trainable at Lynx/Bitcoin scale and makes path sampling the default forecast story.

---

## Neural time warping: DTAN / RF-DTAN

### Core idea

A network predicts a smooth diffeomorphic warp to align an **ensemble** of series (temporal transformer / flow). Trained for within-class alignment, not for a single series’ likelihood or forecast.

### Overlap and divergence

Same family of smooth monotone warps, different job. DTAN aligns many signals for averaging or classification; warp regression explains one series against a reference shape and scores future paths. No $\sigma_t$, no cycle-length forecast.

### Tooling

Reference repos (`BGU-CS-VIL/dtan`, `RF-DTAN`); not a general forecasting library.

---

## Curve registration: Elastic FDA (SRVF / Fisher–Rao)

### Core idea

Separate phase and amplitude variability across a sample of curves; register to a Karcher mean via SRVF / Fisher–Rao geometry.

### Overlap and divergence

Same shape-vs-timing split for *already observed* curves. Warps are per-curve dynamic programs, not a generative path with a future. Excellent for metrics and template estimation; not a substitute for path-sampled forecasting.

### Tooling

`fdasrvf`, `scikit-fda`.

---

## Regime-switching: Markov-switching models

### Core idea

Discrete latent regimes with Markov transitions; “cycle length” is a sojourn-time distribution, not a warped waveform.

### Overlap and divergence

Answers a version of “when does the next episode start?” for on/off phenomena (e.g. recessions). A poor match to smooth lynx/Bitcoin-style waves that want a continuous path, not a handful of means/variances.

### Tooling

`statsmodels` Markov regression / autoregression; R `MSwM`.

---

## Deep sequence models: LSTM / Transformers / DeepAR / N-BEATS

### Core idea

Large sequence models learn temporal structure from data with minimal hand-specified cycles. Probabilistic heads (DeepAR, quantile losses) give forecast distributions without an explicit warp.

### Overlap and divergence

Maximum flexibility, and PyTorch-native. On Lynx-scale $n$ they are usually the wrong inductive bias: no readable $\sigma_t$, no first-class cycle period, and timing is entangled in latent states. Prefer them when there is **no** parametric shape to warp and data are plentiful.

### Tooling

`GluonTS`, `PyTorch Forecasting`, `Darts`, `NeuralForecast`.

---

## Summary

| Method | Warping | Regression likelihood | Cycle-length law | Path-sampled forecasts |
|--------|---------|----------------------|------------------|------------------------|
| Warp regression (this repo) | PWL path + soft-warp | Dual error + terror | Yes ($\sigma_t$ path draws) | Yes |
| DTW / soft-DTW | Alignment path | Loss / distance | No | No |
| Harvey UCM / TBATS | Fixed $\lambda_c$ cycle | State-space ML | Indirect (state noise) | Kalman, not path-RW |
| BoTorch `Warp` + GP | Low-capacity $w(t)$ | GP MLL | Not without DIY | Evaluate MAP $w$, not RW paths |
| Custom GP / mono-net warps | Rich $g(t)$ possible | If you build it | DIY | DIY |
| DTAN | Ensemble warp | Alignment loss | No | No |
| Elastic FDA | Registration $\gamma$ | Metric / template | No | No |
| Markov-switching | Discrete sojourns | Regime likelihood | Geometric-like | Regime probs |
| Deep sequence nets | Implicit | Usually yes | No explicit | Quantiles / sampling, not path-RW |

See the README table for the short version; notebooks 2, 5, and 6 for the empirical probes.
