# Warp Regression: Timing errors, warps, and a generative “terror”

For quite some time I've been interested in the idea of 'time' error, i.e. early, late, or in the time-series context, warped or unwarped. The machine learning field clings strongly to the idea using y-axis error or loss, but it's not the only way to optimize. There is the whole error-in-variables field, for example, that includes x-axis error too. But in real life, we often talk explicitly in terms of time errors: your (probably Deutsche Bahn) train is 10 minutes late; your Amazon package gets to you a day after the expected schedule. I've wondered if I can use this explicitly in a regression context.

This is especially interesting to me in the case of time-series modelling. Warped series or cyclical variables with irregular phase are somewhat common in the real world, but painfully difficult to solve. 

{insert the bitcoin log series as an example image}

Dynamic time warping is common in the field, but it's more of a pattern-matching tool. It doesn't easily support regression models, inference, or forecasting. Can we build something similar that explicitly models the warping as a generative process?*

Therefore, the idea behind this is simple. A given input time series `x` is deformed in time by some unknown or undescribed source. This deformation takes the form of slow, seemingly random expansions and contractions. What process could describe this non-stationary drift back and forth? I've assumed a Gaussian random walk for this time error. Not because it's the only option, but because it's a simple, generative way to let the series drift early and late. From there we can derive an explicit likelihood for this time error. I'm calling this "terror" (1) because of the obvious contraction and (2) because the derivation did terrorize my modest mathematical abilities.

And while fitting a random walk is infeasible (a free RW wants a value at every time step, so you get a jagged path with roughly one parameter per observation. They're too rough to interpret and too high-dimensional to fit usefully), it's much more tractable to fit a piecewise-linear model approximating the path. From there, we can explicitly derive the expected likelihood between the knots: a Brownian bridge.

From there, while a bit tricky, the machinery works itself out. We have a loss function balancing y-axis and terror likelihood.

**Dual likelihood:**

```text
ŷ = f(warp(x, p))

J = λ · (−log p(y | ŷ, σ_y)) − (1 − λ) · log p(p | σ_t)
```

So `λ` trades off ordinary fit (error, scale `σ_y`) against timing plausibility (terror, scale `σ_t`). At `λ = 0.5` the two terms get equal weight. Toward 1 you mostly fit `y`; toward 0, terror takes over.

The loss function is differentiable and scalable, I've built it in this repo with PyTorch (and included a JAX version for Bayesian workflows).

And more importantly, I think there are genuine real-life use cases for this technique. Time series with cyclical patterns (not seasonal) are very tricky to analyze with out-of-the-box tooling — and so are response shapes whose *duration* wanders (media carry-over, lagged physiological effects, and so on). I do not know of a single packaged tool that can do all of the following for cyclical series:

- Time warping
- Parameterized for regression and inference
- Proper uncertainty over cycle lengths
- Forecasts that sample different warping paths — reflecting not only what will happen, but when

To my knowledge, there are several similar approaches that could do one or more of these things. Gaussian processes, state-space models, DTW, and DTW style Neural Networks can do some of these things, but I think not all, and to my knowledge it requires highly customized, research-style approaches.

To illustrate, I have put together five examples in this repository. Each one is the same dual idea — shape on `x`, timing on `p` — in a different setting.

**1) Simulated warped sinusoid.**  
Start with a known sine, push it through a hidden warp path, add noise, and try to recover the path and amplitude. This is the grade-school test of the method: if you cannot get the timing back when you built the truth yourself, nothing else is worth trusting. Prefit finds a rough `A` / `C`, dual fit balances residual error against terror, and you can check that the recovered path tracks the true one. Forecasts continue the path as a random walk so the bands reflect *when* the next peak might land, not only how high.

**2) Canadian lynx.**  
The Hudson Bay trapping series is the classic irregular boom–bust cycle: the *shape* of the oscillation repeats, but the period stretches and compresses. Here two sine covariates share **one** warp — both components speed up or slow down together — with a small nonlinear readout on top. That shared-path constraint is the modelling claim: one biological clock, two harmonic pieces. Holdout forecasts sample future warps so uncertainty covers the next peak’s height *and* its calendar position.

**3) Bitcoin.**  
Daily log-price sits on a strong secular trend with a rough multi-year rhythm on top. The model is log-trend plus an envelope sine; the warp absorbs early and late cycle peaks instead of forcing a fixed ~4-year metronome. Out-of-sample bands again mix observation noise with terror-sampled timing, which is the point for a series where “the next top” is as much a *when* question as a *what* question.

**4) Fully Bayesian version of (1).**  
Same synthetic warped sinusoid and the same dual geometry, but sampled with PyMC + JAX / NumPyro instead of a point-estimate optimizer. You get posteriors on amplitude, intercept, path knots, and the two scales (`σ_y`, `σ_t`) — so path uncertainty is not a bootstrap afterthought but part of the fit. Useful when you care about credible intervals on timing, or when you want to check that the dual likelihood is a coherent Bayesian model and not only an optimization trick.

**5) Marketing-mix style: warped adstock.**  
Classical MMM already stretches spend with an adstock (carry-over). What it usually does *not* do is let the **duration** of that effect wander as a generative timing process. Here the covariate is the already-adstocked series; pulse onsets stay fixed (`p[i]=i` at each spend start) so campaigns do not slide on the calendar, while the path between onsets can expand or compress. Terror is masked when spend is off, so timing is scored where the driver is active. A dual / Bayesian fit recovers the media amplitude and the warp; terror-scale random-walk bridges around the fitted path then give a distribution over **how long** a unit spend's effect might last — not only how large it is.

So the checklist above is not only for cycles. Anywhere a known shape is stretched in time — a media decay, a physiological response, a logistics lag — a generative warp plus terror is a way to put *when* into the likelihood instead of burying it in the residual.