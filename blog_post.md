# Warp Regression: Timing errors, warps, and a generative “terror”

For quite some time I've been interested in the idea of 'time' error, i.e. early, late, or in the time-series context, warped or unwarped. The machine learning field clings strongly to the idea using y-axis error or loss, but it's not the only way to optimize. There is the whole error-in-variables field, for example, that includes x-axis error too. But in real life, we often talk explicitly in terms of time errors: your (probably Deutsche Bahn) train is 10 minutes late; your Amazon package gets to you a day after the expected schedule. I've wondered if I can use this explicitly in a regression context.

This is especially interesting to me in the case of time-series modelling. Warped series or cyclical variables with irregular phase are somewhat common in the real world, but painfully difficult to solve. 

{insert the bitcoin log series as an example image}

Dynamic time warping is common in the field, but it's more of a pattern-matching tool. It doesn't easily support regression models, inference, or forecasting. Can we build something similar that explicitly models the warping as a generative process?*

Therefore, the idea behind this is simple. A given input time series $x$ is deformed in time by some unknown or undescribed source. This deformation takes the form of slow, seemingly random expansions and contractions. What process could describe this non-stationary drift back and forth? I've assumed a Gaussian random walk for this time error. Not because it's the only option, but because it's a simple, generative way to let the series drift early and late. From there we can derive an explicit likelihood for this time error. I'm calling this "terror" (1) because of the obvious contraction and (2) because the derivation did terrorize my modest mathematical abilities.

And while fitting a random walk is infeasible (a free RW wants a value at every time step, so you get a jagged path with roughly one parameter per observation. They're too rough to interpret and too high-dimensional to fit usefully), it's much more tractable to fit a piecewise-linear model approximating the path. From there, we can explicitly derive the expected likelihood between the knots: a Brownian bridge.

From there, while a bit tricky, the machinery works itself out. We have a loss function balancing y-axis and terror likelihood.

{Dual likelihood:}

$$
\hat{y} = f(\mathrm{warp}(x, p))
$$

$$
\mathcal{J}
=
\lambda \cdot (-\log p(y \mid \hat{y}, \sigma_y))
-
(1-\lambda)\cdot \log p(p \mid \sigma_t)
$$

So $\lambda$ trades off ordinary fit (error, scale $\sigma_y$) against timing plausibility (terror, scale $\sigma_t$). At $\lambda = 0.5$ the two terms get equal weight. Toward 1 you mostly fit $y$; toward 0, terror takes over.

The loss function is differentiable and scalable, I've built it in this repo with PyTorch (and included a JAX version for Bayesian workflows).

And more importantly, I think there are genuine real-life use cases for this technique. Time series with cyclical patterns (not seasonal) are very tricky to analyze with out-of-the-box tooling. I do not know of a single packaged tool that can do all of the following for cyclical series:

- Time warping
- Parameterized for regression and inference
- Proper uncertainty over cycle lengths
- Forecasts that sample different warping paths — reflecting not only what will happen, but when

{“Packaged / out-of-the-box” is the claim to defend — not “no paper has ever done a piece of this.”}

Neighbouring methods cover pieces of that list. Gaussian processes, state-space models, DTW, and some DTW-style networks can each do one or more of these things, but to my knowledge getting all of them usually means a highly customized, research-style approach.

{Fair. Input-warped GPs (e.g. BoTorch `Warp`) are packaged but low-capacity; SSMs forecast with fixed seasonal frequencies; DTW aligns rather than sampling future warps. Overlap on pieces, rarely the full list.}
