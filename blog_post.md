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

And more importantly, I think there are genuine real-life use cases for this technique. Time series with cyclical patterns (not seasonal) are very tricky to analyze with out-of-the-box tooling — and so are response shapes whose *duration* wanders (media carry-over, lagged physiological effects, and so on). I do not know of a single packaged tool that can do all of the following for cyclical series:

- Time warping
- Parameterized for regression and inference
- Proper uncertainty over cycle lengths
- Forecasts that sample different warping paths — reflecting not only what will happen, but when

To my knowledge, there are several similar approaches that could do one or more of these things. Gaussian processes, state-space models, DTW, and DTW style Neural Networks can do some of these things, but I think not all, and to my knowledge it requires highly customized, research-style approaches.

To illustrate, I have put together examples which can be found in this repository:

1) A simple simulated, warped sinusoid  
2) The famous Canadian lynx trapping dataset  
3) Analysis of the Bitcoin price  
4) A fully Bayesian implementation of (1)  
5) A marketing-mix style demo: sparse spend → geometric adstock → **pinned** warp of media effects  

That last one is a different flavour of the same idea. Classical MMM already stretches spend with an adstock (carry-over). What it usually does *not* do is let the **duration** of that effect wander as a generative timing process. Here the covariate is the already-adstocked series; pulse onsets stay fixed (`p[i]=i` at each spend start) so campaigns do not slide on the calendar, while the path between onsets can expand or compress. Terror is masked when spend is off, so timing is scored where the driver is active. A dual / Bayesian fit recovers the media amplitude and the warp; terror-scale random-walk bridges around the fitted path then give a distribution over **how long** a unit spend's effect might last — not only how large it is.

So the checklist above is not only for cycles. Anywhere a known shape is stretched in time — a media decay, a physiological response, a logistics lag — a generative warp plus terror is a way to put *when* into the likelihood instead of burying it in the residual.