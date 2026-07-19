# Comparable Methods to Warp Regression

The README's `## How this compares` table gives a compact, side-by-side view of warp regression against seven
other approaches that overlap with it in some way: they warp time, they model cycles, or they give you a
probabilistic forecast. A table cell can only say so much, so this note walks through each of the seven in more
detail: what it actually does, the key formula, where to find it, and an honest paragraph on how it overlaps
with and diverges from warp regression.

## Sequence alignment: Dynamic Time Warping (DTW) and soft-DTW

### Core idea

DTW measures the discrepancy between two time series of possibly different lengths by finding the alignment
(a monotone, non-decreasing correspondence between indices in one series and indices in the other) that
minimises the total elementwise cost. It does not model a generative process for *why* one series' timing
differs from the other's; it searches a discrete space of possible alignments, one path through an $n \times m$
grid, for a cost minimum via dynamic programming.

### Formulation

Given sequences $x_{1:n}$, $y_{1:m}$ and a pointwise cost $d(x_i, y_j)$ (e.g. squared distance), the cumulative
alignment cost is defined recursively as

$$
\gamma(i,j) = d(x_i, y_j) + \min\big\{\gamma(i-1,j),\ \gamma(i,j-1),\ \gamma(i-1,j-1)\big\},
$$

with $\mathrm{DTW}(x,y) = \gamma(n,m)$. This `min` makes the classic recursion non-differentiable. Soft-DTW
(Cuturi & Blondel, 2017) replaces the hard `min` with a soft-minimum,

$$
\min\nolimits^{\gamma}\{a_1,\dots,a_k\} = -\gamma \log \sum_i e^{-a_i/\gamma},
$$

so the whole recursion, and its gradient, can be computed in closed form via dynamic programming, letting
soft-DTW be used directly as a differentiable training loss for any model that outputs a sequence.

### References

- Sakoe & Chiba, "Dynamic Programming Algorithm Optimization for Spoken Word Recognition" (1978), the
  original DTW algorithm.
- Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for Time-Series", ICML 2017.

### Tooling

`tslearn`, `dtaidistance`, `fastdtw` for classic/soft DTW in Python; `sdtw-cuda-torch` for a GPU-accelerated,
memory-efficient soft-DTW PyTorch loss.

### Overlap and divergence with warp regression

Both DTW and warp regression treat the time axis as something to be nonlinearly stretched or compressed rather
than fixed. But DTW's warp is the *output* of a one-shot optimisation against a specific target sequence (or,
for soft-DTW, a per-example training loss), not a *parameter with a prior and a posterior*. There is no analog
of $\sigma_t$: nothing stops DTW's alignment from being maximally exotic if that is what the optimiser finds,
and there is no generative story for what an unseen future alignment should look like. That is exactly why DTW
shows up in classification, clustering, and retrieval far more often than in forecasting with a confidence
interval; see also the README's own "Why not just dynamic time warping?" section, which makes the same point
about regularisation, uncertainty, and joint inference.

## Structural time series: Harvey's Unobserved Components Model (UCM) and TBATS

### Core idea

Harvey's structural time series model (1989) decomposes an observed series into additive unobserved
components, each following its own stochastic state-space recursion: a level (or trend), a stochastic cycle,
and irregular noise. The stochastic cycle is a damped, rotating AR(2)-like recursion with a fixed frequency and
its own disturbance variance, so the cycle's *phase* wanders even though its *frequency* is fixed at
estimation time. Everything is estimated jointly by maximum likelihood via the Kalman filter, which also gives
the forecast distribution for free (state covariance propagated forward analytically). TBATS (De Livera,
Hyndman & Snyder, 2011) is a related, more seasonally flexible cousin: it stacks a Box-Cox transform, ARMA
errors, multiple trigonometric (Fourier) seasonal terms, and a trend, so it natively handles several
superimposed periodicities where the vanilla UCM's `cycle=True` option only gives you one.

### Formulation

The stochastic cycle component $(\psi_t,\ \psi_t^*)$ follows

$$
\begin{pmatrix} \psi_t \\ \psi_t^* \end{pmatrix}
= \rho
\begin{pmatrix} \cos\lambda_c & \sin\lambda_c \\ -\sin\lambda_c & \cos\lambda_c \end{pmatrix}
\begin{pmatrix} \psi_{t-1} \\ \psi_{t-1}^* \end{pmatrix}
+ \begin{pmatrix} \kappa_t \\ \kappa_t^* \end{pmatrix},
$$

where $\rho \in (0,1]$ is a damping factor, $\lambda_c$ the cycle frequency, and $\kappa_t, \kappa_t^*$ i.i.d.
Gaussian disturbances; the observation equation adds a level/trend and irregular term on top,
$y_t = \mu_t + \psi_t + \varepsilon_t$.

### References

- Harvey, *Forecasting, Structural Time Series Models and the Kalman Filter* (1989).
- Harvey & Jaeger, "Detrending, Stylized Facts and the Business Cycle" (1993).
- De Livera, Hyndman & Snyder, "Forecasting Time Series with Complex Seasonal Patterns Using Exponential
  Smoothing" (2011), the TBATS paper.

### Tooling

`statsmodels.tsa.statespace.structural.UnobservedComponents` (Python), R's `KFAS` and `bsts` packages,
`tbats`/`sktime` for TBATS.

### Overlap and divergence with warp regression

This is the model actually benchmarked head-to-head against warp regression in `src/notebooks/LynxForecast.ipynb`
(Step 7). Both give a generative account of a wandering cyclical phase with propagated uncertainty, and both
are honest about explicit variance parameters. The difference is what is allowed to move: warp regression pins
down the cycle's *shape* (a specific sine, or sum of sines, with fixed amplitude) and lets only its *timing*
wander via the warp path, while the UCM's stochastic cycle lets both phase *and* amplitude wander via its own
AR(2)-like recursion, and (in the vanilla `UnobservedComponents` class) is capped at a single cycle
frequency, so a series like Lynx with two superimposed cycles needs either TBATS's multiple seasonal terms or
a hand-rolled multi-cycle state-space extension. In the actual Lynx comparison this shows up as similar
point-forecast accuracy but a visibly wider predictive interval from the UCM at the same nominal coverage,
because its cycle (and, in a random-walk-level specification, its level too) compounds uncertainty over the
forecast horizon in a way that is less constrained than warp regression's fixed-shape assumption.

## Neural time warping: Diffeomorphic Temporal Alignment Nets (DTAN / RF-DTAN)

### Core idea

DTAN (Shapira Weber, Katz & Freifeld, NeurIPS 2019) is a learning-based method for jointly aligning an
ensemble of time series. A neural network (a "temporal transformer" layer, structurally similar to a spatial
transformer network but for one dimension) takes a signal as input and predicts a smooth, invertible
(diffeomorphic) warp, which is then applied to align the signal with the rest of the ensemble. Because the
warp is a function of the input signal itself, rather than a bespoke optimisation solved per pair, a trained
DTAN aligns previously-unseen signals with a single cheap forward pass, unlike DTW which re-solves an
alignment problem from scratch for every new pair. The original formulation needed a hand-tuned regulariser to
stop the network from degenerately collapsing signals onto each other; the follow-up RF-DTAN (Shapira Weber et
al., ICML 2023) replaces that regulariser with an "Inverse Consistency Averaging Error" loss and also supports
variable-length input signals.

### Formulation

The warp is parameterised as the flow of a continuous-time velocity field predicted by the network, integrated
to guarantee a smooth, monotone (diffeomorphic) map $\tau_\theta : [0,1] \to [0,1]$. The aligned signal is
$x(\tau_\theta(t))$, and $\theta$ (the network's weights) is trained by minimising an unsupervised alignment
loss (within-class sum of squares, or the newer ICAE loss) over the whole ensemble, not a per-signal
likelihood.

### References

- Shapira Weber, Katz & Freifeld, "Diffeomorphic Temporal Alignment Nets", NeurIPS 2019.
- Shapira Weber et al., "Regularization-free Diffeomorphic Temporal Alignment Nets", ICML 2023.

### Tooling

Reference PyTorch implementations at `github.com/BGU-CS-VIL/dtan` and `github.com/BGU-CS-VIL/RF-DTAN`; not
packaged as a general-purpose library.

### Overlap and divergence with warp regression

This is the closest thing to a "warping neural network" in the literature, and the most direct technical
cousin of warp regression's soft-warped path: both use a smooth, differentiable, monotone warp function
trained end-to-end in PyTorch by gradient descent. The difference is the objective the warp serves. DTAN's
warp exists to align an *ensemble* of signals to each other (or to a learned template) for downstream
averaging or classification; there is no likelihood over future values and no forecast. Warp regression's warp
exists to explain a *single* series' timing against a known reference shape, scored by an explicit dual
likelihood that also assigns a probability to any candidate future path, which is what makes forecasting and
cycle-length distributions possible in the first place.

## Curve registration: Elastic Functional Data Analysis (SRVF / Fisher-Rao)

### Core idea

Functional data analysis treats each observed curve as a sample from a space of functions, and elastic curve
registration separates two sources of variability that a naive pointwise comparison conflates: *phase* (when
features occur) and *amplitude* (how big they are). The key technical trick is the Square-Root Velocity
Function (SRVF) representation, which turns the Fisher-Rao metric (a proper, warping-invariant distance
between functions) into an ordinary $L^2$ distance, so finding the optimal warp between two curves becomes
dynamic programming over a Hilbert sphere rather than an intractable variational problem. Curves are jointly
registered to a Karcher mean template by iteratively estimating a warp for each curve and refitting the
template.

### Formulation

For a curve $f$ define its SRVF $q(t) = \dot f(t)/\sqrt{|\dot f(t)|}$. For two curves $f_1, f_2$ with SRVFs
$q_1, q_2$, and a warp $\gamma$, a monotone reparameterisation of $[0,1]$ with $\gamma(0)=0,\ \gamma(1)=1$, the
elastic (Fisher-Rao) distance is

$$
d_{FR}(f_1, f_2) = \min_{\gamma} \Big\| q_1 - (q_2 \circ \gamma)\sqrt{\dot\gamma}\, \Big\|_{L^2},
$$

and the minimising $\hat\gamma$ is the estimated warping function used to align $f_2$ onto $f_1$.

### References

- Srivastava, Wu, Kurtek, Klassen & Marron, "Registration of Functional Data Using Fisher-Rao Metric" (2011).
- Srivastava, Klassen, Joshi & Jermyn, "Shape Analysis of Elastic Curves in Euclidean Spaces" (2010).

### Tooling

`fdasrvf` (R and Python), `scikit-fda`, `tidyfun`/`fdars` (R).

### Overlap and divergence with warp regression

Both frameworks name the same underlying idea: separating a stable shape from a time-varying
reparameterisation. Both insist the warp be smooth and monotone. Elastic FDA is primarily a *metric and
estimation* framework: it gives a proper distance between curves and a template, with classical
hypothesis-testing machinery built on that distance, but each curve's warp is fit by an independent
dynamic-programming optimisation rather than jointly with a generative forecast model, and there is no notion
of extrapolating a warp path into the future or attaching a probability to how far the next cycle's timing
might drift. Warp regression borrows the same shape/timing split but commits to a stochastic model of the warp
path itself (the terror likelihood), which is what lets it forecast rather than only register already-observed
curves.

## Gaussian process regression: Warped Gaussian Processes

### Core idea

A Gaussian process places a prior directly over functions, defined by a mean function and a covariance
(kernel) function; conditioning on observed data gives a full posterior distribution over functions, including
calibrated uncertainty, with no explicit parametric formula for the underlying signal. "Warping" enters a GP in
two different, complementary ways: *output* warping (Snelson, Rasmussen & Ghahramani, 2003) applies a learned
monotone transform to the target variable before modelling it with a standard GP, letting the model discover
something like a log-transform automatically; *input* warping instead transforms the input (time) axis before
computing the kernel, which is the more direct analog of warp regression's approach, letting a locally
nonstationary, non-uniformly periodic process be modelled with an otherwise-stationary kernel in the warped
time coordinate. In practice, most published wandering-phase periodic GPs (the quasi-periodic kernels used for
stellar-rotation light curves, for instance) skip explicit input warping altogether and instead mix a periodic
covariance term with a separate decay or noise term directly in the output covariance. That trades an explicit,
interpretable warp path for closed-form, exact inference.

### Formulation

For an input warp $u(t)$ and a stationary kernel $k$, an input-warped GP models

$$
y(t) = f(u(t)) + \varepsilon, \qquad f \sim \mathcal{GP}\big(0,\ k(u(t), u(t'))\big),
$$

where $u(\cdot)$ is often itself given a GP or monotonic-neural-net prior (as in sparse-spectrum warped input
measures), letting the effective "local frequency" of a periodic kernel vary smoothly over time. A plain GP
prior on $u(\cdot)$ does not by itself guarantee $u$ is monotone increasing; practical constructions either
restrict $u$ to a monotonic parametric family (e.g. Snelson's chained tanh layers) or place the prior on
$\log \dot u(t)$ instead, so that $u(t) = \int_0^t e^{g(s)}\,ds$ is monotone by construction.

### References

- Snelson, Rasmussen & Ghahramani, "Warped Gaussian Processes", NeurIPS 2003.
- Vinokur & Tolpin, "Warped Input Gaussian Processes for Time Series Forecasting" (2021).
- Shen et al., "Sparse Spectrum Warped Input Measures for Nonstationary Kernel Learning", NeurIPS 2020.

### Tooling

`GPyTorch`, `GPflow`, `scikit-learn` (`GaussianProcessRegressor`), `george`.

### Overlap and divergence with warp regression

Warped GPs and warp regression share the instinct that "warp the time axis, then apply a simpler model" is
more tractable than hand-designing a single nonstationary function directly, and both are fully differentiable
and Bayesian-compatible in modern implementations (GPyTorch). The GP's advantage is that its warp and its
function are both nonparametric, so it commits to no fixed shape at all, and it handles irregular or missing
timestamps natively since the kernel is defined pointwise rather than on a fixed grid. Its disadvantage for
this repo's use case is that a GP's warp, however it is built, gives no explicit, interpretable object for
reading "how long until the next peak" off directly: that has to be recovered after the fact by sampling the
posterior and running peak detection on the result, the same limitation deep sequence models have. Exact
inference is also $O(n^3)$ for a generic kernel, uncomfortable for a long daily series like Bitcoin; kernels
with a finite-dimensional state-space representation (Matern, or the stochastically-driven-oscillator kernels
used for wandering-phase stellar rotation) recover exact $O(n)$ inference the same way Harvey's Kalman filter
does, but that efficiency and an explicit warp path are not something mainstream tooling offers at the same
time.

## Regime-switching: Markov-switching models

### Core idea

Rather than warping continuous time, Hamilton's Markov-switching model (1989) supposes the series is generated
by one of a small number of discrete regimes (e.g. "expansion" vs. "recession", or "boom" vs. "bust") at each
time step, with regime-specific dynamics (different means, variances, or even different AR coefficients), and
an unobserved Markov chain governing when the regime switches. A "cycle" here is not a warped continuous
waveform; it is the sequence of regime durations implied by the chain's transition probabilities, which are
themselves geometrically distributed under a first-order Markov chain. Informally, a regime-switching model
does produce a "how long until the next switch" distribution, just via a completely different mechanism than
warp regression's continuous path.

### Formulation

With a latent state $S_t \in \{1, \dots, K\}$ following a Markov chain with transition matrix
$P_{ij} = \Pr(S_t = j \mid S_{t-1} = i)$, and regime-conditional dynamics
$y_t \mid S_t = k \sim \mathcal N(\mu_k, \sigma_k^2)$ (or a regime-conditional AR process), the model is
estimated by maximising the marginal likelihood

$$
\mathcal L = \sum_{S_{1:T}} \Pr(y_{1:T},\, S_{1:T}),
$$

computed efficiently via the Hamilton filter, a discrete analog of the Kalman filter that recursively updates
the filtered regime probabilities $\Pr(S_t \mid y_{1:t})$.

### References

- Hamilton, "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle"
  (1989).
- Kim & Nelson, *State-Space Models with Regime Switching* (1999).

### Tooling

`statsmodels.tsa.regime_switching.markov_regression` / `markov_autoregression` (Python), R's `MSwM` package.

### Overlap and divergence with warp regression

Both models answer a version of "when does the next episode start", but from opposite ends. Warp regression
assumes the cyclical *shape* is known and lets its *timing* drift continuously; Markov-switching assumes
discrete, qualitatively different *states* and lets the *sequence of switches* between them be stochastic,
with no commitment to what a cycle looks like inside a given regime beyond its mean and variance.
Markov-switching is the more natural choice when the phenomenon really is "on/off" (e.g. NBER recession
dating) rather than a continuously wandering periodic waveform, and it comes with decades of classical and
Bayesian estimation machinery. It is a poor fit when the series has a genuinely smooth, wave-like shape (like
the lynx population cycle or Bitcoin's price cycle) that a two- or three-state regime model would only crudely
approximate.

## Deep sequence models: LSTM/xLSTM, Transformers, DeepAR, N-BEATS

### Core idea

Rather than committing to any explicit cyclical or warping structure, this family throws a large, flexible
function approximator (recurrent networks like LSTM/xLSTM, attention-based Transformers, or specialised
forecasting architectures like DeepAR and N-BEATS) at the raw sequence and lets gradient descent discover
whatever temporal structure is there, periodic or otherwise, given enough data. Recent architectures often add
domain-specific inductive biases back in (N-BEATS's trend/seasonality basis functions, the Temporal Fusion
Transformer's static/dynamic covariate split, DeepAR's autoregressive likelihood head) without ever explicitly
warping the time axis.

### Formulation

No single equation applies across the family, but the common thread is an autoregressive or
sequence-to-sequence factorisation

$$
p(y_{t+1:t+H} \mid y_{1:t}) = \prod_{h=1}^{H} p\big(y_{t+h} \mid y_{1:t},\ \hat y_{t+1:t+h-1};\ \theta\big),
$$

with $\theta$ a large neural network trained by (stochastic) gradient descent on a likelihood or quantile loss.

### References

- Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2020).
- Oreshkin et al., "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting" (2020).
- Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2021).

### Tooling

`GluonTS`, `PyTorch Forecasting`, `Darts`, `NeuralForecast`, Hugging Face `transformers` /
`time-series-transformers`.

### Overlap and divergence with warp regression

Deep sequence models are the most flexible entry in this comparison and, like warp regression, are natively
differentiable and PyTorch/JAX-friendly. Their weakness relative to warp regression on this repo's target
problem is exactly their strength elsewhere: with enough data they can approximate structure no one had to
specify by hand, but that means they need far more data than Lynx's 90 annual points to reliably learn a
periodic-but-drifting pattern, their parameters carry no interpretable meaning (no analog of $\sigma_t$, no
directly-readable cycle period), and getting calibrated uncertainty (rather than a single point forecast)
requires bolting on a quantile loss, MC dropout, or a deep ensemble rather than getting it for free from a
likelihood.

---

See the README's `## How this compares` table for a condensed, side-by-side summary of these same seven
methods against warp regression across fourteen properties.
