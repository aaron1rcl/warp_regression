# Soft Warping vs. Discrete Warping

The original research prototype (`legacy/warp.py` and `legacy/warp_network.py`) implemented warping as a
**discrete index shift**: each row of data got moved to a nearby integer column of an oversampled grid. It
worked, sort of, but it was the wrong data structure for a model you want to train with `loss.backward()`.
Switching to **soft warping**, i.e. reading the driver at a continuous fractional index via linear
interpolation, is what actually made the whole thing trainable end-to-end in torch. This note walks through
both approaches in detail, with the specific functions involved on each side.

## The problem with discrete warping

### Attempt 1: hard integer shift (`legacy/warp.py`)

The very first version moved data by literally re-indexing it. `X_shift` takes a diagonal matrix `X` (built
by `matrix_decomp`, one column per time step, values on the diagonal and `NaN` elsewhere) and, for every row,
calls `shift(x_r, int(p[i].item()))`:

```python
def X_shift(X: torch.Tensor, sr: int, p: torch.Tensor):
    ...
    for i in range(1, int(X.shape[0]/sr)):
        x_r = X_out[i*sr, :]
        x_shift = shift(x_r, int(p[i].item()))
        X_out[i*sr, :] = x_shift
    return X_out
```

Two things kill differentiability here outright:

- `p[i].item()` pulls a Python float out of the autograd graph. From that point on, `p` is just a number
  used to pick an integer offset; there is no tensor operation left for `backward()` to walk through.
- `int(...)` rounds to the nearest column. Even if the `.item()` call weren't there, rounding is a
  discontinuous, zero-gradient-almost-everywhere operation, i.e. exactly what you'd expect from "which bucket
  does this fall into" logic.

So `warp.py`'s discrete warp is a **numpy-style reindexing operation wearing a torch.Tensor costume**. It
runs, but you cannot backprop a warp path through it. This is fine for a one-off "shift the series and look
at the plot" script, but useless for jointly learning a warp path by gradient descent.

### Attempt 2: soft-looking, still hard in practice (`legacy/warp_network.py`)

`ShiftNetwork.create_shift_matrix` is a more serious effort to make the shift differentiable. It builds a
similarity matrix between "output position" and "shifted source position" and uses it as a soft gather:

```python
def create_shift_matrix(self, p, n_max, tau=10, clamp_max=1000, pad=False):
    pp = p + torch.arange(p.shape[0]).unsqueeze(1)
    POS = torch.tensor(np.arange(p.shape[0])).repeat(p.shape[0], 1)
    POS = POS[::self.sr, ::]
    pp = pp[::self.sr]
    power_ = (torch.abs(POS - pp) + 0.5)
    for i in range(tau):
        power_ = power_**2
        power_ = torch.clamp(power_, max=clamp_max)
    P = torch.exp(-1*(power_))
    return P
```

The idea is reasonable: `P[j, k]` should be large when output row `j` should read from source column `k`,
and near zero otherwise, so `x_shift = P * B` (an elementwise multiply against the broadcast input, summed
with `torch.nansum`) acts like a soft, differentiable gather. In principle gradients can flow through `P`
back into `p`.

In practice this falls apart for a few reasons:

- **The kernel is sharpened until it's basically one-hot.** Squaring `power_` fifteen times (`tau=15` is used
  at the call site) turns any nonzero distance into a huge number almost immediately, so `exp(-power_)` is
  either ≈1 at the matching column or ≈0 everywhere else. That's a hard argmax wearing a soft mask. The
  gradient of a near-one-hot bump function is close to zero everywhere except in an infinitesimally narrow
  region around the peak, i.e. classic **vanishing gradients**. The repeated squaring itself is also
  numerically unstable, which is why the code has to clamp at every iteration to avoid `inf`.
- **It needs an oversampling factor `sr` (10, in this codebase) *and* a sharpness schedule (`tau`,
  `clamp_max`)** just to approximate something that should be a one-line interpolation. Every one of those is
  a hyperparameter you now have to tune, with no principled way to pick it beyond "try a few values and see
  what doesn't explode."
- **The memory and compute cost is quadratic.** `POS` and `pp` are both `(n·sr/sr) × (n·sr)`-ish matrices
  built per forward pass. For the ~100-point Lynx series this is annoying; for the ~4,000-point daily Bitcoin
  series it's a non-starter.
- **Boundary handling is a patch, not a model.** Columns that no input row ever lands on become `NaN` after
  the `torch.nansum`, and `fill_nan_with_last_value` forward-fills them one element at a time in a Python
  loop calling `.item()` (breaking the graph again, incidentally, though by this point it's operating on
  already-detached output). The result is a warped curve with flat plateaus wherever the discrete shift
  happened to skip a column, rather than a smooth function of the warp path.

So even the "soft" discrete implementation ends up close to hard-discrete in the regime where it actually
needs to be sharp enough to look like a real shift, which is exactly the regime where its gradients vanish.
Training would stall unless the warp path started almost exactly right, defeating the entire point of
learning it.

## Soft warping: linear interpolation instead of reindexing

The fix is to stop thinking of "warp" as "move this value to a different array slot" and instead treat the
warp path `p(i)` as a **continuous fractional index** into the driver, and read the driver at that index by
linear interpolation. This is the same trick spatial transformer networks use for bilinear image sampling
(`torch.nn.functional.grid_sample`), just in one dimension and hand-rolled so the rest of the model (B-spline
knots, dual likelihood, MLP readouts) can stay in plain torch. It lives in
`src/warp_regression/core/warp.py` as `soft_warp_torch` (and its numpy twin `soft_warp_numpy`, used for
plotting and offline analysis where gradients don't matter):

```python
def soft_warp_torch(
    x: Tensor,
    p: Tensor,
    reverse_path: Optional[bool] = None,
    path_mode: str = "identity",
    path_anchor: PathAnchor = DEFAULT_PATH_ANCHOR,
) -> Tensor:
    if reverse_path is None:
        reverse_path = path_anchor == "start"
    p_use = path_for_warp_torch(p, path_mode=path_mode) if reverse_path else p
    n = x.shape[0]
    p_use = p_use.clamp(0.0, float(n) - 1.001)
    i0 = torch.floor(p_use).long().clamp(0, n - 2)
    i1 = i0 + 1
    w = p_use - i0.to(p_use.dtype)
    return (1.0 - w) * x[i0] + w * x[i1]
```

Walking through the torch calls:

- **`p_use.clamp(0.0, n - 1.001)`** keeps the fractional index inside the valid range of the driver so the
  two neighbours (`i0`, `i0 + 1`) always exist. This is the only place boundary handling happens, and it's a
  single vectorised clamp, not a per-element Python branch.
- **`torch.floor(p_use).long()`** finds the left neighbour index `i0`. `floor` itself has zero gradient (it's
  piecewise constant), but that's fine: it's only used to select *which two array elements* participate in
  the interpolation, not to compute a value directly. The `.clamp(0, n - 2)` guards against the floor landing
  on the very last index, which would leave no right neighbour.
- **`i1 = i0 + 1`** is the right neighbour, a plain integer tensor op with no gradient concerns since it's an
  index, not a value.
- **`w = p_use - i0.to(p_use.dtype)`** is the fractional part of `p_use`. This is where the actual gradient
  path lives: `w` is a smooth (in fact affine) function of `p_use` for any fixed `i0`.
- **`x[i0]`, `x[i1]`** use torch's fancy/advanced indexing to gather both neighbours for every element of `p`
  in one vectorised call, no Python loop over rows at all.
- **`(1.0 - w) * x[i0] + w * x[i1]`** is the linear interpolation itself, a convex combination of the two
  neighbouring driver values.

Because every operation here (`clamp`, arithmetic, advanced indexing, elementwise multiply/add) is a standard
differentiable torch primitive, autograd can walk straight back through `soft_warp_torch` to `p`, and from
`p` (via `path_from_B_torch`, see below) all the way back to the learnable knot values `B`. There is no
`.item()`, no Python-level branching on tensor values, and no discrete lookup table to make discontinuous.

The gradient itself is easy to write down and sanity-check: holding `i0` fixed (true almost everywhere,
except exactly at integer boundaries where the two segments agree anyway),

$$
\frac{\partial}{\partial p}\ \mathrm{soft\_warp}(x, p) = x_{i_0+1} - x_{i_0},
$$

i.e. the local slope of the driver at the point you're reading from. That is a well-scaled, well-behaved
gradient everywhere the driver isn't flat, which is a world away from the near-zero-everywhere gradient the
discrete shift matrix produced.

### The periodic special case: skip interpolation entirely

For the sine-shaped drivers used in every notebook in this repo, there's an even cleaner option:
`soft_warp_sine_torch` doesn't gather from a precomputed array at all. Since a sine wave has a closed form,
it just evaluates the sine directly at the continuous warped time:

```python
def soft_warp_sine_torch(p, n, omega, phase, time_scale=1.0, t_shift=0.0, ...):
    p_use = path_for_warp_torch(p, path_mode=path_mode) if reverse_path else p
    t_warp = (p_use + 0.5) / n
    return torch.sin(2.0 * math.pi * omega * time_scale * t_warp + phase + t_shift)
```

There's no `floor`, no neighbour gather, no interpolation error at all: the gradient
`d/dp sin(2πω·((p+½)/n) + φ) = (2πω/n)·cos(...)` is exact, not a piecewise-linear approximation. This is used
wherever the driver is a known sine (Lynx's two components, Bitcoin's macro cycle) instead of an arbitrary
numeric array.

### Why this also unblocks forecasting, not just training

The interpolation-based `soft_warp` is `O(n)` per call, with no oversampling factor and no giant matrix.
That matters twice over: it's what lets the dual-loss training loop (`compute_dual_loss` /
`train_dual_warp` in `src/warp_regression/core/training.py`) backprop through thousands of epochs cheaply,
and it's what makes Monte Carlo forecasting practical. `build_forecast_bands` and
`predict_forecast_realisations_torch` sample hundreds to thousands of future warp paths and re-run
`soft_warp` on each one to build terror/combined bands; at `O(n · sr²)`-ish cost from the old shift-matrix
approach, that workload would have been impractical for the ~4,000-point daily Bitcoin series.

## Side by side

| | Discrete (`legacy/warp.py`) | "Soft" discrete (`legacy/warp_network.py`) | Soft warp (`src/warp_regression/core/warp.py`) |
|---|---|---|---|
| Core operation | Python-loop reindex with `int(p[i].item())` | Sharpened Gaussian-like kernel matrix, `P @ x` | Linear interpolation between two gathered neighbours |
| Gradient w.r.t. path `p` | None (`.item()` detaches from the graph) | Present in theory, vanishes in practice once the kernel is sharpened | Exact, well-scaled everywhere (`x[i1] - x[i0]`) |
| Extra hyperparameters | `sr` (oversample rate) | `sr`, `tau` (sharpening iterations), `clamp_max` | None beyond the warp path itself |
| Cost | `O(n)` python loop, but not batched/vectorised | `O(n·sr)` memory for the shift matrix per forward pass | `O(n)`, fully vectorised |
| Boundary behaviour | Silent truncation via slicing | Unmatched columns become `NaN`, forward-filled into flat plateaus | Single vectorised clamp, continuous everywhere |
| External deps | `xitorch.interpolate.Interp1D` (unused by the actual shift path, imported regardless) | `patsy.dmatrix` for the (separate) spline basis | None; pure `torch` / `numpy` |
| Usable for joint torch training | No | Not in practice (gradients too weak to learn a good path from a bad start) | Yes: this is what `WarpParametricModel.fit` actually trains with |

## Where soft warping shows up in the active codebase

- `soft_warp_numpy` / `soft_warp_torch`: generic linear-interpolation warp for an arbitrary driver array
  (`src/warp_regression/core/warp.py`).
- `soft_warp_sine_numpy` / `soft_warp_sine_torch`: closed-form warp for a sine driver, no interpolation
  error at all.
- `WarpParametricModel.predict` (`src/warp_regression/readouts/parametric.py`) calls `soft_warp_torch`
  directly inside the forward pass, so `loss.backward()` updates the B-spline knot parameters `B` and the
  readout parameters (`A`, `C`, or an MLP's weights) in a single backward call.
- `path_from_B_torch` (`src/warp_regression/core/path.py`) turns the learnable knot values `B` into a full
  path `p` via a small amount of piecewise-linear interpolation of its own (`_offset_from_G_torch`), keeping
  the whole `B → p → soft_warp(x, p) → ŷ` chain differentiable end to end.
- `build_forecast_bands`, `predict_forecast_realisations_torch`, `sample_warp_paths_future*`
  (`src/warp_regression/forecast.py`) reuse the same `soft_warp` functions to cheaply evaluate thousands of
  sampled future warp paths for the terror/combined uncertainty bands.

There is also a small, deliberately simplified `discrete_warp_numpy` left in the current codebase
(`src/warp_regression/core/warp.py`) that rounds each warped index to its nearest integer column and
median-aggregates collisions. It's kept around purely as a numpy-only reference point for "what would a naive
discrete warp look like", not as something anything in the active training path calls.
