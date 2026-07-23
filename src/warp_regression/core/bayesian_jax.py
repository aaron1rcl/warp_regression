"""JAX dual-likelihood kernels + PyMC Op for fully Bayesian warp regression.

Requires the optional ``bayes`` extra (``pymc``, ``jax``, ``numpyro``).

The main training stack is PyTorch. PyMC's JAX / NumPyro sampler needs the same
soft-warp / path / terror math in JAX so the model log-density compiles
end-to-end. This module ports those kernels and wraps the dual potential as a
PyTensor ``Op`` with ``jax_funcify`` registration.

Optional ``pins`` / ``terror_mask`` / ``baseline`` mirror the PyTorch dual-loss
API used for sparse / MMM problems.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.link.jax.dispatch import jax_funcify

from .path import allocate_interval_knots, normalize_pins, pins_parameter_size

PinsLike = Optional[Union[Sequence[int], np.ndarray]]
ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------------
# JAX kernels (mirrors ``core.path`` / ``core.warp`` / ``core.training``)
# ---------------------------------------------------------------------------


def zero_gradient_G(B):
    parts = [B[0]]
    cum = B[0]
    for i in range(1, int(B.shape[0])):
        gi = -cum + B[i]
        parts.append(gi)
        cum = cum + gi
    return jnp.stack(parts)


def offset_from_G(G, n: int, n_knots: int):
    knot_x = jnp.linspace(0.0, n - 1.0, n_knots)
    idx = jnp.arange(n, dtype=G.dtype)
    offset = jnp.zeros(n, dtype=G.dtype)
    for j in range(n_knots - 1):
        x0, x1 = knot_x[j], knot_x[j + 1]
        g0, g1 = G[j], G[j + 1]
        w = jnp.where(x1 > x0, (idx - x0) / (x1 - x0), 0.0)
        seg = (1.0 - w) * g0 + w * g1
        mask = (idx >= x0) & (idx <= x1)
        offset = jnp.where(mask, seg, offset)
    return offset


def _segment_offset(B_seg, n_seg: int, n_knots_seg: int, *, pin_end: bool):
    """PWL offset on a segment; start always pinned, optional end pin."""
    if n_seg <= 1:
        return jnp.zeros((n_seg,), dtype=B_seg.dtype)
    k = max(int(n_knots_seg), 2)
    if int(B_seg.shape[0]) < k:
        B_use = jnp.concatenate(
            [B_seg, jnp.zeros((k - int(B_seg.shape[0]),), dtype=B_seg.dtype)]
        )
    else:
        B_use = B_seg[:k]
    G = zero_gradient_G(B_use)
    G = G.at[0].set(0.0)
    offset = offset_from_G(G, n_seg, k)
    offset = offset - offset[0]
    if pin_end:
        ramp = jnp.linspace(0.0, 1.0, n_seg, dtype=offset.dtype)
        offset = offset - offset[-1] * ramp
    return offset


def path_from_B(B, n: int, n_knots: int, pins: PinsLike = None):
    """Identity path from knot coefficients ``B``.

    Without ``pins``, start-anchored (``p[0]=0``) as in notebook 4.
    With ``pins``, concatenated both-ends-pinned bridges so ``p[i]=i`` exactly
    at every pin (same construction as ``path_from_B_torch``).
    """
    pin_arr = normalize_pins(pins, n)
    if pin_arr is None:
        G = zero_gradient_G(B)
        G = G.at[0].set(0.0)
        offset = offset_from_G(G, n, n_knots)
        offset = offset.at[0].set(0.0)
        p = jnp.arange(n, dtype=B.dtype) + offset
        return p.at[0].set(0.0)

    pin_list = [int(i) for i in pin_arr.tolist()]
    pin_set = set(pin_list)
    cuts = [0] + [i for i in pin_list if i not in (0, n - 1)] + [n - 1]
    cuts = sorted(set(cuts))

    intervals = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        trailing_open = b == n - 1 and (n - 1) not in pin_set
        intervals.append((a, b, not trailing_open))

    lengths = [b - a + 1 for a, b, _ in intervals]
    knot_counts = allocate_interval_knots(lengths, n_knots)

    offset = jnp.zeros(n, dtype=B.dtype)
    cursor = 0
    for (a, b, pin_end), k_i in zip(intervals, knot_counts):
        n_seg = b - a + 1
        B_seg = B[cursor : cursor + k_i]
        cursor += k_i
        seg_off = _segment_offset(B_seg, n_seg, k_i, pin_end=pin_end)
        offset = offset.at[a : a + n_seg].set(seg_off)

    p = jnp.arange(n, dtype=B.dtype) + offset
    for i in pin_set:
        p = p.at[i].set(float(i))
    p = p.at[0].set(0.0)
    return p


def soft_warp(x, p):
    n = x.shape[0]
    p = jnp.clip(p, 0.0, n - 1.001)
    i0 = jnp.floor(p).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, n - 2)
    i1 = i0 + 1
    w = p - i0.astype(p.dtype)
    return (1.0 - w) * x[i0] + w * x[i1]


def reverse_softplus(x):
    return jnp.where(x > -10.0, -jnp.log1p(jnp.exp(-x)), x)


def expected_likelihood(d, sd, n_steps: float):
    u = d / n_steps
    post_sd = jnp.sqrt((sd**2) * (n_steps - 1.0) / n_steps)
    alpha = jnp.sqrt(1.0 / (2.0 * post_sd**2) + 1.0 / (2.0 * sd**2))
    alpha_beta = u / (2.0 * post_sd**2)
    gamma = (u**2) / (2.0 * post_sd**2)
    beta = alpha_beta / alpha
    p1 = 1.0 / (2.0 * sd * post_sd * alpha * math.sqrt(math.pi))
    p2 = jnp.exp(-gamma + beta**2)
    el = jnp.log(p1 * p2)
    return reverse_softplus(el * n_steps)


def terror_likelihood(p_off, sigma_t, n_knots: int, mask=None):
    """Sum of expected segment likelihoods; optional length-``n`` terror mask."""
    n = int(p_off.shape[0])
    k_values = np.arange(0, n_knots, dtype=np.float64) * (float(n) / float(n_knots))
    k_values = np.append(k_values, n - 1.0)
    rw_width = float(n / n_knots)
    w = None if mask is None else jnp.asarray(mask, dtype=p_off.dtype).reshape(-1)
    total = 0.0
    for i in range(1, len(k_values)):
        x1, x2 = int(k_values[i - 1]), int(k_values[i])
        weight = 1.0
        if w is not None:
            lo = min(x1 + 1, x2)
            hi = x2 + 1
            seg = w[x1 : x2 + 1] if hi <= lo else w[lo:hi]
            weight = jnp.mean(seg) if seg.size else 0.0
        d = p_off[x2] - p_off[x1]
        total = total + weight * expected_likelihood(d, sigma_t, rw_width)
    return total


def gaussian_error_nll(resid, sigma):
    var = sigma**2 + 1e-8
    return jnp.sum(0.5 * (resid**2 / var + jnp.log(2.0 * math.pi * var)))


def dual_log_density(
    B,
    A,
    bias,
    sigma_y,
    sigma_t,
    x,
    y,
    n_knots: int,
    lam: float = 0.5,
    *,
    pins: PinsLike = None,
    terror_mask=None,
    baseline=None,
):
    """Dual log-density matching training at ``fit_lambda=lam``.

    ``bias`` is either scalar ``C`` (no baseline) or vector ``beta`` when
    ``baseline`` is an ``(n, k)`` design matrix:
    ``y_hat = A·warp(x,p) + baseline@beta``.
    """
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must be in (0, 1], got {lam}")
    n = int(x.shape[0])
    p = path_from_B(B, n, n_knots, pins=pins)
    warped = A * soft_warp(x, p)
    if baseline is None:
        y_hat = warped + bias
    else:
        y_hat = warped + baseline @ bias
    obj_err = gaussian_error_nll(y - y_hat, sigma_y)
    p_off = p - jnp.arange(n, dtype=p.dtype)
    terror_ll = terror_likelihood(p_off, sigma_t, n_knots, mask=terror_mask)
    return -obj_err + ((1.0 - lam) / lam) * terror_ll


# ---------------------------------------------------------------------------
# PyMC / PyTensor Op
# ---------------------------------------------------------------------------


def make_dual_logp_op(
    x: np.ndarray,
    y: np.ndarray,
    n_knots: int,
    lam: float = 0.5,
    *,
    pins: PinsLike = None,
    terror_mask: Optional[ArrayLike] = None,
    baseline: Optional[np.ndarray] = None,
) -> Tuple[Op, Callable]:
    """Build a PyTensor Op for the dual log-density and register JAX dispatch.

    Parameters
    ----------
    x, y
        Covariate and target (1-d float64).
    n_knots, lam
        Path knot count and dual weight (0.5 = equal error / terror).
    pins, terror_mask
        Optional multi-pin path / terror masking (MMM / sparse drivers).
    baseline
        Optional ``(n, k)`` unwarped design matrix. When set, the Op signature
        is ``(B, A, beta, sigma_y, sigma_t)`` with ``beta`` length ``k``;
        otherwise ``(B, A, C, sigma_y, sigma_t)`` as in notebook 4.

    Returns
    -------
    op, jax_fn
        Scalar dual log-density Op and the underlying JAX callable.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_j = jnp.asarray(x)
    y_j = jnp.asarray(y)
    pin_arr = normalize_pins(pins, len(x))
    mask_j = None if terror_mask is None else jnp.asarray(terror_mask, dtype=jnp.float64)
    base_j = None if baseline is None else jnp.asarray(baseline, dtype=jnp.float64)
    if base_j is not None and base_j.ndim != 2:
        raise ValueError("baseline must be a 2-d (n, k) array")

    def jax_fn(B, A, bias, sigma_y, sigma_t):
        return dual_log_density(
            B,
            A,
            bias,
            sigma_y,
            sigma_t,
            x_j,
            y_j,
            n_knots,
            lam,
            pins=pin_arr,
            terror_mask=mask_j,
            baseline=base_j,
        )

    if base_j is None:

        class DualLogpOp(Op):
            itypes = [pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
            otypes = [pt.dscalar]

            def perform(self, node, inputs, outputs):
                B, A, C, sigma_y, sigma_t = inputs
                val = jax_fn(
                    jnp.asarray(B),
                    jnp.asarray(A),
                    jnp.asarray(C),
                    jnp.asarray(sigma_y),
                    jnp.asarray(sigma_t),
                )
                outputs[0][0] = np.asarray(val, dtype=np.float64)

            def make_node(self, B, A, C, sigma_y, sigma_t):
                return Apply(
                    self,
                    [
                        pt.as_tensor_variable(B),
                        pt.as_tensor_variable(A),
                        pt.as_tensor_variable(C),
                        pt.as_tensor_variable(sigma_y),
                        pt.as_tensor_variable(sigma_t),
                    ],
                    [pt.dscalar()],
                )

    else:
        k = int(base_j.shape[1])

        class DualLogpOp(Op):
            itypes = [pt.dvector, pt.dscalar, pt.dvector, pt.dscalar, pt.dscalar]
            otypes = [pt.dscalar]

            def perform(self, node, inputs, outputs):
                B, A, beta, sigma_y, sigma_t = inputs
                beta = np.asarray(beta, dtype=np.float64).reshape(-1)
                if beta.size != k:
                    raise ValueError(f"beta length {beta.size} != baseline cols {k}")
                val = jax_fn(
                    jnp.asarray(B),
                    jnp.asarray(A),
                    jnp.asarray(beta),
                    jnp.asarray(sigma_y),
                    jnp.asarray(sigma_t),
                )
                outputs[0][0] = np.asarray(val, dtype=np.float64)

            def make_node(self, B, A, beta, sigma_y, sigma_t):
                return Apply(
                    self,
                    [
                        pt.as_tensor_variable(B),
                        pt.as_tensor_variable(A),
                        pt.as_tensor_variable(beta),
                        pt.as_tensor_variable(sigma_y),
                        pt.as_tensor_variable(sigma_t),
                    ],
                    [pt.dscalar()],
                )

    op = DualLogpOp()

    @jax_funcify.register(DualLogpOp)
    def _jax_funcify_dual(_op, **kwargs):  # noqa: ARG001
        return jax_fn

    return op, jax_fn


def predict_from_params(
    B,
    A,
    bias,
    x,
    n_knots: int,
    *,
    pins: PinsLike = None,
    baseline=None,
):
    """Point prediction ``A · soft_warp(x, p(B)) + bias`` (or ``+ baseline@bias``)."""
    n = int(x.shape[0])
    p = path_from_B(B, n, n_knots, pins=pins)
    warped = A * soft_warp(x, p)
    if baseline is None:
        return warped + bias, p
    return warped + baseline @ bias, p


__all__ = [
    "dual_log_density",
    "make_dual_logp_op",
    "path_from_B",
    "pins_parameter_size",
    "predict_from_params",
    "soft_warp",
    "terror_likelihood",
]
