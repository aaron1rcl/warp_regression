"""JAX dual-likelihood kernels + PyMC Op for fully Bayesian warp regression.

Requires the optional ``bayes`` extra (``pymc``, ``jax``, ``numpyro``).

The main training stack is PyTorch. PyMC's JAX / NumPyro sampler needs the same
soft-warp / path / terror math in JAX so the model log-density compiles
end-to-end. This module ports those kernels and wraps the dual potential as a
PyTensor ``Op`` with ``jax_funcify`` registration.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.link.jax.dispatch import jax_funcify


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


def path_from_B(B, n: int, n_knots: int):
    """Start-anchored identity path from knot coefficients ``B``."""
    G = zero_gradient_G(B)
    G = G.at[0].set(0.0)
    offset = offset_from_G(G, n, n_knots)
    offset = offset.at[0].set(0.0)
    p = jnp.arange(n, dtype=B.dtype) + offset
    return p.at[0].set(0.0)


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


def terror_likelihood(p_off, sigma_t, n_knots: int):
    n = int(p_off.shape[0])
    k_values = np.arange(0, n_knots, dtype=np.float64) * (float(n) / float(n_knots))
    k_values = np.append(k_values, n - 1.0)
    rw_width = float(n / n_knots)
    total = 0.0
    for i in range(1, len(k_values)):
        x1, x2 = int(k_values[i - 1]), int(k_values[i])
        d = p_off[x2] - p_off[x1]
        total = total + expected_likelihood(d, sigma_t, rw_width)
    return total


def gaussian_error_nll(resid, sigma):
    var = sigma**2 + 1e-8
    return jnp.sum(0.5 * (resid**2 / var + jnp.log(2.0 * math.pi * var)))


def dual_log_density(B, A, C, sigma_y, sigma_t, x, y, n_knots: int, lam: float = 0.5):
    """Dual log-density matching training at ``fit_lambda=lam``, untempered for priors.

    ``−error + terror``.
    """
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must be in (0, 1], got {lam}")
    n = int(x.shape[0])
    p = path_from_B(B, n, n_knots)
    y_hat = A * soft_warp(x, p) + C
    obj_err = gaussian_error_nll(y - y_hat, sigma_y)
    p_off = p - jnp.arange(n, dtype=p.dtype)
    terror_ll = terror_likelihood(p_off, sigma_t, n_knots)
    return -obj_err + ((1.0 - lam) / lam) * terror_ll


# ---------------------------------------------------------------------------
# PyMC / PyTensor Op
# ---------------------------------------------------------------------------


def make_dual_logp_op(
    x: np.ndarray,
    y: np.ndarray,
    n_knots: int,
    lam: float = 0.5,
) -> Tuple[Op, Callable]:
    """Build a PyTensor Op for the dual log-density and register JAX dispatch.

    Parameters
    ----------
    x, y
        Covariate and target (1-d float64), typically the full series.
    n_knots, lam
        Path knot count and dual weight (0.5 = equal error / terror).

    Returns
    -------
    op, jax_fn
        ``op(B, A, C, sigma_y, sigma_t)`` → scalar; ``jax_fn`` is the underlying
        JAX callable (also used after sampling for posterior fits).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_j = jnp.asarray(x)
    y_j = jnp.asarray(y)

    def jax_fn(B, A, C, sigma_y, sigma_t):
        return dual_log_density(B, A, C, sigma_y, sigma_t, x_j, y_j, n_knots, lam)

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

    op = DualLogpOp()

    @jax_funcify.register(DualLogpOp)
    def _jax_funcify_dual(_op, **kwargs):  # noqa: ARG001
        return jax_fn

    return op, jax_fn


def predict_from_params(B, A, C, x, n_knots: int):
    """Point prediction ``A · soft_warp(x, p(B)) + C`` (JAX arrays / scalars)."""
    n = int(x.shape[0])
    p = path_from_B(B, n, n_knots)
    return A * soft_warp(x, p) + C, p
