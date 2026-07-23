"""Tests for sparse helpers, multi-pin paths, and terror masks."""

from __future__ import annotations

import numpy as np
import torch

from warp_regression import (
    WarpPath,
    geometric_adstock,
    pulse_start_indices,
    soft_warp_numpy,
    terror_mask_from_values,
)
from warp_regression.core.path import path_from_B_torch, pins_parameter_size
from warp_regression.core.training import compute_dual_loss, terror_likelihood_torch
from warp_regression.core.warp import soft_warp_torch


def test_pulse_start_indices_impulses_and_blocks():
    x = np.array([0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0, np.nan, 3.0])
    starts = pulse_start_indices(x)
    np.testing.assert_array_equal(starts, np.array([2, 4, 8]))


def test_geometric_adstock_recurrence():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    s = geometric_adstock(x, 0.5)
    np.testing.assert_allclose(s, [1.0, 0.5, 0.25, 0.125])


def test_terror_mask_from_values():
    x = np.array([0.0, 1.0, np.nan, -2.0])
    m = terror_mask_from_values(x)
    np.testing.assert_array_equal(m, np.array([False, True, False, True]))


def test_path_pins_exact_and_gradients():
    n, n_knots = 60, 8
    pins = [5, 20, 40]
    bdim = pins_parameter_size(n, n_knots, pins)
    B = torch.randn(bdim, requires_grad=True)
    p = path_from_B_torch(B, n, n_knots, pins=pins)
    for i in pins:
        assert abs(float(p[i].detach()) - float(i)) < 1e-5
    (p.pow(2).sum()).backward()
    assert B.grad is not None and float(B.grad.norm()) > 0.0

    path = WarpPath(n, n_knots, pins=pins)
    pp = path.path()
    for i in pins:
        assert abs(float(pp[i].detach()) - float(i)) < 1e-5


def test_terror_mask_semantics():
    n, n_knots = 40, 6
    off = torch.randn(n)
    sigma = torch.tensor(0.4)
    full = terror_likelihood_torch(off, sigma, n_knots)
    ones = terror_likelihood_torch(off, sigma, n_knots, mask=np.ones(n))
    zeros = terror_likelihood_torch(off, sigma, n_knots, mask=np.zeros(n))
    assert abs(float(full - ones)) < 1e-5
    assert abs(float(zeros)) < 1e-8


def test_pinned_adstock_recovery_smoke():
    """Noiseless affine recovery with pulse pins + terror mask."""
    n, n_knots = 80, 10
    spend = np.zeros(n)
    spend[[10, 11, 35, 55, 56]] = [1.0, 0.5, 1.2, 0.8, 0.4]
    adstock = geometric_adstock(spend, 0.5)
    pins = pulse_start_indices(spend)
    mask = terror_mask_from_values(spend)

    path_true = WarpPath(n, n_knots, pins=pins)
    with torch.no_grad():
        path_true.B.copy_(torch.linspace(-0.8, 0.8, path_true.B.numel()))
        p_true = path_true.path().numpy()
    A_true, C_true = 1.7, -0.3
    y = A_true * soft_warp_numpy(adstock, p_true) + C_true

    layer_path = WarpPath(n, n_knots, pins=pins)
    A = torch.nn.Parameter(torch.tensor(1.0))
    C = torch.nn.Parameter(torch.tensor(0.0))
    log_sy = torch.nn.Parameter(torch.tensor(np.log(0.05)))
    opt = torch.optim.Adam(
        [layer_path.B, A, C, log_sy, layer_path.log_sigma_t], lr=0.05
    )
    x_t = torch.tensor(adstock, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    for _ in range(2500):
        opt.zero_grad()
        p = layer_path.path()
        y_hat = A * soft_warp_torch(x_t, p) + C
        loss, _ = compute_dual_loss(
            y_t,
            y_hat,
            p,
            log_sy,
            layer_path.log_sigma_t,
            n_knots,
            0.5,
            terror_mask=mask,
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        p_hat = layer_path.path().numpy()
        for i in pins:
            assert abs(p_hat[i] - i) < 1e-4
        assert abs(float(A.detach()) - A_true) < 0.15
        assert abs(float(C.detach()) - C_true) < 0.15
        assert float(np.mean(np.abs(p_hat - p_true))) < 3.0
