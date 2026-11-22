"""Unit tests for solver.NS2D. Run: python -m pytest tests -q"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver import NS2D, rel_l2

M = 64
x = torch.arange(M, dtype=torch.float64) * 2 * math.pi / M
A, B = torch.meshgrid(x, x, indexing="ij")
METHODS = ["cn_euler", "cn_heun", "ifrk4"]


def smooth_field(seed, kmax=6):
    g = torch.Generator().manual_seed(seed)
    w = torch.zeros(M, M, dtype=torch.float64)
    for i in range(-kmax, kmax + 1):
        for j in range(-kmax, kmax + 1):
            if i or j:
                a, ph = torch.randn(2, generator=g, dtype=torch.float64)
                w += a / (i * i + j * j) * torch.cos(i * A + j * B + ph)
    return w


@pytest.mark.parametrize("method", METHODS)
def test_eigenfunction_decays_exactly(method):
    """A sum of modes with one |k| has zero advection, so w(t) = w0 exp(-nu |k|^2 t)."""
    nu, T = 0.01, 1.0
    w0 = torch.sin(3 * A) * torch.cos(4 * B) + torch.cos(5 * A)
    w = NS2D(M, nu=nu, dt=1e-2, method=method).solve(torch.zeros(M, M), T=T, w0=w0)
    tol = {"cn_euler": 1e-4, "cn_heun": 1e-4, "ifrk4": 1e-12}[method]   # CN is 2nd order in diffusion
    assert rel_l2(w, w0 * math.exp(-25 * nu * T)) < tol


@pytest.mark.parametrize("method", METHODS)
def test_steady_forcing_from_rest(method):
    """Forcing by an eigenfunction g with |k|^2 = K: w(t) = g (1 - exp(-nu K t)) / (nu K)."""
    nu, T, K = 0.05, 2.0, 25
    g = torch.cos(3 * A + 4 * B)
    w = NS2D(M, nu=nu, dt=1e-2, method=method).solve(g, T=T)
    exact = g * (1 - math.exp(-nu * K * T)) / (nu * K)
    assert rel_l2(w, exact) < 1e-3


def test_convergence_orders():
    """Nonlinear flow: error vs a fine ifrk4 reference shrinks at the expected rates."""
    w0, f = smooth_field(0), 0.1 * smooth_field(1)
    ref = NS2D(M, nu=1 / 500, dt=1e-4, method="ifrk4").solve(f, T=1.0, w0=w0)
    for method, order in (("cn_euler", 1), ("cn_heun", 2), ("ifrk4", 4)):
        errs = [rel_l2(NS2D(M, nu=1 / 500, dt=dt, method=method).solve(f, T=1.0, w0=w0), ref).item()
                for dt in (5e-3, 2.5e-3)]
        rate = math.log2(errs[0] / errs[1])
        assert abs(rate - order) < 0.5, (method, errs, rate)


def test_sign_flip_is_axis_swap():
    f = 0.5 * smooth_field(2)
    s1 = NS2D(M, dt=1e-2, sign=-1.0).solve(f, T=1.0)
    s2 = NS2D(M, dt=1e-2).solve(f.T.contiguous(), T=1.0).T
    assert rel_l2(s1, s2) < 1e-12


def test_resolution_independence():
    """Smooth forcing: solving on 128^2 and reading out on 64^2 agrees with the 64^2 solve."""
    f = 0.5 * smooth_field(3, kmax=5)
    lo = NS2D(64, dt=1e-2).solve(f, T=1.0)
    hi = NS2D(128, dt=1e-2).solve(f, T=1.0, out_res=64)
    assert rel_l2(lo, hi) < 1e-6


def test_batched_equals_single():
    f = torch.stack([0.3 * smooth_field(4), 0.3 * smooth_field(5)])
    s = NS2D(M, dt=1e-2)
    both = s.solve(f, T=0.5)
    assert rel_l2(both[1], s.solve(f[1], T=0.5)) < 1e-13
