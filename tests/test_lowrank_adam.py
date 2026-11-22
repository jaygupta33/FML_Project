"""Unit tests for optim.LowRankAdam. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optim import LowRankAdam
from nsbench.core import _tensor_bytes

torch.set_default_dtype(torch.float64)       # exactness checks without float32 rounding noise


def run_pair(shape, dtype, steps=30, **kw):
    """Same parameter and gradient sequence through torch Adam and LowRankAdam; return max abs diff."""
    g = torch.Generator().manual_seed(0)
    w0 = torch.randn(shape, dtype=dtype, generator=g)
    a, b = w0.clone().requires_grad_(), w0.clone().requires_grad_()
    oa = torch.optim.Adam([a], lr=1e-2, foreach=False)
    ob = LowRankAdam([b], lr=1e-2, **kw)
    for _ in range(steps):
        grad = torch.randn(shape, dtype=dtype, generator=g)
        a.grad, b.grad = grad.clone(), grad.clone()
        oa.step(); ob.step()
    return (a - b).abs().max().item(), ob


@pytest.mark.parametrize("shape", [(6, 40), (40, 6), (4, 5, 3, 2)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_identity_basis_equals_adam(shape, dtype):
    """Full-rank identity projection must reproduce torch.optim.Adam, real and complex, both orientations."""
    diff, opt = run_pair(shape, dtype, proj="identity", min_numel=1)
    assert opt.ranks(), "tensor was not routed through the compressed path"
    assert diff < 1e-12


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_uncompressed_equals_adam(dtype):
    diff, opt = run_pair((6, 40), dtype, min_numel=10**9)
    assert not opt.ranks()
    assert diff < 1e-12


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_fixed_rank_state_shapes_and_memory(dtype):
    m, n, r = 32, 300, 4
    p = torch.randn(m, n, dtype=dtype, requires_grad=True)
    opt = LowRankAdam([p], rank=r, min_numel=1)
    p.grad = torch.randn(m, n, dtype=dtype)
    opt.step()
    st = opt.state[p]
    assert st["P"].shape == (m, r) and st["exp_avg"].shape == (r, n) and st["exp_avg_sq"].shape == (r, n)
    eye = st["P"].mH @ st["P"]
    assert torch.allclose(eye, torch.eye(r, dtype=dtype), atol=1e-10)         # orthonormal basis
    adam_bytes = 2 * p.numel() * p.element_size()
    ours = _tensor_bytes(opt.state[p])
    assert ours == (2 * r * n + m * r) * p.element_size()                      # moments + P
    assert ours < adam_bytes / 5


def test_projection_on_smaller_side():
    """A tall matrix (m > n) is projected along n, so P is n x r."""
    p = torch.randn(300, 32, requires_grad=True)
    opt = LowRankAdam([p], rank=4, min_numel=1)
    p.grad = torch.randn(300, 32)
    opt.step()
    assert opt.state[p]["P"].shape == (32, 4) and opt.state[p]["exp_avg"].shape == (4, 300)


def low_rank_grad(m, n, k, dtype, g):
    A = torch.randn(m, k, dtype=dtype, generator=g)
    B = torch.randn(k, n, dtype=dtype, generator=g)
    return A @ B + 1e-6 * torch.randn(m, n, dtype=dtype, generator=g)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("k", [3, 7])
def test_adaptive_rank_finds_true_rank(dtype, k):
    g = torch.Generator().manual_seed(1)
    p = torch.randn(32, 200, dtype=dtype, requires_grad=True)
    opt = LowRankAdam([p], rank_mode="adaptive", tau=0.999, r_min=1, min_numel=1, spectrum_beta=0.0)
    p.grad = low_rank_grad(32, 200, k, dtype, g)
    opt.step()
    assert opt.ranks() == [k]
    assert opt.rank_log[-1][3] > 0.999


def test_adaptive_rank_clamped():
    g = torch.Generator().manual_seed(2)
    p = torch.randn(32, 200, requires_grad=True)
    opt = LowRankAdam([p], rank_mode="adaptive", tau=0.9999, r_min=2, r_max=5, min_numel=1)
    p.grad = torch.randn(32, 200, generator=g)              # full-rank gradient wants r ~ 32
    opt.step()
    assert opt.ranks() == [5]


@pytest.mark.parametrize("mode", ["keep", "reset", "rotate"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_rank_change_on_refresh(mode, dtype):
    """Rank goes 3 -> 6 -> 2 across refreshes; moment shapes follow and stay finite."""
    g = torch.Generator().manual_seed(3)
    p = torch.randn(32, 120, dtype=dtype, requires_grad=True)
    opt = LowRankAdam([p], rank_mode="adaptive", tau=0.999, r_min=1, refresh=2, spectrum_beta=0.0,
                      on_change=mode, min_numel=1)
    seen = []
    for k in (3, 3, 6, 6, 2, 2):
        p.grad = low_rank_grad(32, 120, k, dtype, g)
        opt.step()
        st = opt.state[p]
        seen.append(st["P"].shape[1])
        assert st["exp_avg"].shape == st["exp_avg_sq"].shape == (seen[-1], 120)
        assert torch.isfinite(torch.view_as_real(p.detach()) if p.is_complex() else p.detach()).all()
    assert seen == [3, 3, 6, 6, 2, 2]
    if mode == "reset":
        assert opt.state[p]["step"] == 2


def test_rotate_same_basis_is_noop():
    p = torch.randn(16, 80, dtype=torch.complex128, requires_grad=True)
    opt = LowRankAdam([p], rank=4, min_numel=1, on_change="rotate")
    p.grad = torch.randn(16, 80, dtype=torch.complex128)
    opt.step()
    st = opt.state[p]
    m0, v0 = st["exp_avg"].clone(), st["exp_avg_sq"].clone()
    opt._change_basis(st, st["P"], st["P"])
    assert torch.allclose(st["exp_avg"], m0) and torch.allclose(st["exp_avg_sq"], v0)


def test_refresh_schedule():
    p = torch.randn(16, 80, requires_grad=True)
    opt = LowRankAdam([p], rank=2, min_numel=1, refresh=10, refresh_early=2, early_steps=6)
    for _ in range(25):
        p.grad = torch.randn(16, 80)
        opt.step()
    assert [s for s, *_ in opt.rank_log] == [1, 3, 5, 11, 21]


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_low_rank_training_decreases_loss(dtype):
    """Least squares with a low-rank target: fixed-rank GaLore should still drive the loss down."""
    g = torch.Generator().manual_seed(4)
    X = torch.randn(256, 64, dtype=dtype, generator=g)
    W_true = low_rank_grad(64, 48, 4, dtype, g)
    W = torch.zeros(64, 48, dtype=dtype, requires_grad=True)
    opt = LowRankAdam([W], lr=3e-2, rank=8, refresh=20, min_numel=1)
    loss0 = None
    for _ in range(1000):
        loss = (X @ W - X @ W_true).abs().pow(2).mean()
        loss0 = loss0 or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 2e-2 * loss0          # slower than Adam (rank-8 subspace) but converging


def test_fno_spectral_weights_compressed():
    """On the real FNO only the four 64x64x16x9 complex spectral weights take the low-rank path."""
    torch.set_default_dtype(torch.float32)
    try:
        from nsbench import FourierNeuralOperator
        model = FourierNeuralOperator()
        opt = LowRankAdam(model.parameters(), rank=8)
        x = torch.randn(2, 1, 64, 64)
        model(x).pow(2).mean().backward()
        opt.step()
        comp = [p for p in model.parameters() if opt.compressed(p)]
        assert len(comp) == 4 and all(p.shape == (64, 64, 16, 9) and p.is_complex() for p in comp)
        assert opt.ranks() == [8, 8, 8, 8]
        full = sum(2 * p.numel() * p.element_size() for p in model.parameters())
        ours = sum(_tensor_bytes(s) for s in opt.state.values())
        assert ours < 0.2 * full
    finally:
        torch.set_default_dtype(torch.float64)
