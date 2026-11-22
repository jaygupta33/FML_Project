"""Batched pseudo-spectral solver for 2-D incompressible Navier-Stokes in vorticity form on [0, 2pi)^2:

    d_t w + u . grad w = nu * Lap w + f,     u = s * (d_b psi, -d_a psi),   -Lap psi = w,   w(0) = w0 (default 0)

Fields are arrays [..., M, M] with axes (a, b). s = +1 is the usual convention; s = -1 is the same
equation with the two axes swapped (a reflection), kept as an option because the axis order of the
dataset is not documented. Spatial derivatives are exact in Fourier space; the nonlinear term is
de-aliased with the 2/3 rule.

time steppers
  cn_euler  Crank-Nicolson diffusion, forward-Euler advection + forcing (first order)
  cn_heun   Crank-Nicolson diffusion, Heun predictor-corrector on advection + forcing (second order)
  ifrk4     integrating-factor RK4: diffusion exact, RK4 on advection + forcing (fourth order)

Coefficients use norm="forward" so a Fourier coefficient means the same amplitude on any grid,
which makes spectral resampling (e.g. solve on 256^2, compare on 128^2) a plain mode selection.
"""
import math
import time

import torch


class NS2D:
    def __init__(self, M=128, nu=1 / 500, dt=1e-3, method="cn_heun", sign=1.0,
                 device="cpu", dtype=torch.float64, dealias=True):
        if method not in ("cn_euler", "cn_heun", "ifrk4"):
            raise ValueError(method)
        self.M, self.nu, self.dt, self.method, self.sign = M, nu, dt, method, sign
        self.device, self.dtype = torch.device(device), dtype
        ka = torch.fft.fftfreq(M, 1 / M, dtype=dtype, device=self.device)[:, None]    # integer wavenumbers
        kb = torch.fft.rfftfreq(M, 1 / M, dtype=dtype, device=self.device)[None, :]
        self.ika, self.ikb = 1j * ka, 1j * kb
        self.k2 = ka ** 2 + kb ** 2
        self.inv_k2 = torch.where(self.k2 > 0, 1 / self.k2.clamp_min(1e-30), torch.zeros_like(self.k2))
        cut = M / 3
        self.mask = ((ka.abs() < cut) & (kb.abs() < cut)).to(dtype) if dealias else torch.ones_like(self.k2)
        L = -nu * self.k2
        self.cn_a, self.cn_b = 1 + 0.5 * dt * L, 1 / (1 - 0.5 * dt * L)               # CN factors
        self.E, self.E2 = torch.exp(0.5 * dt * L), torch.exp(dt * L)                  # integrating factors

    # ------------------------------------------------------------------ transforms
    def fwd(self, w):
        return torch.fft.rfft2(w.to(self.device, self.dtype), norm="forward")

    def inv(self, wh, M=None):
        return torch.fft.irfft2(wh, s=(M or self.M, M or self.M), norm="forward")

    def resample(self, wh, N):
        """Fourier coefficients on an M grid -> coefficients on an N grid (truncate or zero-pad)."""
        M = wh.shape[-2]
        if N == M:
            return wh
        out = wh.new_zeros(*wh.shape[:-2], N, N // 2 + 1)
        h = min(M, N) // 2                                        # modes |k| < h are kept
        out[..., :h, :h] = wh[..., :h, :h]
        out[..., -h + 1:, :h] = wh[..., -h + 1:, :h]
        return out

    # ------------------------------------------------------------------ dynamics
    def explicit(self, wh, fh):
        """f_hat - FFT(u . grad w), de-aliased."""
        psih = wh * self.inv_k2
        ua = self.sign * self.inv(self.ikb * psih)
        ub = -self.sign * self.inv(self.ika * psih)
        wa, wb = self.inv(self.ika * wh), self.inv(self.ikb * wh)
        return fh - self.mask * self.fwd(ua * wa + ub * wb)

    def step(self, wh, fh):
        dt = self.dt
        if self.method == "cn_euler":
            return (self.cn_a * wh + dt * self.explicit(wh, fh)) * self.cn_b
        if self.method == "cn_heun":
            e0 = self.explicit(wh, fh)
            pred = (self.cn_a * wh + dt * e0) * self.cn_b
            return (self.cn_a * wh + 0.5 * dt * (e0 + self.explicit(pred, fh))) * self.cn_b
        E, E2 = self.E, self.E2                                    # ifrk4
        k1 = self.explicit(wh, fh)
        k2 = self.explicit(E * (wh + 0.5 * dt * k1), fh)
        k3 = self.explicit(E * wh + 0.5 * dt * k2, fh)
        k4 = self.explicit(E2 * wh + dt * E * k3, fh)
        return E2 * wh + dt / 6 * (E2 * k1 + 2 * E * (k2 + k3) + k4)

    @torch.no_grad()
    def solve(self, f, T=5.0, w0=None, out_res=None, save_times=()):
        """f: forcing [..., N, N] on any grid N <= M (resampled spectrally). Returns w(T) on out_res
        (default: the forcing's grid), plus {t: w(t)} for save_times if given."""
        N = f.shape[-1]
        out_res = out_res or N
        fh = self.resample(torch.fft.rfft2(f.to(self.device, self.dtype), norm="forward"), self.M)
        wh = torch.zeros_like(fh) if w0 is None else \
            self.resample(torch.fft.rfft2(w0.to(self.device, self.dtype), norm="forward"), self.M)
        n = round(T / self.dt)
        if abs(n * self.dt - T) > 1e-9:
            raise ValueError(f"T={T} is not a multiple of dt={self.dt}")
        saves = {round(t / self.dt): t for t in save_times}
        snaps = {}
        for i in range(1, n + 1):
            wh = self.step(wh, fh)
            if i in saves:
                snaps[saves[i]] = self.inv(self.resample(wh, out_res), out_res)
        w = self.inv(self.resample(wh, out_res), out_res)
        return (w, snaps) if save_times else w


def timed_solve(solver, f, T=5.0, repeats=3, warmup=1, **kw):
    """Wall-clock seconds per solve (median of repeats) with warm-up and CUDA synchronisation."""
    sync = (lambda: torch.cuda.synchronize()) if solver.device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        solver.solve(f, T=min(T, 50 * solver.dt), **kw)
    times = []
    for _ in range(repeats):
        sync(); t0 = time.perf_counter()
        out = solver.solve(f, T=T, **kw)
        sync(); times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], out


def rel_l2(a, b):
    """Per-sample ||a - b|| / ||b|| over the last two dims."""
    return (a - b).flatten(-2).norm(dim=-1) / b.flatten(-2).norm(dim=-1)


def forcing_spectrum_model(k, scale=27.0, tau2=9.0, power=4.0):
    """Eigenvalues of the forcing covariance scale * (-Lap + tau2)^(-power) at |k|."""
    return scale * (k ** 2 + tau2) ** (-power)
