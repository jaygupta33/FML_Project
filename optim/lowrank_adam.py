"""Adam with low-rank gradient projection (GaLore-style), fixed or adaptive rank, complex-aware.

Large weight tensors are reshaped to a matrix G (m x n, m = first dim). Along the smaller side we keep
an orthonormal basis P (m x r) of G's top-r left singular vectors and run Adam on the compressed
gradient R = P^H G (r x n). The update P @ Adam(R) is mapped back to the weight's shape.
Small tensors use ordinary Adam. Which tensors are compressed is decided by shape/size only,
because the Trainer passes model.parameters() without names.

rank modes
  fixed      r is constant (GaLore baseline)
  adaptive   at every refresh, r = smallest k whose cumulative gradient energy >= tau
             (energy spectrum smoothed with an EMA across refreshes), clamped to [r_min, r_max]

Adam's second moment is element-wise on the real and imaginary parts (exactly what torch.optim.Adam
does for complex tensors), so with an identity basis the optimizer reproduces torch.optim.Adam.

Everything the method needs to keep (Adam moments, P, smoothed spectrum) lives in self.state,
so a Trainer that sums self.state tensors reports the true optimizer-state memory.
"""
import math

import torch
from torch.optim import Optimizer


def _as_real(t):
    return torch.view_as_real(t) if t.is_complex() else t


class LowRankAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 rank=8, rank_mode="fixed", tau=0.99, r_min=2, r_max=None,
                 refresh=200, refresh_early=None, early_steps=0, spectrum_beta=0.5,
                 scale=1.0, on_change="keep", min_numel=100_000, proj="svd"):
        """
        rank           fixed rank (rank_mode="fixed") or starting rank (unused for adaptive)
        tau            energy threshold for rank_mode="adaptive"
        refresh        steps between basis refreshes; refresh_early is used for the first early_steps
        spectrum_beta  EMA weight on the previous smoothed energy spectrum (0 = no smoothing)
        scale          GaLore update scale alpha
        on_change      what happens to Adam moments when the basis is refreshed:
                       keep   (official GaLore; rows padded with zeros / truncated if r changes)
                       reset  (zero moments, restart bias correction)
                       rotate (m <- (P_new^H P_old) m, v <- |P_new^H P_old|^2 v)
        min_numel      tensors with at least this many elements (and ndim >= 2) are compressed
        proj           "svd" or "identity" (identity basis, full rank; for testing only)
        """
        if rank_mode not in ("fixed", "adaptive"):
            raise ValueError(rank_mode)
        if on_change not in ("keep", "reset", "rotate"):
            raise ValueError(on_change)
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.cfg = dict(rank=rank, rank_mode=rank_mode, tau=tau, r_min=r_min, r_max=r_max, refresh=refresh,
                        refresh_early=refresh_early or refresh, early_steps=early_steps,
                        spectrum_beta=spectrum_beta, scale=scale, on_change=on_change,
                        min_numel=min_numel, proj=proj)
        self.global_step = 0
        self.rank_log = []          # (step, param index, rank, energy captured): diagnostics, not state

    # ------------------------------------------------------------------ helpers
    def compressed(self, p):
        return p.ndim >= 2 and p.numel() >= self.cfg["min_numel"]

    @staticmethod
    def _adam_dir(m, v, step, b1, b2, eps):
        """Bias-corrected Adam direction, same arithmetic as torch.optim.Adam (non-foreach)."""
        bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
        denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
        return (m / bc1) / denom

    def _basis(self, G, st):
        """Return (P, r, energy captured) for matrix G, updating the smoothed spectrum in st."""
        m = G.shape[0]
        c = self.cfg
        if c["proj"] == "identity":
            return torch.eye(m, dtype=G.dtype, device=G.device), m, 1.0
        # eigen-decomposition of G G^H (m x m) is much cheaper than an SVD of G (m x n, n >> m)
        evals, evecs = torch.linalg.eigh(G @ G.mH)
        energy = evals.real.flip(0).clamp_min(0)                  # sigma^2, descending
        U = evecs.flip(1)
        total = energy.sum().clamp_min(1e-30)
        if c["rank_mode"] == "adaptive":
            if "spectrum" in st:
                st["spectrum"].mul_(c["spectrum_beta"]).add_(energy / total, alpha=1 - c["spectrum_beta"])
            else:
                st["spectrum"] = energy / total
            s = st["spectrum"]
            cum = s.cumsum(0) / s.sum().clamp_min(1e-30)
            r = int((cum < c["tau"]).sum().item()) + 1
            r = max(c["r_min"], min(r, c["r_max"] or m, m))
        else:
            r = min(c["rank"], m)
        captured = float(energy[:r].sum() / total)
        return U[:, :r].contiguous(), r, captured

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        c = self.cfg
        self.global_step += 1
        gap = c["refresh_early"] if self.global_step <= c["early_steps"] else c["refresh"]
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not self.compressed(p):                        # ordinary Adam
                    if not st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                    st["step"] += 1
                    m, v, gr = _as_real(st["exp_avg"]), _as_real(st["exp_avg_sq"]), _as_real(g)
                    m.lerp_(gr, 1 - b1)
                    v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                    p.add_(torch.view_as_complex(self._adam_dir(m, v, st["step"], b1, b2, group["eps"]))
                           if p.is_complex() else self._adam_dir(m, v, st["step"], b1, b2, group["eps"]),
                           alpha=-group["lr"])
                    continue

                G = g.reshape(g.shape[0], -1)
                transpose = G.shape[0] > G.shape[1]               # project along the smaller side
                if transpose:
                    G = G.mT.conj() if G.is_complex() else G.mT
                if "P" not in st or (self.global_step - 1) % gap == 0:
                    P_new, r, captured = self._basis(G, st)
                    self.rank_log.append((self.global_step, idx, r, captured))
                    if "P" not in st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros(r, G.shape[1], dtype=G.dtype, device=G.device)
                        st["exp_avg_sq"] = torch.zeros(r, G.shape[1], dtype=G.dtype, device=G.device)
                    else:
                        self._change_basis(st, st["P"], P_new)
                    st["P"] = P_new
                P = st["P"]
                R = P.mH @ G                                      # r x n compressed gradient
                st["step"] += 1
                m, v, Rr = _as_real(st["exp_avg"]), _as_real(st["exp_avg_sq"]), _as_real(R)
                m.lerp_(Rr, 1 - b1)
                v.mul_(b2).addcmul_(Rr, Rr, value=1 - b2)
                N = self._adam_dir(m, v, st["step"], b1, b2, group["eps"])
                N = torch.view_as_complex(N) if R.is_complex() else N
                U = P @ N                                         # back to full space
                if transpose:
                    U = U.mT.conj() if U.is_complex() else U.mT
                p.add_(U.reshape(p.shape), alpha=-group["lr"] * c["scale"])
        return loss

    def _change_basis(self, st, P_old, P_new):
        mode = self.cfg["on_change"]
        r_old, r_new = P_old.shape[1], P_new.shape[1]
        if mode == "reset":
            n = st["exp_avg"].shape[1]
            st["exp_avg"] = st["exp_avg"].new_zeros(r_new, n)
            st["exp_avg_sq"] = st["exp_avg_sq"].new_zeros(r_new, n)
            st["step"] = 0
        elif mode == "rotate":
            C = P_new.mH @ P_old                                  # r_new x r_old
            st["exp_avg"] = C @ st["exp_avg"]
            W = (C.abs() ** 2).to(_as_real(st["exp_avg_sq"]).dtype)
            v = _as_real(st["exp_avg_sq"])                        # r_old x n (x 2 if complex)
            v_new = torch.einsum("ab,bn...->an...", W, v)
            st["exp_avg_sq"] = torch.view_as_complex(v_new.contiguous()) if v.ndim == 3 else v_new
        elif r_new != r_old:                                      # keep: pad with zeros / truncate rows
            for k in ("exp_avg", "exp_avg_sq"):
                t = st[k]
                st[k] = t[:r_new] if r_new < r_old else torch.cat([t, t.new_zeros(r_new - r_old, t.shape[1])])

    # ------------------------------------------------------------------ reporting
    def ranks(self):
        """Current rank of every compressed tensor, in parameter order."""
        return [st["P"].shape[1] for group in self.param_groups for p in group["params"]
                if (st := self.state.get(p)) and "P" in st]
