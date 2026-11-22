"""Spectral diagnostics, same definitions as the main notebook's frequency section."""
import pandas as pd
import torch

BANDS = {"low (k < 8)": (0, 7), "mid (8 ≤ k ≤ 32)": (8, 32), "high (k > 32)": (33, 63)}


@torch.no_grad()
def radial_spectrum(f, chunk=256):
    """Radially binned power spectrum of fields f shaped [N, n, n]: |fft2|^2 averaged over samples,
    summed over rings k = round(|k|). Returns bins k = 0 ... n/2 - 1 (float64)."""
    N, n = f.shape[0], f.shape[-1]
    power = torch.zeros(n, n, dtype=torch.float64)
    for i in range(0, N, chunk):
        power += torch.fft.fft2(f[i:i + chunk].cpu().double()).abs().pow(2).sum(0)
    power /= N
    k1 = torch.fft.fftfreq(n) * n
    k = torch.sqrt(k1[:, None] ** 2 + k1[None, :] ** 2).round().long()
    return torch.bincount(k.flatten(), weights=power.flatten())[: n // 2]


def band_errors(preds, y_true):
    """preds: {name: [N, n, n]} in the same (raw) units as y_true. Relative error energy per band."""
    true_spec = radial_spectrum(y_true)
    rows = []
    for name, p in preds.items():
        err = radial_spectrum(p - y_true)
        rows.append({"model": name, **{b: (err[lo:hi + 1].sum() / true_spec[lo:hi + 1].sum()).item()
                                       for b, (lo, hi) in BANDS.items()}})
    return pd.DataFrame(rows)
