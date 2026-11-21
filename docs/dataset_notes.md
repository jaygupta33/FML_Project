# Dataset notes: `nsforcing_128` (Zenodo 12825163)

Status: **paper definition found; numerical confirmation still pending (Stage 4).**
Labels: *[paper]* = stated in arXiv 2310.00120; *[code]* = neuraloperator source; *[measured]* = our notebook; *[not stated]* = no source found.

## What the files contain
| Item | Value | Source |
|---|---|---|
| Record / archive | Zenodo 12825163, `nsforcing_128.tgz` (1,450,200,928 B, md5 `70a389207ac93935d5ff3d4289d43581`) | [measured] |
| Files | `nsforcing_train_128.pt`, `nsforcing_test_128.pt`, dicts with keys `x`, `y` | [measured], [code] |
| Shapes | train `[10000,128,128]`, test `[2000,128,128]` | [measured] |
| Stats | `x` std 0.154, `y` std 0.708 (train), means ≈ 0 | [measured] |
| Resolutions offered by loader | 128 and 1024 | [code] `available_resolutions = [128, 1024]` |

## The problem the data comes from
arXiv 2310.00120 (Kossaifi et al., *Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs*), Sec. 4.1, Eq. (6):

$$\partial_t \omega + \nabla^\perp \phi \cdot \nabla\omega = \tfrac{1}{Re}\,\Delta\omega + f,\qquad -\Delta\phi = \omega,\quad \textstyle\int \phi = 0,\qquad x\in\mathbb{T}^2,\ t\in(0,T]$$

| Item | Value | Source |
|---|---|---|
| Domain | torus $\mathbb{T}^2 \cong [0,2\pi)^2$ | [paper] Sec. 4.1 |
| Reynolds number | $Re = 500$ (viscous term $\tfrac{1}{Re}\Delta\omega$, i.e. $\nu = 1/500$) | [paper] |
| Initial condition | $\omega(0,\cdot) = 0$ | [paper] |
| Operator learned | $f \mapsto \omega(T,\cdot)$, $T = 5$ | [paper] |
| Forcing distribution | Gaussian $\mathcal{N}(0, C)$, $C = 27(-\Delta + 9I)^{-4}$, sampled by KL expansion on eigenfunctions of $C$; $f \in \dot L^2$ (zero mean) | [paper], citing De Hoop et al. 2022 |
| Solver | "pseudo-spectral scheme" (Chandler & Kerswell 2013) | [paper] |
| Samples | 10K train, 2K test | [paper] = [measured] |
| Time integrator, `dt`, dealiasing | — | [not stated] (Heun is named only for the paper's Burgers data) |
| Native grid, downsampling to 128 | — | [not stated] |
| KL truncation / number of modes, extra amplitude scaling | — | [not stated] |

**Conflicting source:** the neuraloperator `NavierStokesDataset` docstring says *"Input and output are both 2d fields with one channel of data which describes the vorticity at each point."* It gives no time, Re, domain or forcing, and the repo contains no data-generation code. The paper is the more specific source, so the working reading is **`x` = forcing `f`, `y` = vorticity at `T = 5` starting from rest** — not yet verified.

## How to confirm (Stage 4)
1. **Spectrum check (cheap, no solver):** if `x ~ N(0, 27(-Δ+9I)^{-4})`, its radially averaged power spectrum should follow $(|k|^2+9)^{-4}$ up to a constant, and the constant fixes the amplitude convention (our measured std 0.154 vs the value implied by the KL normalisation used).
2. **Solver check (decisive):** run our pseudo-spectral solver from $\omega = 0$ with forcing `x`, $\nu = 1/500$, domain $[0,2\pi)^2$, to $T = 5$, on ≥ 20 test samples; compare with `y` by relative $L_2$. Converge `dt` first. Unknown integrator/`dt` in the paper means small differences are expected; large ones mean a wrong reading (domain, scaling, T, or what `x` is).

Until both checks pass, notebook text says only "input field `x`" and "output field `y`".

## Sources
- arXiv 2310.00120, HTML v1, Sec. 4.1 "Data", 4.2 (read 2026-11-21).
- `neuralop/data/datasets/navier_stokes.py` (neuraloperator `main`, read 2026-11-21).
- Zenodo record 12825163.
