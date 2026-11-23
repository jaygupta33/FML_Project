# Dataset notes: `nsforcing_128` (Zenodo 12825163)

Status: **confirmed by Stage 4 (2026-11-22): `x` is the forcing $f$, `y` is $\omega(T=5)$ from rest, $\nu=1/500$.**
Labels: *[paper]* = stated in arXiv 2310.00120; *[code]* = neuraloperator source; *[measured]* = our notebook; *[not stated]* = no source found; *[Stage 4]* = measured with our solver (`notebooks/stage4_solver_validation.ipynb`).

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

**Conflicting source:** the neuraloperator `NavierStokesDataset` docstring says *"Input and output are both 2d fields with one channel of data which describes the vorticity at each point."* It gives no time, Re, domain or forcing, and the repo contains no data-generation code. The paper is the more specific source; Stage 4 confirms its reading (below). The docstring is wrong for this dataset.

## Stage 4 confirmation
| Check | Result | Source |
|---|---|---|
| Spectrum of `x` | mean power per mode follows $c(|k|^2+9)^{-4}$; ratio 0.85–1.07 for $1\le|k|\le31$ | [Stage 4] |
| Axis convention | $u = (\partial_b\psi, -\partial_a\psi)$ for array axes $(a,b)$; the swapped convention gives rel. $L^2$ 0.86 | [Stage 4] |
| Viscosity | $\nu=1/500$: rel. $L^2$ $2.2\times10^{-5}$; $\nu=1/1000$: $2.2\times10^{-2}$ (8 train samples) | [Stage 4] |
| Agreement on 50 **test** samples | rel. $L^2$ mean $2.7\times10^{-5}$, median $1.7\times10^{-5}$, max $2.1\times10^{-4}$ (128², CN–Heun, dt $10^{-3}$) | [Stage 4] |
| Native grid | 256² halves the remaining error ($3.7\times10^{-6}$); 512² gives no further change, so the data were likely generated on ≥256² | [Stage 4] |
| Integrator | not identifiable: CN–Heun and IF-RK4 agree to $<10^{-6}$ at these settings | [Stage 4] |

Settings were chosen on training samples; the test agreement was measured once, afterwards.
Notebook text can now say "forcing $f$" for `x` and "vorticity $\omega(T{=}5)$" for `y`.

## Sources
- arXiv 2310.00120, HTML v1, Sec. 4.1 "Data", 4.2 (read 2026-11-21).
- `neuralop/data/datasets/navier_stokes.py` (neuraloperator `main`, read 2026-11-21).
- Zenodo record 12825163.
