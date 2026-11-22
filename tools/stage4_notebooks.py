"""Build the Stage 4 solver-validation notebook (runs on CPU) and its Kaggle kernel folder."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_notebook import build, SETUP
from stage1_notebooks import kernel, ROOT

VALID = [
    ("md", "# Stage 4: own solver vs the dataset"),
    ("md", """
Checks whether our pseudo-spectral solver reproduces the dataset's operator $f \\mapsto \\omega(T)$.

$$\\partial_t \\omega + u\\cdot\\nabla\\omega = \\nu\\Delta\\omega + f,\\qquad u = \\nabla^\\perp\\psi,\\quad -\\Delta\\psi=\\omega,\\qquad \\omega(0)=0,\\ T=5$$

Facts used from the paper: $\\nu = 1/500$, $\\omega(0)=0$, $T=5$, forcing covariance $27(-\\Delta+9I)^{-4}$.
Not stated (tested here): axis convention, time step, solver grid.

Settings are chosen on **training** samples; the final agreement is reported on **test** samples that
played no part in any choice. Dataset fields and solver outputs are never mixed in later training.
"""),
    ("md", "## Setup"),
    ("code", SETUP + '''from solver import NS2D, timed_solve, rel_l2, forcing_spectrum_model
from nsbench.data import find_data_dir
torch.set_num_threads(os.cpu_count())
'''),
    ("md", "## Raw data (no normalization)"),
    ("code", '''
dd = find_data_dir()
train, test = torch.load(f"{dd}/nsforcing_train_128.pt"), torch.load(f"{dd}/nsforcing_test_128.pt")
xtr, ytr = train["x"].double(), train["y"].double()
xte, yte = test["x"].double(), test["y"].double()
pd.DataFrame([{"split": s, "x mean": x.mean().item(), "x std": x.std().item(), "y mean": y.mean().item(),
               "y std": y.std().item()} for s, x, y in (("train", xtr, ytr), ("test", xte, yte))])
'''),
    ("md", "## Spectrum of the input field"),
    ("md", """
If $x$ is the forcing, its per-mode power should follow $c\\,(|k|^2+9)^{-4}$ for some constant $c$
(the constant depends on the KL normalization, which the paper does not give).
"""),
    ("code", '''
def ring_power(f):
    """Mean |f_hat_k|^2 per mode (norm='forward'), averaged over samples, per ring |k| = round."""
    n = f.shape[-1]
    P = (torch.fft.fft2(f, norm="forward").abs() ** 2).mean(0)
    k1 = torch.fft.fftfreq(n, 1 / n)
    kr = torch.sqrt(k1[:, None] ** 2 + k1[None, :] ** 2).round().long().flatten()
    return (torch.bincount(kr, P.flatten()) / torch.bincount(kr).double())[: n // 2]

px, py = ring_power(xtr[:2000]), ring_power(ytr[:2000])
k = torch.arange(len(px), dtype=torch.float64)
model = forcing_spectrum_model(k) / 27                     # shape only
fit = slice(1, 21)
c = torch.exp((px[fit].log() - model[fit].log()).mean())
ratio = px / (c * model)
spec = pd.DataFrame({"k": k[1:33].long(), "x power": px[1:33], "model c(k^2+9)^-4": (c * model)[1:33],
                     "ratio": ratio[1:33], "y power": py[1:33]})
save_csv(spec, "stage4_input_spectrum")
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(k[1:64], px[1:64], label="input field x")
ax.loglog(k[1:64], (c * model)[1:64], "--", label="$c\\\\,(|k|^2+9)^{-4}$ fit on $1\\\\leq k\\\\leq 20$")
ax.loglog(k[1:64], py[1:64], label="output field y")
ax.set(xlabel="$|k|$", ylabel="mean power per mode"); ax.legend(); fig.tight_layout()
save_fig("stage4_input_spectrum", fig)
pd.DataFrame([{"fitted c / 27": c.item(), "ratio range k=1..20": f"{ratio[fit].min():.3f} to {ratio[fit].max():.3f}",
               "ratio range k=21..40": f"{ratio[21:41].min():.3g} to {ratio[21:41].max():.3g}"}])
'''),
    ("md", "## Axis convention and viscosity (8 training samples)"),
    ("code", '''
idx = torch.arange(8)
rows = []
for sign in (1.0, -1.0):
    for nu in (1 / 500, 1 / 1000):
        s = NS2D(128, nu=nu, dt=1e-3, method="cn_heun", sign=sign)
        e = rel_l2(s.solve(xtr[idx], T=5.0), ytr[idx])
        rows.append({"sign": sign, "nu": nu, "rel L2 mean": e.mean().item(), "rel L2 max": e.max().item()})
hyp = pd.DataFrame(rows)
save_csv(hyp, "stage4_hypotheses")
SIGN = float(hyp.loc[hyp["rel L2 mean"].idxmin(), "sign"])
NU = float(hyp.loc[hyp["rel L2 mean"].idxmin(), "nu"])
hyp
'''),
    ("md", "## Time-step and grid convergence (8 training samples)"),
    ("code", '''
ref = NS2D(256, nu=NU, dt=1e-3, method="ifrk4", sign=SIGN).solve(xtr[idx], T=5.0, out_res=128)
rows = []
for method, M, dt in [("cn_euler", 128, 2e-3), ("cn_euler", 128, 1e-3), ("cn_heun", 128, 2e-2),
                      ("cn_heun", 128, 1e-2), ("cn_heun", 128, 4e-3), ("cn_heun", 128, 1e-3),
                      ("ifrk4", 128, 2e-3), ("cn_heun", 256, 4e-3)]:
    s = NS2D(M, nu=NU, dt=dt, method=method, sign=SIGN)
    w = s.solve(xtr[idx], T=5.0, out_res=128)
    rows.append({"method": method, "grid": M, "dt": dt,
                 "vs reference (256, ifrk4, dt 1e-3)": rel_l2(w, ref).mean().item(),
                 "vs dataset y": rel_l2(w, ytr[idx]).mean().item()})
rows.append({"method": "ifrk4", "grid": 256, "dt": 1e-3, "vs reference (256, ifrk4, dt 1e-3)": 0.0,
             "vs dataset y": rel_l2(ref, ytr[idx]).mean().item()})
conv = pd.DataFrame(rows)
save_csv(conv, "stage4_convergence")
conv
'''),
    ("md", "## Coarse solves: accuracy vs grid and time step (16 training samples)"),
    ("md", """
A solver run on a coarser grid is cheaper and less accurate. The speedup of a surrogate must be judged
against the cheapest solver setting that reaches the surrogate's accuracy, so this table maps the
trade-off (forcing truncated to the grid, output zero-padded back to $128^2$). Timing is in Stage 5.
"""),
    ("code", '''
idx16 = torch.arange(16)
rows = []
for M in (16, 24, 32, 48, 64, 96, 128):
    for dt in (1e-2, 2e-2, 5e-2, 1e-1):
        w = NS2D(M, nu=NU, dt=dt, method="cn_heun", sign=SIGN).solve(xtr[idx16], T=5.0, out_res=128)
        e = rel_l2(w, ytr[idx16])
        rows.append({"grid": M, "dt": dt, "steps": round(5 / dt), "rel L2 mean": e.mean().item(),
                     "rel L2 max": e.max().item(), "stable": bool(torch.isfinite(e).all() and e.max() < 1)})
coarse = pd.DataFrame(rows)
save_csv(coarse, "stage4_coarse_accuracy")
coarse.pivot(index="grid", columns="dt", values="rel L2 mean")
'''),
    ("md", "## Agreement on held-out test samples"),
    ("code", '''
N_TEST = 50
cfg = dict(M=128, nu=NU, dt=1e-3, method="cn_heun", sign=SIGN)
w_te = torch.cat([NS2D(**cfg).solve(xte[i:i + 10], T=5.0) for i in range(0, N_TEST, 10)])
e = rel_l2(w_te, yte[:N_TEST])
agree = pd.DataFrame([{**cfg, "test samples": N_TEST, "rel L2 mean": e.mean().item(), "rel L2 median": e.median().item(),
                       "rel L2 max": e.max().item(), "rel L2 min": e.min().item()}])
save_csv(pd.DataFrame({"sample": range(N_TEST), "rel_L2": e.numpy()}), "stage4_test_per_sample")
save_csv(agree, "stage4_test_agreement")
agree
'''),
    ("md", "## Example fields"),
    ("code", '''
fig, ax = plt.subplots(2, 4, figsize=(14, 6.5))
for r, i in enumerate((int(e.argmin()), int(e.argmax()))):
    for c_, (img, t) in enumerate([(xte[i], "input field x"), (yte[i], "dataset y"), (w_te[i], "our solver"),
                                   (w_te[i] - yte[i], f"difference (rel L2 {e[i]:.2e})")]):
        im = ax[r, c_].imshow(img, cmap="RdBu_r"); ax[r, c_].set_title(f"test {i}: {t}", fontsize=9)
        ax[r, c_].axis("off"); fig.colorbar(im, ax=ax[r, c_], fraction=0.046)
fig.tight_layout()
save_fig("stage4_examples", fig)
'''),
]

if __name__ == "__main__":
    nb = ROOT / "notebooks/stage4_solver_validation.ipynb"
    build(VALID, nb)
    # runs locally on CPU (tools/run_local.py); no Kaggle kernel needed
