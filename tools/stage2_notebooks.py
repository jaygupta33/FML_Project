"""Build the Stage 2 screening notebook and its Kaggle kernel folder (kaggle/stage2_screen)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_notebook import build, SETUP, DATA
from stage1_notebooks import kernel, ROOT

SCREEN = [
    ("md", "# Stage 2: optimizer screening on the FNO"),
    ("md", """
Short screening runs to choose finalists. All runs share the model, data split, seed, epochs and
learning rate; only the optimizer changes.

| setting | value |
|---|---|
| model | FNO, 2,410,177 params; the four $64\\times64\\times16\\times9$ complex spectral weights are compressed |
| matrix view | $G \\in \\mathbb{C}^{64 \\times 9216}$, projected along the 64 side |
| lr / scale | $10^{-3}$ / 1 for every optimizer (a single lr, no per-optimizer tuning) |
| basis refresh | every 200 steps (250 steps per epoch) |
| epochs / batch / seed | 10 / 32 / 0 |

| optimizer | rank rule |
|---|---|
| Adam | none (full state) |
| GaLore fixed | $r \\in \\{4, 8, 16\\}$ |
| adaptive | smallest $r$ with cumulative gradient energy $\\ge \\tau$, $\\tau \\in \\{0.95, 0.99, 0.999\\}$, $r \\in [2, 64]$ |
"""),
    ("md", "## Setup"),
    ("code", SETUP + "from optim import LowRankAdam\n"),
    ("md", "## Data"),
    ("code", DATA),
    ("md", "## Configurations"),
    ("code", '''
EPOCHS, LR, SEED, REFRESH = 10, 1e-3, 0, 200
CONFIGS = {"Adam": dict()}
CONFIGS.update({f"GaLore r={r}": dict(rank_mode="fixed", rank=r) for r in (4, 8, 16)})
CONFIGS.update({f"adaptive tau={t}": dict(rank_mode="adaptive", tau=t, r_min=2, r_max=64) for t in (0.95, 0.99, 0.999)})

def make_opt(cfg):
    if not cfg:
        return lambda p: torch.optim.Adam(p, lr=LR)
    return lambda p: LowRankAdam(p, lr=LR, refresh=REFRESH, **cfg)

pd.DataFrame([{"name": k, **v} for k, v in CONFIGS.items()])
'''),
    ("md", "## Training"),
    ("code", '''
rows, hists, ranklogs, mem = [], [], [], []
for name, cfg in CONFIGS.items():
    tr = Trainer(nsbench.FourierNeuralOperator, make_opt(cfg), D.train_loader, D.val_loader, D.test_loader,
                 D.y_mean, D.y_std, device, seed=SEED, label=f"FNO {name}").fit(EPOCHS)
    rows.append(run_record(tr.summary(), run="FNO", optimizer=name, lr=LR, seed=SEED, epochs=EPOCHS,
                           batch=32, loss_fn="MSE", monitor="loss", refresh=REFRESH if cfg else None, **cfg))
    hists.append(tr.history().assign(optimizer=name))
    # optimizer-state memory split by role
    split = {"projection P": 0, "compressed moments": 0, "full-size moments": 0}
    for p, st in tr.opt.state.items():
        for k, t in st.items():
            if torch.is_tensor(t) and t.ndim > 0:
                role = "projection P" if k == "P" else "spectrum" if k == "spectrum" else \\
                       "compressed moments" if "P" in st else "full-size moments"
                split[role] = split.get(role, 0) + t.numel() * t.element_size()
    mem.append({"optimizer": name, **{k: v / 1e6 for k, v in split.items()}})
    if isinstance(tr.opt, LowRankAdam):
        ranklogs.append(pd.DataFrame(tr.opt.rank_log, columns=["step", "param", "rank", "energy"]).assign(optimizer=name))
    del tr; torch.cuda.empty_cache()
screen = pd.concat(rows, ignore_index=True)
save_csv(screen, "stage2_screen")
save_csv(pd.concat(hists, ignore_index=True), "stage2_screen_history")
save_csv(pd.concat(ranklogs, ignore_index=True), "stage2_screen_ranks")
save_csv(pd.DataFrame(mem).fillna(0), "stage2_screen_memory_split")
screen[["optimizer", "best_epoch", "best_val_loss", "test_rel_L2", "opt_state_MB", "peak_gpu_MB", "sec_per_epoch"]]
'''),
    ("md", "## Optimizer-state memory by role (MB)"),
    ("code", '''
pd.DataFrame(mem).fillna(0).set_index("optimizer")
'''),
    ("md", "## Rank over training (adaptive runs)"),
    ("code", '''
R = pd.concat(ranklogs, ignore_index=True)
summary = R.groupby(["optimizer", "param"]).agg(first_rank=("rank", "first"), last_rank=("rank", "last"),
                                                 mean_rank=("rank", "mean"), mean_energy=("energy", "mean"))
fig, ax = plt.subplots(figsize=(7, 4))
for name, r in R[R.optimizer.str.startswith("adaptive")].groupby("optimizer"):
    m = r.groupby("step")["rank"].mean()
    ax.plot(m.index / 250, m.values, marker="o", ms=3, label=name)
ax.set(xlabel="epoch", ylabel="rank (mean over 4 spectral weights)"); ax.legend(); fig.tight_layout()
save_fig("stage2_adaptive_ranks", fig)
summary
'''),
    ("md", "## Validation curves"),
    ("code", '''
H = pd.concat(hists, ignore_index=True)
fig, ax = plt.subplots(figsize=(7, 4))
for name, h in H.groupby("optimizer", sort=False):
    ax.plot(h.epoch, h.val_rel_L2, label=name, ls="-" if name == "Adam" else "--" if "GaLore" in name else ":")
ax.set(xlabel="epoch", ylabel="validation relative $L^2$", yscale="log"); ax.legend(fontsize=8); fig.tight_layout()
save_fig("stage2_screen_curves", fig)
'''),
]

if __name__ == "__main__":
    nb = ROOT / "notebooks/stage2_screen.ipynb"
    build(SCREEN, nb)
    kernel("stage2_screen", "ns-stage2-screen", "NS Stage2 screen", nb)
