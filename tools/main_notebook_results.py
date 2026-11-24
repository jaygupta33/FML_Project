"""Add the stage results to navier-stokes.ipynb, so the whole project reads in one notebook.

The new cells (ids starting with "res-") only load saved results from results/ and figures/; no training.
They are executed here in a fresh kernel (cwd = repo root) and their outputs are spliced into the main
notebook, so the earlier cells and their outputs are left untouched. Re-running replaces the "res-" cells.
"""
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "navier-stokes.ipynb"
SUMMARY_ID = "a7c3f1e2"           # closing summary cell, moved to the end of the new part

CELLS = [
    ("md", "# Part 2: The Stage Experiments"),
    ("md", r"""
The sections below show the saved results of the follow-up experiments. Each stage was run by its own notebook in
`notebooks/`; here we only **load** what those runs saved in `results/` and `figures/`, so these cells take seconds
and need no GPU. Run them from the repository root.

| Stage | What it measures | Notebook | Ran on |
|---|---|---|---|
| 1A | FNO + Adam, 3 seeds | `stage1_fno_seeds` | Kaggle T4 |
| 1B | U-Net learning-rate sweep, spectral loss | `stage1_unet_control` | Kaggle T4 |
| 2 | optimizer screen (10 epochs), finalists (20 epochs × 3 seeds) | `stage2_screen`, `stage2_final_a/b` | Kaggle T4 |
| 3 | $K=3$ ensembles from saved test predictions | `stage3_ensembles` | local CPU |
| 4 | our solver against the dataset | `stage4_solver_validation` | local CPU |
| 5 | solver cost-accuracy frontier vs surrogate inference time | `stage5_timing` | Kaggle T4 |
| 5b | FNO inference in half precision and compiled | `stage5b_fp16` | Kaggle T4 |
| 7 | active learning: uncertainty vs random selection | `stage7_al_uncertainty`, `stage7_al_random` | Kaggle T4 |

Unless stated otherwise, accuracy is relative $L_2$ error on the 2,000 test samples, in raw units.
"""),
    ("code", '''
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

RES, FIG = Path("results"), Path("figures")
assert RES.is_dir(), "run this part from the repository root, where results/ lives"

def load(name, cols=None, digits=4):
    df = pd.read_csv(RES / f"{name}.csv")
    return (df[cols] if cols else df).round(digits)

def show_fig(path, width=720):
    display(Image(filename=str(FIG / path), width=width))
'''),

    # ---------------------------------------------------------------- Stage 1A
    ("md", "## Stage 1A: FNO + Adam Over Three Seeds"),
    ("md", r"""
The FNO baseline from Part 1, re-trained with seeds 0, 1 and 2 (Adam, lr $10^{-3}$, MSE, 20 epochs, batch 32).
This is the reference every optimizer below is compared against.
"""),
    ("code", '''
COLS = ["seed", "best_epoch", "test_rel_L2", "test_mse_norm", "opt_state_MB", "peak_gpu_MB", "sec_per_epoch"]
load("stage1_fno_adam_seeds", COLS)
'''),
    ("code", '''
load("stage1_fno_adam_mean_std", digits=5).rename(columns={"Unnamed: 0": "metric"})
'''),
    ("code", 'show_fig("stage1_fno_adam_curves.png")'),
    ("md", r"""
- **Test error:** $0.0364 \pm 0.0019$ (seeds range from 0.0342 to 0.0377).
- **Memory:** Adam's state is 38.2 MB, twice the 19.1 MB of parameters. Peak GPU memory is about 4 GB.
"""),

    # ---------------------------------------------------------------- Stage 1B
    ("md", "## Stage 1B: A Fair U-Net Control"),
    ("md", r"""
The U-Net + MSE was re-trained at three learning rates and the best was picked by validation MSE. The `k2` spectral
loss ($\lambda = 0.5$) was then trained at that rate. Seed 0, 20 epochs. The results are discussed in
*Results: Spectral Loss vs MSE* above.
"""),
    ("code", '''
COLS = ["label", "lr", "best_epoch", "best_val_loss", "test_rel_L2", "sec_per_epoch"]
pd.concat([load("stage1_unet_lr_sweep", COLS),
           load("stage1_unet_control", COLS).iloc[1:]], ignore_index=True)
'''),
    ("md", "Error energy divided by true energy, per frequency band:"),
    ("code", 'load("stage1_unet_band_errors")'),
    ("code", 'show_fig("final/fig_unet_bands.png", width=520)'),

    # ---------------------------------------------------------------- Stage 2 screen
    ("md", "## Stage 2: The Low-Rank Optimizer"),
    ("md", r"""
`optim/lowrank_adam.py` implements low-rank Adam for the FNO's four complex spectral weights (reshaped to
$64 \times 9216$). Every 200 steps it computes a basis $P$ from the gradient, $G \approx P P^{H} G$, and keeps
Adam's moments for $R = P^{H} G$ only. All other tensors use ordinary Adam.

| Mode | Rank $r$ per spectral weight |
|---|---|
| GaLore (fixed) | the same $r$ for every weight, all run long |
| adaptive | smallest $r \in [2, 64]$ holding a share $\tau$ of a running average of the gradient energy, chosen per weight at each refresh |

With the full rank, it reproduces `torch.optim.Adam` exactly (unit test in `tests/test_lowrank_adam.py`).
"""),
    ("md", "## Stage 2a: Screen (10 epochs, seed 0)"),
    ("code", '''
COLS = ["optimizer", "best_epoch", "test_rel_L2", "opt_state_MB", "peak_gpu_MB", "sec_per_epoch"]
load("stage2_screen", COLS)
'''),
    ("md", "Where the optimizer state goes, in MB (end of training):"),
    ("code", 'load("stage2_screen_memory_split")'),
    ("code", 'show_fig("stage2_screen_curves.png")'),
    ("md", r"""
- **Ranking.** After 10 epochs, accuracy follows the rank: $r=16$ (0.0522) and $\tau=0.999$ (0.0530) are closest to
  Adam (0.0456).
- **What was dropped.** $\tau=0.95$ was the least accurate run (0.0639), so it was dropped.
- **What went on.** The other five optimizers became the finalists.
- **Where the memory is.** The compressed moments are almost all of the low-rank state. The uncompressed small
  tensors add 0.41 MB.
"""),

    # ---------------------------------------------------------------- Stage 2 finalists
    ("md", "## Stage 2b: Finalists (20 epochs × 3 seeds)"),
    ("md", r"""
Same protocol as Stage 1A, with seeds 0, 1 and 2. For the adaptive runs, the state changes during training. The
**peak** is reconstructed from the logged ranks (MB per unit rank, checked against the fixed-rank runs), and it is
what an ensemble has to budget for.
"""),
    ("code", '''
COLS = ["optimizer", "seeds", "test_rel_L2_mean", "test_rel_L2_std", "opt_state_MB_peak", "opt_state_MB_end",
        "peak_gpu_MB", "sec_per_epoch"]
load("final_optimizers", COLS)
'''),
    ("code", 'show_fig("final/fig_accuracy_vs_state.png", width=560)'),
    ("code", 'show_fig("final/fig_val_curves.png")'),
    ("code", 'show_fig("final/fig_adaptive_rank.png", width=600)'),
    ("md", r"""
- **Memory vs accuracy.** Low-rank Adam cuts optimizer state 4–14× at a cost of 6–28% in test error.
- **Peak GPU memory barely changes** (about 2%): parameters plus optimizer state are under 60 MB of a peak near
  4 GB.
- **Adaptive rank does not beat fixed rank** once the peak state is counted:
  - $\tau=0.999$ needs 13.7 MB at its peak for 0.0388, while $r=16$ gets 0.0386 with 9.9 MB.
  - $\tau=0.99$ (4.6 MB) is less accurate than $r=8$ (5.1 MB).
- **How the adaptive rank behaves.** For $\tau=0.999$, the mean rank per spectral weight is about 12 in epoch 1,
  peaks at about 19 in epoch 2, then falls to about 10 by epoch 20.
- **Speed.** Time per epoch is the same within 3% for all optimizers.
"""),

    # ---------------------------------------------------------------- Stage 3
    ("md", "## Stage 3: Ensembles"),
    ("md", r"""
Each ensemble averages the test predictions of the three seeds of one optimizer:
$\bar y = \frac{1}{K}\sum_k \hat y_k$ with $K = 3$. The member spread, the per-pixel standard deviation across
members, is the uncertainty estimate.
"""),
    ("code", 'load("stage3_ensembles")'),
    ("code", 'load("stage3_band_errors")'),
    ("code", 'show_fig("final/fig_spread_error.png")'),
    ("md", r"""
- **Accuracy.** Averaging three members lowers the error by 14–16% for every optimizer. The Adam ensemble reaches
  0.0310, and the $r=16$ ensemble 0.0333.
- **Uncertainty ranks the hard samples.** Spread and error are rank-correlated, with Spearman 0.55–0.66.
- **But it is too small.** The spread is only 0.70–0.78 of the actual error.
"""),

    # ---------------------------------------------------------------- Stage 4
    ("md", "## Stage 4: Our Solver Reproduces the Dataset"),
    ("md", r"""
`solver/ns2d.py` is a batched pseudo-spectral solver for 2D Navier-Stokes in vorticity form (periodic grid, 2/3
dealiasing). We run it from rest with `x` as the forcing and compare $\omega(T{=}5)$ with `y`. Settings were chosen
on training samples. The test comparison was run once, afterwards.
"""),
    ("md", "Which reading of the data fits (8 training samples). `sign` $= -1$ is the swapped axis convention:"),
    ("code", 'load("stage4_hypotheses", digits=6)'),
    ("md", "Agreement on 50 test samples:"),
    ("code", 'load("stage4_test_agreement", digits=6)'),
    ("md", "Sensitivity to the time step and method (training samples):"),
    ("code", 'load("stage4_convergence", digits=6)'),
    ("code", 'show_fig("stage4_input_spectrum.png", width=560)'),
    ("md", r"""
- **What the data are.** Only the standard axis convention with $\nu = 1/500$ fits: error $2.2 \times 10^{-5}$,
  against $2.2 \times 10^{-2}$ for $\nu = 1/1000$ and 0.86 for the swapped axes.
- **Test agreement.** Mean error $2.7 \times 10^{-5}$, worst $2.1 \times 10^{-4}$. So `x` is the forcing and `y` is
  $\omega(T{=}5)$.
- **Forcing spectrum.** The power of `x` per mode follows $c\,(|k|^2 + 9)^{-4}$ within a factor of 0.85–1.07.
"""),

    # ---------------------------------------------------------------- Stage 5
    ("md", "## Stage 5: Surrogate vs Solver Speed"),
    ("md", r"""
The solver was timed on a T4 at grids from $24^2$ to $256^2$, each with its accuracy against `y` on 64 test
samples. FNO and U-Net inference was timed with the same protocol: warm-up, `cuda.synchronize`, repeats, and batch
sizes 1, 32 and 256. Below is the best time per sample over the batch sizes.
"""),
    ("code", '''
t = pd.read_csv(RES / "stage5_timing.csv")
best = t.loc[t.groupby(["what", "precision"])["sec_per_sample"].idxmin()]
best = best.assign(ms_per_sample=best.sec_per_sample * 1e3).sort_values("ms_per_sample")
best[["what", "precision", "batch", "ms_per_sample", "test_rel_L2_mean"]].round(5).reset_index(drop=True)
'''),
    ("md", "How many queries the FNO needs before its training time is paid back, against each solver setting:"),
    ("code", 'load("stage8_breakeven", digits=6)'),
    ("code", 'show_fig("final/fig_speed_accuracy.png", width=560)'),
    ("md", r"""
- **At matched accuracy the FNO is not faster.** The FNO (0.0364) takes 3.5 ms per sample, while a $32^2$ solve
  that is already more accurate (0.0326) takes 0.30 ms. That is about 12× faster than the FNO.
- **Against the high-accuracy solve it is.** The FNO is 4.4× faster than a $128^2$ solve, but needs about 145,000
  queries to repay its training time.
"""),

    # ---------------------------------------------------------------- Stage 5b
    ("md", "## Stage 5b: Faster FNO Inference (Half Precision, Compilation)"),
    ("md", r"""
Can a faster FNO close the gap? The seed-0 FNO was re-trained with the Stage 1A protocol, and the same weights
were timed and evaluated in four inference modes in one session on a T4: float32, half precision
(`torch.autocast`, float16), `torch.compile`, and both. Notebook: `stage5b_fp16`.
"""),
    ("code", '''
fp = pd.read_csv(RES / "stage5b_fp16.csv").dropna(subset=["sec_per_sample"])
fb = fp.loc[fp.groupby("mode")["sec_per_sample"].idxmin()].copy()
fb["ms_per_sample"] = fb.sec_per_sample * 1e3
fb["speedup vs float32"] = fb.loc[fb["mode"] == "float32", "ms_per_sample"].iloc[0] / fb.ms_per_sample
fb.sort_values("ms_per_sample")[["mode", "batch", "ms_per_sample", "speedup vs float32", "test_rel_L2"]].round(5).reset_index(drop=True)
'''),
    ("md", r"""
- **Accuracy is unchanged in every mode** (test error 0.0378). The re-trained model matches the Stage 1A seed-0
  run (0.03776 vs 0.03774).
- **Speed-ups are modest.** Half precision alone is 1.11× faster than float32, compilation alone 1.15×, and both
  together 1.34× (2.55 ms per sample).
- **The conclusion of Stage 5 stands.** The $32^2$ solve (0.30 ms, error 0.033) is still 8.6× faster than the
  fastest FNO mode.
"""),

    # ---------------------------------------------------------------- Stage 7
    ("md", "## Stage 7: Active Learning"),
    ("md", r"""
Can the ensemble's disagreement choose which samples to label next? Both runs start from the same 1,000 random
training samples and add 500 per round up to 3,000. Each round retrains a $K=3$ FNO ensemble from scratch
(low-rank Adam $r=16$, seeds 0–2, 20 epochs). **Uncertainty** selection adds the 500 pool samples with the
largest member spread; **random** selection adds 500 at random. The 8,000 training samples already have labels,
so "running a simulation" means revealing a label. One run per strategy.
"""),
    ("code", '''
al = pd.concat([pd.read_csv(RES / f"stage7_{s}.csv") for s in ("uncertainty", "random")], ignore_index=True)
w = al.pivot(index="n_labelled", columns="strategy", values="ensemble_test_rel_L2")
w.assign(**{"uncertainty vs random (%)": 100 * (w.uncertainty / w.random - 1)}).round({"random": 4, "uncertainty": 4, "uncertainty vs random (%)": 1})
'''),
    ("md", "Per round: mean member error, how well spread ranks the test error, and the spread of the chosen samples vs the pool:"),
    ("code", '''
al.round(4)[["strategy", "n_labelled", "member_test_rel_L2_mean", "spearman_spread_error_test", "pool_spread_mean",
    "chosen_spread_mean", "round_minutes"]]
'''),
    ("code", 'show_fig("final/fig_active_learning.png", width=600)'),
    ("md", r"""
- **Uncertainty selection was worse than random at every round**, by 4.1–6.2% in ensemble test error. At 3,000
  samples it reached 0.0517, against 0.0497 for random.
- **The selection worked as designed.** The chosen samples had 1.3–1.7× the mean spread of the pool.
- **Spread stopped tracking the error.** After training on high-spread samples, the rank correlation between
  spread and test error fell from 0.56 to 0.19–0.26. With random selection it stayed at 0.64–0.65.
- **One run per strategy.** The sign of the gap is the same in all four rounds, but its size is uncertain.
"""),

    # ---------------------------------------------------------------- combined
    ("md", "## All Results in One Table"),
    ("md", "`results/FINAL.csv` collects every number above, one row per value, with its unit, seed count and source CSV."),
    ("code", '''
final = pd.read_csv(RES / "FINAL.csv")
with pd.option_context("display.max_rows", None):
    display(final)
'''),
]


def to_cells():
    out = []
    for i, (kind, text) in enumerate(CELLS):
        text = text.strip("\n")
        cid = f"res-{i:02d}"
        c = nbformat.v4.new_markdown_cell(text) if kind == "md" else nbformat.v4.new_code_cell(text)
        c["id"] = cid
        out.append(c)
    return out


def main():
    raw = json.loads(NB.read_text())
    cells = [c for c in raw["cells"] if not c.get("id", "").startswith("res-")]

    # execute the new cells on their own
    new = to_cells()
    tmp = nbformat.v4.new_notebook(cells=new)
    NotebookClient(tmp, timeout=600, kernel_name="fml-venv", resources={"metadata": {"path": str(ROOT)}}).execute()
    new = [json.loads(json.dumps(c)) for c in tmp.cells]

    # summary cell goes after the new part; the clean-up cell stays last
    summary = next(c for c in cells if c.get("id") == SUMMARY_ID)
    cells.remove(summary)
    s = "".join(summary["source"])
    s = s.replace("# Where the Project Went Next", "# Summary of Findings", 1)
    s = s.replace("The experiments that build on this notebook are in `notebooks/` (one per stage), with results in "
                  "`results/` and the combined table `results/FINAL.csv`. ", "The main results of Parts 1 and 2. ")
    summary["source"] = s.splitlines(keepends=True)
    cleanup = cells[-1]
    assert "".join(cleanup["source"]).startswith("import shutil"), "last cell should be the clean-up cell"
    raw["cells"] = cells[:-1] + new + [summary, cleanup]

    nbformat.validate(nbformat.reads(json.dumps(raw), as_version=4))
    NB.write_text(json.dumps(raw, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(new)} cells added; notebook now has {len(raw['cells'])} cells")


if __name__ == "__main__":
    main()
