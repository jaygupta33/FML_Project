"""Build the Stage 2 finalist notebooks (20 epochs x 3 seeds, predictions saved for Stage 3 ensembles).
Adam is not rerun: Stage 1A already has the same setting (FNO, lr 1e-3, 20 epochs, seeds 0-2, predictions)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_notebook import build, SETUP, DATA
from stage1_notebooks import kernel, ROOT

CFGS = {
    "GaLore r=16": dict(rank_mode="fixed", rank=16),
    "GaLore r=8": dict(rank_mode="fixed", rank=8),
    "GaLore r=4": dict(rank_mode="fixed", rank=4),
    "adaptive tau=0.999": dict(rank_mode="adaptive", tau=0.999, r_min=2, r_max=64),
    "adaptive tau=0.99": dict(rank_mode="adaptive", tau=0.99, r_min=2, r_max=64),
}
# 15 runs split into two kernels; each memory-matched pair stays within one kernel
SPLIT = {
    "a": [("adaptive tau=0.999", s) for s in (0, 1, 2)] + [("GaLore r=8", s) for s in (0, 1, 2)] + [("GaLore r=16", 0)],
    "b": [("adaptive tau=0.99", s) for s in (0, 1, 2)] + [("GaLore r=4", s) for s in (0, 1, 2)] + [("GaLore r=16", s) for s in (1, 2)],
}


def cells(part):
    runs = SPLIT[part]
    table = "\n".join(f"| {n} | {s} |" for n, s in runs)
    return [
        ("md", f"# Stage 2 finalists, part {part.upper()}"),
        ("md", f"""
Finalists from the Stage 2 screen, trained with the Stage 1A protocol (FNO, lr $10^{{-3}}$, MSE,
20 epochs, batch 32, basis refresh every 200 steps). Test predictions are saved for the Stage 3 ensembles.
Adam with the same protocol is Stage 1A and is not rerun.

| optimizer | seed |
|---|---|
{table}
"""),
        ("md", "## Setup"),
        ("code", SETUP + "from optim import LowRankAdam\n"),
        ("md", "## Data"),
        ("code", DATA),
        ("md", "## Runs"),
        ("code", f'''
EPOCHS, LR, REFRESH = 20, 1e-3, 200
CFGS = {CFGS!r}
RUNS = {runs!r}
pd.DataFrame([{{"optimizer": n, "seed": s, **CFGS[n]}} for n, s in RUNS])
'''),
        ("md", "## Training"),
        ("code", f'''
rows, hists, ranklogs = [], [], []
for name, seed in RUNS:
    cfg = CFGS[name]
    tr = Trainer(nsbench.FourierNeuralOperator, lambda p, cfg=cfg: LowRankAdam(p, lr=LR, refresh=REFRESH, **cfg),
                 D.train_loader, D.val_loader, D.test_loader, D.y_mean, D.y_std, device,
                 seed=seed, label=f"FNO {{name}} seed {{seed}}").fit(EPOCHS)
    rows.append(run_record(tr.summary(), run="FNO", optimizer=name, lr=LR, seed=seed, epochs=EPOCHS, batch=32,
                           loss_fn="MSE", monitor="loss", refresh=REFRESH, **cfg))
    hists.append(tr.history().assign(optimizer=name, seed=seed))
    ranklogs.append(pd.DataFrame(tr.opt.rank_log, columns=["step", "param", "rank", "energy"]).assign(optimizer=name, seed=seed))
    save_predictions(tr, D.x_test, f"fno_{{name}}_seed{{seed}}")
    # save after every run so a timeout keeps finished runs
    save_csv(pd.concat(rows, ignore_index=True), "stage2_final_{part}")
    save_csv(pd.concat(hists, ignore_index=True), "stage2_final_{part}_history")
    save_csv(pd.concat(ranklogs, ignore_index=True), "stage2_final_{part}_ranks")
    del tr; torch.cuda.empty_cache()
final = pd.concat(rows, ignore_index=True)
final[["optimizer", "seed", "best_epoch", "test_rel_L2", "opt_state_MB", "peak_gpu_MB", "sec_per_epoch"]]
'''),
        ("md", "## Mean ± std per optimizer"),
        ("code", '''
final.groupby("optimizer")[["test_rel_L2", "test_mse_norm", "opt_state_MB", "peak_gpu_MB", "sec_per_epoch"]].agg(["mean", "std", "count"])
'''),
    ]


if __name__ == "__main__":
    for part in SPLIT:
        nb = ROOT / f"notebooks/stage2_final_{part}.ipynb"
        build(cells(part), nb)
        kernel(f"stage2_final_{part}", f"ns-stage2-final-{part}", f"NS Stage2 final {part}", nb)
