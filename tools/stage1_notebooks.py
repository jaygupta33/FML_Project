"""Build the two Stage 1 notebooks and their Kaggle kernel folders (kaggle/stage1_fno, kaggle/stage1_unet)."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_notebook import build, SETUP, DATA

ROOT = Path(__file__).resolve().parents[1]


def kernel(folder, slug, title, nb):
    d = ROOT / "kaggle" / folder
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(nb, d / Path(nb).name)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"jaygupta33/{slug}", "title": title, "code_file": Path(nb).name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["jaygupta33/ns-surrogate-code"],
        "kernel_sources": ["jaygupta33/ns-data"], "competition_sources": []}, indent=2))


# ----------------------------------------------------------------------------- A: FNO + Adam, 3 seeds
FNO = [
    ("md", "# Stage 1A: FNO + Adam, three seeds"),
    ("md", """
Reference baseline for every later comparison. Same split, data, epochs and learning rate as the main
notebook; only the seed changes.

| setting | value |
|---|---|
| model | FNO (neuralop), 2,410,177 params |
| optimizer | Adam, lr $10^{-3}$ |
| loss | MSE (standardized units) |
| epochs / batch | 20 / 32 |
| seeds | 0, 1, 2 |
"""),
    ("md", "## Setup"),
    ("code", SETUP),
    ("md", "## Data"),
    ("code", DATA),
    ("md", "## Training"),
    ("code", '''
SEEDS, EPOCHS, LR = [0, 1, 2], 20, 1e-3
rows, hists = [], []
for s in SEEDS:
    tr = Trainer(nsbench.FourierNeuralOperator, lambda p: torch.optim.Adam(p, lr=LR),
                 D.train_loader, D.val_loader, D.test_loader, D.y_mean, D.y_std, device,
                 seed=s, label=f"FNO Adam seed {s}").fit(EPOCHS)
    rows.append(run_record(tr.summary(), run="FNO", optimizer="Adam", lr=LR, seed=s,
                           epochs=EPOCHS, batch=32, loss_fn="MSE", monitor="loss"))
    hists.append(tr.history().assign(seed=s))
    save_predictions(tr, D.x_test, f"fno_adam_seed{s}")
runs = pd.concat(rows, ignore_index=True)
save_csv(runs, "stage1_fno_adam_seeds")
save_csv(pd.concat(hists, ignore_index=True), "stage1_fno_adam_history")
runs[["label", "best_epoch", "test_rel_L2", "test_mse_norm", "opt_state_MB", "peak_gpu_MB", "sec_per_epoch"]]
'''),
    ("md", "## Mean ± std over seeds"),
    ("code", '''
cols = ["test_rel_L2", "test_mse_norm", "best_val_loss", "opt_state_MB", "peak_gpu_MB", "sec_per_epoch"]
agg = runs[cols].agg(["mean", "std"]).T
agg["n_seeds"] = len(runs)
save_csv(agg, "stage1_fno_adam_mean_std", index=True)
agg
'''),
    ("md", "## Validation curves"),
    ("code", '''
H = pd.concat(hists, ignore_index=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for s, h in H.groupby("seed"):
    ax[0].plot(h.epoch, h.val_mse_norm, label=f"seed {s}")
    ax[1].plot(h.epoch, h.val_rel_L2, label=f"seed {s}")
ax[0].set(xlabel="epoch", ylabel="validation MSE (standardized)", yscale="log")
ax[1].set(xlabel="epoch", ylabel="validation relative $L^2$", yscale="log")
ax[0].legend(); fig.suptitle("FNO + Adam, three seeds"); fig.tight_layout()
save_fig("stage1_fno_adam_curves", fig)
'''),
]

# ----------------------------------------------------------------------------- B: U-Net control
UNET = [
    ("md", "# Stage 1B: U-Net control"),
    ("md", """
Fair control for the spectral-loss comparison: the U-Net + MSE learning rate is tuned first, and
the H1-type spectral loss ($\\lambda=0.5$, $|k|^2$ weights) is then trained at the same learning rate.
The learning rate is picked on validation loss only; the test set is not used for any choice.

| setting | value |
|---|---|
| model | U-Net (base 32), 2,140,065 params |
| lr sweep (MSE) | $10^{-3}$, $3\\cdot10^{-4}$, $10^{-4}$ |
| selection | lowest best validation MSE |
| spectral run | SpectralLoss(0.5, "k2") at the selected lr, best epoch by validation MSE |
| epochs / batch / seed | 20 / 32 / 0 |

The ring-weighted variant is not rerun (it had a $k=0$ ring bug in v1).
"""),
    ("md", "## Setup"),
    ("code", SETUP),
    ("md", "## Data"),
    ("code", DATA),
    ("md", "## Learning-rate sweep (MSE)"),
    ("code", '''
LRS, EPOCHS, SEED = [1e-3, 3e-4, 1e-4], 20, 0
rows, hists, trainers = [], [], {}
for lr in LRS:
    tr = Trainer(nsbench.UNet, lambda p, lr=lr: torch.optim.Adam(p, lr=lr),
                 D.train_loader, D.val_loader, D.test_loader, D.y_mean, D.y_std, device,
                 monitor="loss", seed=SEED, label=f"U-Net MSE lr {lr:g}").fit(EPOCHS)
    rows.append(run_record(tr.summary(), run="U-Net", optimizer="Adam", lr=lr, seed=SEED,
                           epochs=EPOCHS, batch=32, loss_fn="MSE", monitor="loss"))
    hists.append(tr.history().assign(run=f"MSE lr {lr:g}"))
    trainers[lr] = tr
sweep = pd.concat(rows, ignore_index=True)
save_csv(sweep, "stage1_unet_lr_sweep")
sweep[["label", "best_epoch", "best_val_loss", "test_rel_L2", "sec_per_epoch"]]
'''),
    ("md", "## Selected learning rate"),
    ("code", '''
best_lr = float(sweep.loc[sweep.best_val_loss.idxmin(), "lr"])
pd.DataFrame([{"selected lr": best_lr, "criterion": "lowest best_val_loss (validation MSE)"}])
'''),
    ("md", "## Spectral loss at the selected learning rate"),
    ("code", '''
tr_k2 = Trainer(nsbench.UNet, lambda p: torch.optim.Adam(p, lr=best_lr),
                D.train_loader, D.val_loader, D.test_loader, D.y_mean, D.y_std, device,
                loss_fn=nsbench.SpectralLoss(0.5, "k2"), monitor="mse_norm", seed=SEED,
                label=f"U-Net spectral k2 lr {best_lr:g}").fit(EPOCHS)
k2 = run_record(tr_k2.summary(), run="U-Net", optimizer="Adam", lr=best_lr, seed=SEED, epochs=EPOCHS,
                batch=32, loss_fn="SpectralLoss(0.5,k2)", monitor="mse_norm")
hists.append(tr_k2.history().assign(run=f"spectral k2 lr {best_lr:g}"))
control = pd.concat([sweep[sweep.lr == best_lr], k2], ignore_index=True)
save_csv(control, "stage1_unet_control")
save_csv(pd.concat(hists, ignore_index=True), "stage1_unet_history")
control[["label", "best_epoch", "test_rel_L2", "test_mse_norm", "sec_per_epoch"]]
'''),
    ("md", "## Error by frequency band (test set, raw units)"),
    ("code", '''
from nsbench.analysis import band_errors
raw = lambda t: t * D.y_std + D.y_mean
y_raw = raw(D.y_test.squeeze(1))
preds = {"U-Net MSE": raw(save_predictions(trainers[best_lr], D.x_test, f"unet_mse_lr{best_lr:g}")),
         "U-Net spectral k2": raw(save_predictions(tr_k2, D.x_test, f"unet_k2_lr{best_lr:g}"))}
bands = band_errors(preds, y_raw)
save_csv(bands, "stage1_unet_band_errors")
bands
'''),
    ("md", "## Validation curves"),
    ("code", '''
H = pd.concat(hists, ignore_index=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, h in H.groupby("run", sort=False):
    ax[0].plot(h.epoch, h.val_mse_norm, label=name)
    ax[1].plot(h.epoch, h.val_rel_L2, label=name)
ax[0].set(xlabel="epoch", ylabel="validation MSE (standardized)", yscale="log")
ax[1].set(xlabel="epoch", ylabel="validation relative $L^2$", yscale="log")
ax[0].legend(); fig.suptitle("U-Net: lr sweep and spectral loss"); fig.tight_layout()
save_fig("stage1_unet_curves", fig)
'''),
]

if __name__ == "__main__":
    a, b = ROOT / "notebooks/stage1_fno_seeds.ipynb", ROOT / "notebooks/stage1_unet_control.ipynb"
    build(FNO, a); build(UNET, b)
    kernel("stage1_fno", "ns-stage1-fno-seeds", "NS Stage1 FNO seeds", a)
    kernel("stage1_unet", "ns-stage1-unet-control", "NS Stage1 UNet control", b)
