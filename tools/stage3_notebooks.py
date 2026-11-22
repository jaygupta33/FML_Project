"""Build the Stage 3 ensemble notebook (runs locally on CPU from saved test predictions)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_notebook import build, SETUP
from stage1_notebooks import ROOT

ENS = [
    ("md", "# Stage 3: K = 3 ensembles and uncertainty"),
    ("md", """
Each ensemble averages the test predictions of three FNOs trained with the same optimizer and
different seeds (0, 1, 2). No new training: predictions come from Stage 1A (Adam) and the Stage 2
finalists. Errors are in raw units against the dataset's test set.

| quantity | definition |
|---|---|
| ensemble prediction | $\\bar y = \\frac1K \\sum_k \\hat y_k$ |
| member spread | $s(x) = \\sqrt{\\frac1{K-1}\\sum_k (\\hat y_k - \\bar y)^2}$ per pixel |
| relative $L^2$ | $\\|\\bar y - y\\| / \\|y\\|$ per sample, then $\\sum$ errors $/\\sum$ norms as in the Trainer |
| resident state for training $K$ members at once | $K \\times$ (parameters + optimizer state) |
"""),
    ("md", "## Setup"),
    ("code", SETUP + '''from nsbench.data import find_data_dir
import glob
from pathlib import Path
from scipy.stats import spearmanr
'''),
    ("md", "## Test targets in raw units"),
    ("code", '''
te = torch.load(f"{find_data_dir()}/nsforcing_test_128.pt")
y = te["y"].double()
tr = torch.load(f"{find_data_dir()}/nsforcing_train_128.pt")["y"]
perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))
y_tr = tr[perm][: int(0.8 * len(tr))].float()
y_mean, y_std = y_tr.mean().double(), y_tr.std().double()        # same train statistics as nsbench.load
del tr, y_tr
pd.DataFrame([{"test samples": len(y), "y_mean (train)": y_mean.item(), "y_std (train)": y_std.item()}])
'''),
    ("md", "## Available predictions"),
    ("code", '''
PRED_DIRS = sorted(glob.glob("kaggle_out/*/preds"))
files = {Path(p).stem: p for d in PRED_DIRS for p in glob.glob(f"{d}/*.npy")}
GROUPS = {"Adam": "fno_adam_seed{}", "GaLore r=16": "fno_GaLore_r_16_seed{}", "GaLore r=8": "fno_GaLore_r_8_seed{}",
          "GaLore r=4": "fno_GaLore_r_4_seed{}", "adaptive tau=0.999": "fno_adaptive_tau_0.999_seed{}",
          "adaptive tau=0.99": "fno_adaptive_tau_0.99_seed{}"}
avail = {g: [files.get(p.format(s)) for s in (0, 1, 2)] for g, p in GROUPS.items()}
avail = {g: f for g, f in avail.items() if all(f)}
pd.DataFrame([{"optimizer": g, "members": len(f)} for g, f in avail.items()])
'''),
    ("md", "## Ensemble accuracy and spread"),
]

# the real ensemble cell (kept separate for readability)
ENS.append(("code", '''
def rel(p, t):
    return ((p - t).flatten(1).norm(dim=1).sum() / t.flatten(1).norm(dim=1).sum()).item()

rows, per_sample, ens_preds = [], [], {}
for g, fs in avail.items():
    P = torch.stack([torch.from_numpy(np.load(f).astype(np.float64)) for f in fs]) * y_std + y_mean   # [K, N, n, n]
    mean, spread = P.mean(0), P.std(0, unbiased=True)
    err = (mean - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)
    sprd = spread.flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)
    rho = spearmanr(sprd.numpy(), err.numpy()).correlation
    members = [rel(P[k], y) for k in range(len(fs))]
    rows.append({"optimizer": g, "member rel L2 mean": np.mean(members), "member rel L2 std": np.std(members, ddof=1),
                 "ensemble rel L2": rel(mean, y), "gain vs mean member": 1 - rel(mean, y) / np.mean(members),
                 "spread / error (mean)": (sprd / err).mean().item(), "Spearman(spread, error)": rho})
    per_sample.append(pd.DataFrame({"optimizer": g, "sample": range(len(y)), "rel_L2": err.numpy(), "rel_spread": sprd.numpy()}))
    ens_preds[g] = mean
ens = pd.DataFrame(rows)
save_csv(ens, "stage3_ensembles")
save_csv(pd.concat(per_sample, ignore_index=True), "stage3_per_sample")
ens
'''))

ENS += [
    ("md", "## Error by frequency band (ensembles)"),
    ("code", '''
from nsbench.analysis import band_errors
bands = band_errors({g: p.float() for g, p in ens_preds.items()}, y.float())
save_csv(bands, "stage3_band_errors")
bands
'''),
    ("md", "## Does the spread flag the hard samples?"),
    ("code", '''
S = pd.concat(per_sample, ignore_index=True)
fig, ax = plt.subplots(figsize=(6, 4.5))
for g, s in S.groupby("optimizer", sort=False):
    ax.scatter(s.rel_spread, s.rel_L2, s=4, alpha=0.4, label=g)
lim = [S[["rel_spread", "rel_L2"]].min().min(), S[["rel_spread", "rel_L2"]].max().max()]
ax.plot(lim, lim, "k--", lw=0.8, label="spread = error")
ax.set(xscale="log", yscale="log", xlabel="ensemble spread (relative)", ylabel="ensemble error (relative $L^2$)")
ax.legend(fontsize=7, markerscale=3); fig.tight_layout()
save_fig("stage3_spread_vs_error", fig)
'''),
]

if __name__ == "__main__":
    build(ENS, ROOT / "notebooks/stage3_ensembles.ipynb")
