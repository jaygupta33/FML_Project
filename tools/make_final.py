"""Assemble results/FINAL.csv and the report figures (figures/final/) from the committed stage CSVs.
Every number here is read from results/*.csv; nothing is typed in by hand."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures/final"
F.mkdir(parents=True, exist_ok=True)

# palette: validated categorical slots (light surface); identity also carried by marker shape + labels
C = {"Adam": "#2a78d6", "GaLore": "#eb6834", "adaptive": "#1baf7a", "U-Net": "#e87ba4", "solver": "#6250d6"}
MK = {"Adam": "o", "GaLore": "s", "adaptive": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "legend.frameon": False, "savefig.dpi": 200, "figure.facecolor": "white"})


def family(name):
    return "Adam" if name == "Adam" else "GaLore" if name.startswith("GaLore") else "adaptive"


def short(name):
    return name.replace("adaptive tau=", "adaptive τ=").replace("GaLore r=", "GaLore r=")


PER_R = (2 * 9216 * 8 + 64 * 8) / 1e6      # MB per unit rank per spectral weight: complex64 moments + P
UNCOMP = 0.407                              # MB of ordinary Adam state for the small tensors (Stage 2 memory split)

# ------------------------------------------------------------------ optimizer runs (20 epochs x 3 seeds)
adam = pd.read_csv(R / "stage1_fno_adam_seeds.csv").assign(optimizer="Adam")
fin = pd.concat([pd.read_csv(R / f"stage2_final_{p}.csv") for p in "ab"])
runs = pd.concat([adam, fin], ignore_index=True)
ranks = pd.concat([pd.read_csv(R / f"stage2_final_{p}_ranks.csv") for p in "ab"])
hist = pd.concat([pd.read_csv(R / "stage1_fno_adam_history.csv").assign(optimizer="Adam")] +
                 [pd.read_csv(R / f"stage2_final_{p}_history.csv") for p in "ab"])

peak_state = (ranks.groupby(["optimizer", "seed", "step"])["rank"].sum() * PER_R + UNCOMP).groupby("optimizer").max()
mean_state = (ranks.groupby(["optimizer", "seed", "step"])["rank"].sum() * PER_R + UNCOMP).groupby("optimizer").mean()
opt = runs.groupby("optimizer").agg(seeds=("seed", "count"), test_rel_L2_mean=("test_rel_L2", "mean"),
                                    test_rel_L2_std=("test_rel_L2", "std"), opt_state_MB_end=("opt_state_MB", "mean"),
                                    peak_gpu_MB=("peak_gpu_MB", "mean"), sec_per_epoch=("sec_per_epoch", "mean"),
                                    param_MB=("param_MB", "mean"))
opt["opt_state_MB_peak"] = peak_state.reindex(opt.index).fillna(opt["opt_state_MB_end"])
opt["opt_state_MB_mean"] = mean_state.reindex(opt.index).fillna(opt["opt_state_MB_end"])
ens = pd.read_csv(R / "stage3_ensembles.csv").set_index("optimizer")
opt = opt.join(ens[["ensemble rel L2", "Spearman(spread, error)", "spread / error (mean)"]])
opt["K3_opt_state_MB_peak"] = 3 * opt["opt_state_MB_peak"]
opt["K3_resident_MB_peak"] = 3 * (opt["opt_state_MB_peak"] + opt["param_MB"])
order = ["Adam", "GaLore r=16", "adaptive tau=0.999", "GaLore r=8", "adaptive tau=0.99", "GaLore r=4"]
opt = opt.loc[order]
opt.to_csv(R / "final_optimizers.csv")

# ------------------------------------------------------------------ FINAL.csv (long format, self-describing)
rows = []
def add(stage, item, metric, value, unit="", n=1, source=""):
    rows.append(dict(stage=stage, item=item, metric=metric, value=value, unit=unit, n_seeds=n, source=source))

for name, r in opt.iterrows():
    src = "stage1_fno_adam_seeds.csv" if name == "Adam" else "stage2_final_{a,b}.csv"
    add("2 optimizers", f"FNO {name}", "test rel L2 (mean)", r.test_rel_L2_mean, "", r.seeds, src)
    add("2 optimizers", f"FNO {name}", "test rel L2 (std)", r.test_rel_L2_std, "", r.seeds, src)
    add("2 optimizers", f"FNO {name}", "optimizer state, end of training", r.opt_state_MB_end, "MB", r.seeds, src)
    add("2 optimizers", f"FNO {name}", "optimizer state, peak during training", r.opt_state_MB_peak, "MB", r.seeds, "rank logs")
    add("2 optimizers", f"FNO {name}", "peak GPU memory", r.peak_gpu_MB, "MB", r.seeds, src)
    add("2 optimizers", f"FNO {name}", "training time per epoch", r.sec_per_epoch, "s", r.seeds, src)
    add("3 ensembles", f"FNO {name} K=3", "test rel L2", r["ensemble rel L2"], "", 3, "stage3_ensembles.csv")
    add("3 ensembles", f"FNO {name} K=3", "Spearman(spread, error)", r["Spearman(spread, error)"], "", 3, "stage3_ensembles.csv")
    add("3 ensembles", f"FNO {name} K=3", "optimizer state, peak, all members", r.K3_opt_state_MB_peak, "MB", 3, "derived")

ctl = pd.read_csv(R / "stage1_unet_control.csv")
for _, r in ctl.iterrows():
    add("1 baselines", r.label, "test rel L2", r.test_rel_L2, "", 1, "stage1_unet_control.csv")
bands = pd.read_csv(R / "stage1_unet_band_errors.csv")
for _, r in bands.iterrows():
    for b in bands.columns[1:]:
        add("1 baselines", r.model, f"relative error energy, {b}", r[b], "", 1, "stage1_unet_band_errors.csv")

agree = pd.read_csv(R / "stage4_test_agreement.csv").iloc[0]
add("4 solver", "own solver 128^2 CN-Heun dt 1e-3", "rel L2 vs dataset, 50 test samples (mean)", agree["rel L2 mean"], "", 1, "stage4_test_agreement.csv")
add("4 solver", "own solver 128^2 CN-Heun dt 1e-3", "rel L2 vs dataset, 50 test samples (max)", agree["rel L2 max"], "", 1, "stage4_test_agreement.csv")

tim = pd.read_csv(R / "stage5_timing.csv")
best_idx = tim.groupby(["what", "precision"])["sec_per_sample"].idxmin()
t256 = tim.loc[best_idx].reset_index(drop=True)          # best throughput over batch sizes 1/32/256
for _, r in t256.iterrows():
    add("5 timing", f"{r.what} ({r.precision})", "time per sample, best batch size, Tesla T4", r.sec_per_sample * 1e3, "ms", 1, "stage5_timing.csv")
    if not np.isnan(r.test_rel_L2_mean):
        add("5 timing", f"{r.what} ({r.precision})", "test rel L2 vs dataset (64 samples)", r.test_rel_L2_mean, "", 1, "stage5_timing.csv")

# Stage 7 active learning: included once both strategies have results
AL_FILES = [R / f"stage7_{s}.csv" for s in ("uncertainty", "random")]
al = pd.concat([pd.read_csv(p) for p in AL_FILES], ignore_index=True) if all(p.exists() for p in AL_FILES) else None
if al is not None:
    for _, r in al.iterrows():
        src = f"stage7_{r.strategy}.csv"
        add("7 active learning", f"{r.strategy} selection, n={r.n_labelled}", "ensemble test rel L2 (K=3, GaLore r=16)",
            r.ensemble_test_rel_L2, "", 3, src)
        add("7 active learning", f"{r.strategy} selection, n={r.n_labelled}", "mean member test rel L2",
            r.member_test_rel_L2_mean, "", 3, src)

FINAL = pd.DataFrame(rows)
FINAL.to_csv(R / "FINAL.csv", index=False)
print(f"FINAL.csv: {len(FINAL)} rows")

# ------------------------------------------------------------------ Fig 1: accuracy vs optimizer state
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for name, r in opt.iterrows():
    fam = family(name)
    if r.opt_state_MB_peak > r.opt_state_MB_end * 1.01:          # adaptive: end-of-run to peak range
        ax.plot([r.opt_state_MB_end, r.opt_state_MB_peak], [r.test_rel_L2_mean] * 2, color=C[fam], lw=1.2, alpha=0.6)
        ax.plot(r.opt_state_MB_peak, r.test_rel_L2_mean, marker="|", ms=10, color=C[fam])
    ax.errorbar(r.opt_state_MB_end, r.test_rel_L2_mean, yerr=r.test_rel_L2_std, marker=MK[fam], ms=7, color=C[fam],
                mec="white", mew=1, capsize=2, lw=1.2, ls="none")
    off = (-4, -13) if fam == "adaptive" else (7, 3)
    off = (7, -12) if name == "GaLore r=16" else off
    ax.annotate(short(name), (r.opt_state_MB_end, r.test_rel_L2_mean), xytext=off, textcoords="offset points",
                fontsize=8, color=INK2, ha="right" if fam == "adaptive" else "left")
ax.set_xscale("log")
ax.set_xticks([2, 3, 5, 10, 20, 40]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.minorticks_off()
ax.set(xlabel="optimizer state (MB, log scale)", ylabel="test relative $L^2$ (mean ± std, 3 seeds)")
ax.text(0.02, 0.04, "adaptive: marker = end of training, | = peak during training", transform=ax.transAxes,
        fontsize=7.5, color=INK2)
fig.tight_layout(); fig.savefig(F / "fig_accuracy_vs_state.pdf"); fig.savefig(F / "fig_accuracy_vs_state.png"); plt.close(fig)

# ------------------------------------------------------------------ Fig 2: adaptive rank over training
fig, ax = plt.subplots(figsize=(6.2, 3.2))
for name, ls in (("adaptive tau=0.999", "-"), ("adaptive tau=0.99", "--")):
    rr = ranks[ranks.optimizer == name].groupby(["seed", "step"])["rank"].mean().reset_index()
    m = rr.groupby("step")["rank"].agg(["mean", "min", "max"])
    ax.fill_between(m.index / 250, m["min"], m["max"], color=C["adaptive"], alpha=0.15, lw=0)
    ax.plot(m.index / 250, m["mean"], ls=ls, color=C["adaptive"], label=short(name))
for r_, y in ((16, 16), (8, 8), (4, 4)):
    ax.axhline(y, color=C["GaLore"], lw=0.8, ls=":")
    ax.text(20.6, y, f"GaLore r={r_}", va="center", fontsize=7.5, color=INK2)
ax.set(xlabel="epoch", ylabel="rank (mean of 4 spectral weights)", xlim=(0, 23.5))
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(F / "fig_adaptive_rank.pdf"); fig.savefig(F / "fig_adaptive_rank.png"); plt.close(fig)

# ------------------------------------------------------------------ Fig 3: validation curves
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for name in order:
    h = hist[hist.optimizer == name].groupby("epoch")["val_rel_L2"].mean()
    fam = family(name)
    ax.plot(h.index, h.values, color=C[fam], marker=MK[fam], ms=3.5, lw=1.3,
            ls="-" if name in ("Adam", "GaLore r=16", "adaptive tau=0.999") else "--", label=short(name))
ax.set(xlabel="epoch", ylabel="validation relative $L^2$ (mean of 3 seeds)", yscale="log")
ax.legend(fontsize=7.5, ncol=2)
fig.tight_layout(); fig.savefig(F / "fig_val_curves.pdf"); fig.savefig(F / "fig_val_curves.png"); plt.close(fig)

# ------------------------------------------------------------------ Fig 4: solver frontier vs surrogates
sol = t256[t256.what.str.startswith("solver") & (t256.precision == "float32")].sort_values("sec_per_sample")
fno_t = t256[t256.what == "FNO"].sec_per_sample.iloc[0]
unet_t = t256[t256.what == "U-Net"].sec_per_sample.iloc[0]
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(sol.sec_per_sample * 1e3, sol.test_rel_L2_mean, "o-", color=C["solver"], ms=6, mec="white", label="our solver (grid labelled)")
for _, r in sol.iterrows():
    ax.annotate(f"{int(r.grid)}²", (r.sec_per_sample * 1e3, r.test_rel_L2_mean), xytext=(5, 3), textcoords="offset points",
                fontsize=7.5, color=INK2)
ax.plot(fno_t * 1e3, opt.loc["Adam", "test_rel_L2_mean"], marker="o", ms=8, color=C["Adam"], ls="none", mec="white",
        label="FNO + Adam (single model)")
ax.plot(3 * fno_t * 1e3, opt.loc["Adam", "ensemble rel L2"], marker="D", ms=7, color=C["Adam"], ls="none", mec="white",
        label="FNO + Adam, K=3 ensemble")
u = ctl[ctl.label.str.contains("spectral")].test_rel_L2.iloc[0]
ax.plot(unet_t * 1e3, u, marker="s", ms=7, color=C["U-Net"], ls="none", mec="white", label="U-Net (spectral loss)")
ax.set(xscale="log", yscale="log", xlabel="time per sample (ms, best batch size, Tesla T4)", ylabel="test relative $L^2$ vs dataset")
ax.legend(fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(F / "fig_speed_accuracy.pdf"); fig.savefig(F / "fig_speed_accuracy.png"); plt.close(fig)

# ------------------------------------------------------------------ Fig 5: U-Net band errors
b = bands.set_index("model")
fig, ax = plt.subplots(figsize=(5.2, 3.0))
x = np.arange(len(b.columns))
for i, (m, col) in enumerate(((b.index[0], C["U-Net"]), (b.index[1], C["solver"]))):
    ax.bar(x + (i - 0.5) * 0.38, b.loc[m].values, width=0.36, color=col, label=m, edgecolor="white", linewidth=2)
    for xi, v in zip(x, b.loc[m].values):
        ax.text(xi + (i - 0.5) * 0.38, v * 1.15, f"{v:.3g}", ha="center", fontsize=7, color=INK2)
ax.set_xticks(x, [c.replace(" (", "\n(") for c in b.columns]); ax.set_yscale("log")
ax.set(ylabel="error energy / true energy in band"); ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(F / "fig_unet_bands.pdf"); fig.savefig(F / "fig_unet_bands.png"); plt.close(fig)

# ------------------------------------------------------------------ Fig 6: ensemble spread vs error (Adam and best low-rank)
S = pd.read_csv(R / "stage3_per_sample.csv")
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), sharex=True, sharey=True)
for ax, name in zip(axs, ("Adam", "GaLore r=16")):
    s = S[S.optimizer == name]
    fam = family(name)
    ax.scatter(s.rel_spread, s.rel_L2, s=4, alpha=0.35, color=C[fam], lw=0)
    lo, hi = S[["rel_spread", "rel_L2"]].min().min(), S[["rel_spread", "rel_L2"]].max().max()
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
    ax.set(xscale="log", yscale="log", title=f"{short(name)}  (Spearman {ens.loc[name, 'Spearman(spread, error)']:.2f})",
           xlabel="ensemble spread (relative)")
    ax.title.set_fontsize(8.5)
axs[0].set_ylabel("ensemble error (relative $L^2$)")
fig.tight_layout(); fig.savefig(F / "fig_spread_error.pdf"); fig.savefig(F / "fig_spread_error.png"); plt.close(fig)
# ------------------------------------------------------------------ Fig 7: active learning, test error vs labelled samples
AL_STYLE = {"uncertainty": (C["solver"], "o", "uncertainty selection"), "random": (INK2, "s", "random selection (control)")}
if al is not None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for s, g in al.groupby("strategy"):
        col, mk, lab = AL_STYLE[s]
        g = g.sort_values("n_labelled")
        ax.plot(g.n_labelled, g.ensemble_test_rel_L2, marker=mk, color=col, ms=7, mec="white", label=f"{lab}, ensemble")
        ax.plot(g.n_labelled, g.member_test_rel_L2_mean, marker=mk, color=col, ms=5, mec="white", ls="--", lw=1.2,
                alpha=0.7, label=f"{lab}, mean member")
    ax.set(xlabel="labelled training samples", ylabel="test relative $L^2$", yscale="log")
    ax.set_xticks(sorted(al.n_labelled.unique()))
    ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(F / "fig_active_learning.pdf"); fig.savefig(F / "fig_active_learning.png"); plt.close(fig)
print("figures written to", F)

# ------------------------------------------------------------------ Stage 8: break-even
EPOCHS, N_TRAIN = 20, 8000
fno_ms = t256[t256.what == "FNO"].sec_per_sample.iloc[0] * 1e3
train_s = opt.loc["Adam", "sec_per_epoch"] * EPOCHS
gen = t256[(t256.what == "solver 128^2") & (t256.precision == "float32")].iloc[0]
gen_s = gen.sec_per_sample * N_TRAIN
be = []
for _, r in t256[t256.what.str.startswith("solver") & (t256.precision == "float32")].sort_values("grid").iterrows():
    gain_ms = r.sec_per_sample * 1e3 - fno_ms
    be.append({"solver": f"{int(r.grid)}^2, dt {r['dt']:g}", "solver ms/sample": r.sec_per_sample * 1e3,
               "solver rel L2": r.test_rel_L2_mean, "FNO ms/sample": fno_ms,
               "speedup (solver/FNO)": r.sec_per_sample * 1e3 / fno_ms,
               "break-even queries": (train_s + gen_s) / (gain_ms / 1e3) if gain_ms > 0 else np.inf})
be = pd.DataFrame(be)
be.to_csv(R / "stage8_breakeven.csv", index=False)

# ------------------------------------------------------------------ LaTeX tables and macros for the report
G = ROOT / "report/generated"; G.mkdir(parents=True, exist_ok=True)
def tex(name):
    return name.replace("adaptive tau=", r"adaptive $\tau{=}$").replace("GaLore r=", r"GaLore $r{=}$")

lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"optimizer & test rel.\ $L^2$ & \multicolumn{2}{c}{optimizer state (MB)} & peak GPU & s/epoch & $K{=}3$ ensemble \\",
         r" & mean $\pm$ std & end & peak & (MB) & & test rel.\ $L^2$ \\", r"\midrule"]
for name, r in opt.iterrows():
    lines.append(f"{tex(name)} & {r.test_rel_L2_mean:.4f} $\\pm$ {r.test_rel_L2_std:.4f} & {r.opt_state_MB_end:.1f} & "
                 f"{r.opt_state_MB_peak:.1f} & {r.peak_gpu_MB:,.0f} & {r.sec_per_epoch:.1f} & {r['ensemble rel L2']:.4f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(G / "tab_optimizers.tex").write_text("\n".join(lines))

lines = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"optimizer ($K{=}3$) & mean member & ensemble & spread/error & Spearman \\", r"\midrule"]
for name, r in ens.loc[order].iterrows():
    lines.append(f"{tex(name)} & {r['member rel L2 mean']:.4f} & {r['ensemble rel L2']:.4f} & {r['spread / error (mean)']:.2f} & "
                 f"{r['Spearman(spread, error)']:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(G / "tab_ensembles.tex").write_text("\n".join(lines))

lines = [r"\begin{tabular}{lccc}", r"\toprule",
         r"method & ms / sample & test rel.\ $L^2$ & FNO speedup \\", r"\midrule"]
for _, r in be.iterrows():
    lines.append(f"solver ${r.solver.split(',')[0]}$, $\\Delta t={r.solver.split('dt ')[1]}$ & {r['solver ms/sample']:.2f} & "
                 f"{r['solver rel L2']:.1e} & {r['speedup (solver/FNO)']:.2f}$\\times$ \\\\")
lines.append(f"FNO (single, Adam) & {fno_ms:.2f} & {opt.loc['Adam', 'test_rel_L2_mean']:.4f} & -- \\\\")
lines.append(f"FNO ($K{{=}}3$ ensemble) & {3 * fno_ms:.2f} & {opt.loc['Adam', 'ensemble rel L2']:.4f} & -- \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(G / "tab_speed.tex").write_text("\n".join(lines))

u = ctl.set_index("label")
mse_lbl = [l for l in u.index if "MSE" in l][0]; k2_lbl = [l for l in u.index if "spectral" in l][0]
sw = pd.read_csv(R / "stage1_unet_lr_sweep.csv")
lines = [r"\begin{tabular}{lccc}", r"\toprule", r"run & selection & best epoch & test rel.\ $L^2$ \\", r"\midrule"]
for _, r in sw.iterrows():
    lines.append(f"U-Net MSE, lr ${r.lr:g}$ & val.\\ MSE {r.best_val_loss:.4f} & {int(r.best_epoch)} & {r.test_rel_L2:.4f} \\\\")
lines.append(f"U-Net spectral ($\\lambda{{=}}0.5$, $|k|^2$), lr ${u.loc[k2_lbl, 'lr']:g}$ & -- & {int(u.loc[k2_lbl, 'best_epoch'])} & {u.loc[k2_lbl, 'test_rel_L2']:.4f} \\\\")
lines.append(f"FNO + Adam (3 seeds) & -- & -- & {opt.loc['Adam', 'test_rel_L2_mean']:.4f} $\\pm$ {opt.loc['Adam', 'test_rel_L2_std']:.4f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(G / "tab_baselines.tex").write_text("\n".join(lines))

if al is not None:
    w = al.pivot(index="n_labelled", columns="strategy", values="ensemble_test_rel_L2")
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"labelled samples & random & uncertainty & change \\", r"\midrule"]
    for n, r in w.iterrows():
        lines.append(f"{n:,} & {r['random']:.4f} & {r['uncertainty']:.4f} & {100 * (r['uncertainty'] / r['random'] - 1):+.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (G / "tab_active.tex").write_text("\n".join(lines))

hyp = pd.read_csv(R / "stage4_hypotheses.csv"); conv = pd.read_csv(R / "stage4_convergence.csv")
spec = pd.read_csv(R / "stage4_input_spectrum.csv")
M = {
    "FNOrel": f"{opt.loc['Adam','test_rel_L2_mean']:.4f}", "FNOstd": f"{opt.loc['Adam','test_rel_L2_std']:.4f}",
    "AdamState": f"{opt.loc['Adam','opt_state_MB_end']:.1f}", "AdamPeakGPU": f"{opt.loc['Adam','peak_gpu_MB']:,.0f}",
    "ParamMB": f"{opt.loc['Adam','param_MB']:.1f}",
    "RsixteenRel": f"{opt.loc['GaLore r=16','test_rel_L2_mean']:.4f}", "RsixteenState": f"{opt.loc['GaLore r=16','opt_state_MB_end']:.1f}",
    "RfourState": f"{opt.loc['GaLore r=4','opt_state_MB_end']:.1f}", "RfourRel": f"{opt.loc['GaLore r=4','test_rel_L2_mean']:.4f}",
    "AdaHiRel": f"{opt.loc['adaptive tau=0.999','test_rel_L2_mean']:.4f}", "AdaHiEnd": f"{opt.loc['adaptive tau=0.999','opt_state_MB_end']:.1f}",
    "AdaHiPeak": f"{opt.loc['adaptive tau=0.999','opt_state_MB_peak']:.1f}", "AdaHiMean": f"{opt.loc['adaptive tau=0.999','opt_state_MB_mean']:.1f}",
    "AdaLoRel": f"{opt.loc['adaptive tau=0.99','test_rel_L2_mean']:.4f}", "AdaLoPeak": f"{opt.loc['adaptive tau=0.99','opt_state_MB_peak']:.1f}",
    "AdaLoEnd": f"{opt.loc['adaptive tau=0.99','opt_state_MB_end']:.1f}",
    "PeakGPUlow": f"{opt.peak_gpu_MB.iloc[1:].min():,.0f}", "PeakGPUhigh": f"{opt.peak_gpu_MB.iloc[1:].max():,.0f}",
    "EnsAdam": f"{opt.loc['Adam','ensemble rel L2']:.4f}", "EnsRsixteen": f"{opt.loc['GaLore r=16','ensemble rel L2']:.4f}",
    "EnsGainLo": f"{100*ens['gain vs mean member'].min():.0f}", "EnsGainHi": f"{100*ens['gain vs mean member'].max():.0f}",
    "SpearLo": f"{ens['Spearman(spread, error)'].min():.2f}", "SpearHi": f"{ens['Spearman(spread, error)'].max():.2f}",
    "SpreadRatio": f"{ens['spread / error (mean)'].mean():.2f}",
    "KthreeRsixteenState": f"{opt.loc['GaLore r=16','K3_opt_state_MB_peak']:.1f}",
    "KthreeRsixteenResident": f"{opt.loc['GaLore r=16','K3_resident_MB_peak']:.0f}",
    "AdamResident": f"{opt.loc['Adam','opt_state_MB_end'] + opt.loc['Adam','param_MB']:.0f}",
    "SolverMean": f"{agree['rel L2 mean']:.1e}", "SolverMax": f"{agree['rel L2 max']:.1e}", "SolverMedian": f"{agree['rel L2 median']:.1e}",
    "WrongSign": f"{hyp[hyp.sign < 0]['rel L2 mean'].min():.2f}",
    "NuThousand": f"{hyp[(hyp.sign > 0) & (hyp.nu < 0.0015)]['rel L2 mean'].iloc[0]:.1e}",
    "NuFiveHundred": f"{hyp[(hyp.sign > 0) & (hyp.nu > 0.0015)]['rel L2 mean'].iloc[0]:.1e}",
    "SpecLo": f"{spec[spec.k <= 31].ratio.min():.2f}", "SpecHi": f"{spec[spec.k <= 31].ratio.max():.2f}",
    "Grid256": f"{conv[(conv.grid == 256) & (conv.method == 'cn_heun')]['vs dataset y'].iloc[0]:.1e}",
    "FNOms": f"{fno_ms:.2f}",
    "SolThirtyTwoms": f"{be[be.solver.str.startswith('32^2')]['solver ms/sample'].iloc[0]:.2f}",
    "SolThirtyTwoRel": f"{be[be.solver.str.startswith('32^2')]['solver rel L2'].iloc[0]:.4f}",
    "SolThirtyTwoSpeed": f"{1/be[be.solver.str.startswith('32^2')]['speedup (solver/FNO)'].iloc[0]:.0f}",
    "SolFortyEightms": f"{be[be.solver.str.startswith('48^2')]['solver ms/sample'].iloc[0]:.2f}",
    "SolFortyEightRel": f"{be[be.solver.str.startswith('48^2')]['solver rel L2'].iloc[0]:.4f}",
    "SolOneTwentyEightms": f"{be[be.solver.str.startswith('128^2')]['solver ms/sample'].iloc[0]:.1f}",
    "SolOneTwentyEightSpeed": f"{be[be.solver.str.startswith('128^2')]['speedup (solver/FNO)'].iloc[0]:.1f}",
    "BreakEvenFull": f"{be[be.solver.str.startswith('128^2')]['break-even queries'].iloc[0]:,.0f}",
    "TrainMin": f"{train_s/60:.0f}", "GenMin": f"{gen_s/60:.1f}",
    "UnetMSE": f"{u.loc[mse_lbl,'test_rel_L2']:.4f}", "UnetKtwo": f"{u.loc[k2_lbl,'test_rel_L2']:.4f}",
    "BandMidMSE": f"{bands.iloc[0,2]:.3f}", "BandMidKtwo": f"{bands.iloc[1,2]:.3f}",
    "BandHighMSE": f"{bands.iloc[0,3]:.1f}", "BandHighKtwo": f"{bands.iloc[1,3]:.1f}",
}
(G / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
print(be.to_string(float_format="%.4g"))
