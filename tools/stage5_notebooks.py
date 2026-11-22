"""Build the Stage 5 timing notebook (GPU): solver cost vs batch size, and surrogate inference cost."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_notebook import build, SETUP
from stage1_notebooks import kernel, ROOT

SOLVER_CFG = dict(method="cn_heun", sign=1.0, nu=1 / 500)            # from Stage 4
# (grid, dt) along the cost-accuracy frontier found in Stage 4 (largest stable dt per grid)
FRONTIER = [(24, 5e-2), (32, 5e-2), (48, 5e-2), (64, 2e-2), (96, 2e-2), (128, 1e-2), (256, 4e-3)]

TIMING = [
    ("md", "# Stage 5: cost of one solve vs one surrogate prediction"),
    ("md", f"""
Wall-clock time per sample for our solver along its cost-accuracy frontier, and for the surrogates,
on the same GPU. Protocol: one warm-up call, `torch.cuda.synchronize()` around every timed call,
median of 3 repeats. Surrogate timings use freshly initialised weights (inference cost does not depend
on the weight values). Solver accuracy is measured here on 64 test samples against the dataset.

| solver setting | value |
|---|---|
| method | {SOLVER_CFG['method']}, $\\nu = 1/500$, $T = 5$ (from Stage 4) |
| (grid, dt) | {', '.join(f'({m}, {d:g})' for m, d in FRONTIER)} |
| precision | float32 (float64 also timed at $128^2$) |
"""),
    ("md", "## Setup"),
    ("code", SETUP + f'''from solver import NS2D, timed_solve
SOLVER_CFG = {SOLVER_CFG!r}
FRONTIER = {FRONTIER!r}
'''),
    ("md", "## Solver time per sample"),
    ("code", '''
from nsbench.data import find_data_dir
from solver import rel_l2
te = torch.load(f"{find_data_dir()}/nsforcing_test_128.pt")
xte, yte = te["x"][:64], te["y"][:64]
rows = []
configs = [(M, dt, torch.float32) for M, dt in FRONTIER] + [(128, 1e-2, torch.float64)]
for M, dt, prec in configs:
    s = NS2D(M, device=device, dtype=prec, dt=dt, **SOLVER_CFG)
    w = s.solve(xte, T=5.0, out_res=128).cpu()
    e = rel_l2(w.double(), yte.double())
    for bs in (1, 32, 256):
        f = xte.repeat(4, 1, 1)[:bs]
        sec, _ = timed_solve(s, f, T=5.0, repeats=3, warmup=1)
        rows.append({"what": f"solver {M}^2", "grid": M, "dt": dt, "steps": round(5 / dt),
                     "precision": str(prec).split(".")[-1], "batch": bs, "sec_per_call": sec,
                     "sec_per_sample": sec / bs, "test_rel_L2_mean": e.mean().item(), "test_rel_L2_max": e.max().item()})
        print(rows[-1])
solver_t = pd.DataFrame(rows)
solver_t
'''),
    ("md", "## Surrogate inference time per sample"),
    ("code", '''
import time

@torch.no_grad()
def time_model(model, bs, repeats=3, warmup=1, inner=10):
    x = torch.randn(bs, 1, 128, 128, device=device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(inner):
            model(x)
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) / inner)
    return sorted(times)[len(times) // 2]

rows = []
for name, mk in (("FNO", nsbench.FourierNeuralOperator), ("U-Net", nsbench.UNet)):
    torch.manual_seed(0)
    m = mk().to(device).eval()
    for bs in (1, 32, 256):
        sec = time_model(m, bs)
        rows.append({"what": name, "precision": "float32", "batch": bs, "sec_per_call": sec, "sec_per_sample": sec / bs})
    del m; torch.cuda.empty_cache()
model_t = pd.DataFrame(rows)
timing = pd.concat([solver_t, model_t], ignore_index=True)
save_csv(run_record(timing, **SOLVER_CFG, T=5.0), "stage5_timing")
timing.pivot_table(index=["what", "precision"], columns="batch", values="sec_per_sample", sort=False)
'''),
    ("md", "## Solver cost-accuracy frontier vs surrogate inference cost"),
    ("code", '''
best = timing[timing.batch == 256]
sol = best[best.what.str.startswith("solver") & (best.precision == "float32")]
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(sol.sec_per_sample, sol.test_rel_L2_mean, "o-", label="our solver (float32), grid labelled")
for _, r in sol.iterrows():
    ax.annotate(f"{r.grid}", (r.sec_per_sample, r.test_rel_L2_mean), textcoords="offset points", xytext=(4, 4), fontsize=8)
for _, r in best[~best.what.str.startswith("solver")].iterrows():
    ax.axvline(r.sec_per_sample, ls="--", color="gray"); ax.text(r.sec_per_sample, ax.get_ylim()[1], r.what, rotation=90, va="top", fontsize=8)
ax.set(xlabel="seconds per sample (batch 256)", ylabel="test relative $L^2$ vs dataset"); ax.legend(); fig.tight_layout()
save_fig("stage5_timing", fig)
'''),
]

if __name__ == "__main__":
    nb = ROOT / "notebooks/stage5_timing.ipynb"
    build(TIMING, nb)
    kernel("stage5_timing", "ns-stage5-timing", "NS Stage5 timing", nb)
