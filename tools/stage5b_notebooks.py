"""Build the Stage 5b notebook: FNO inference in half precision (and torch.compile) on the T4, with accuracy.

No FNO weights were saved earlier, so the FNO + Adam seed-0 run of Stage 1A is repeated (same protocol), then the
same trained weights are timed and evaluated in each inference mode. Each mode is wrapped so a failure is recorded
in the table instead of stopping the run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_notebook import build, SETUP, DATA
from stage1_notebooks import kernel, ROOT

CELLS = [
    ("md", "# Stage 5b: FNO Inference in Half Precision"),
    ("md", r"""
Stage 5 timed the FNO in float32 only. Here the same trained FNO is timed and evaluated in other inference modes on
the same GPU type (Tesla T4, which has float16 tensor cores but no bfloat16 or TF32 support).

| mode | what changes |
|---|---|
| `float32` | as in Stage 5 (re-timed here so all modes share one session) |
| `autocast fp16` | `torch.autocast("cuda", torch.float16)`: matrix products and convolutions run in float16 |
| `compile float32` | `torch.compile(model)` |
| `compile + autocast fp16` | both |

Training repeats Stage 1A seed 0 (FNO, Adam, lr $10^{-3}$, MSE, 20 epochs, batch 32). Timing protocol as in
Stage 5: one warm-up call, `torch.cuda.synchronize()` around timed calls, median of 3 repeats of 10 calls, batch
sizes 1, 32 and 256. Accuracy: test relative $L^2$ on all 2,000 test samples, raw units.
"""),
    ("md", "## Setup"),
    ("code", SETUP + "import time, traceback\n"),
    ("md", "## Data"),
    ("code", DATA),
    ("md", "## Train the FNO (Stage 1A protocol, seed 0)"),
    ("code", '''
EPOCHS, LR = 20, 1e-3
tr = Trainer(nsbench.FourierNeuralOperator, lambda p: torch.optim.Adam(p, lr=LR),
             D.train_loader, D.val_loader, D.test_loader, D.y_mean, D.y_std, device,
             seed=0, label="FNO Adam seed 0").fit(EPOCHS)
summ = tr.summary()                                   # loads the best-validation weights
torch.save(tr.model.state_dict(), "/kaggle/working/fno_adam_seed0.pt" if os.path.isdir("/kaggle/working") else "fno_adam_seed0.pt")
save_csv(run_record(summ, run="FNO", optimizer="Adam", lr=LR, seed=0, epochs=EPOCHS, batch=32), "stage5b_fno_train")
summ[["best_epoch", "test_rel_L2", "sec_per_epoch"]]
'''),
    ("md", "## Inference modes"),
    ("code", '''
model = tr.model.eval()
x_test, y_test = D.x_test, D.y_test

def runner(mode):
    m = torch.compile(model) if mode.startswith("compile") else model
    amp = "fp16" in mode
    @torch.no_grad()
    def f(x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return m(x)
    return f

@torch.no_grad()
def test_rel_l2(f, batch=64):
    err = norm = 0.0
    for i in range(0, len(x_test), batch):
        p = f(x_test[i:i + batch].to(device)).float() * D.y_std.to(device) + D.y_mean.to(device)
        t = y_test[i:i + batch].to(device) * D.y_std.to(device) + D.y_mean.to(device)
        err += (p - t).flatten(1).norm(dim=1).sum().item(); norm += t.flatten(1).norm(dim=1).sum().item()
    return err / norm

def time_call(f, bs, repeats=3, warmup=1, inner=10):
    x = torch.randn(bs, 1, 128, 128, device=device)
    for _ in range(warmup):
        f(x)
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(inner):
            f(x)
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) / inner)
    return sorted(times)[len(times) // 2]

rows = []
for mode in ("float32", "autocast fp16", "compile float32", "compile + autocast fp16"):
    try:
        torch._dynamo.reset()
        f = runner(mode)
        t0 = time.perf_counter(); acc = test_rel_l2(f); first_s = time.perf_counter() - t0   # includes compilation
        for bs in (1, 32, 256):
            sec = time_call(f, bs)
            rows.append({"mode": mode, "batch": bs, "sec_per_call": sec, "sec_per_sample": sec / bs,
                         "test_rel_L2": acc, "first_pass_s": first_s, "error": ""})
    except Exception as e:
        traceback.print_exc()
        rows.append({"mode": mode, "batch": None, "sec_per_call": None, "sec_per_sample": None,
                     "test_rel_L2": None, "first_pass_s": None, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    torch.cuda.empty_cache()
modes = pd.DataFrame(rows)
save_csv(run_record(modes, model="FNO", trained="Adam seed 0, 20 epochs"), "stage5b_fp16")
modes
'''),
    ("md", "## Best time per sample for each mode"),
    ("code", '''
ok = modes.dropna(subset=["sec_per_sample"])
best = ok.loc[ok.groupby("mode")["sec_per_sample"].idxmin(), ["mode", "batch", "sec_per_sample", "test_rel_L2"]]
best = best.assign(ms_per_sample=best.sec_per_sample * 1e3,
                   speedup_vs_float32=best.sec_per_sample.rdiv(best.loc[best["mode"] == "float32", "sec_per_sample"].iloc[0]))
best.sort_values("ms_per_sample")
'''),
]

if __name__ == "__main__":
    nb = ROOT / "notebooks/stage5b_fp16.ipynb"
    build(CELLS, nb)
    kernel("stage5b_fp16", "ns-stage5b-fp16", "NS Stage5b fp16", nb)
