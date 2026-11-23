"""Build the Stage 7 active-learning notebooks: one kernel per acquisition strategy (uncertainty, random).

Pool-based active learning on the 8,000 training samples: their labels already exist, so "running a simulation"
means revealing a label. Both strategies share the same initial set, schedule, optimizer and seeds; only the
choice of the next samples differs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_notebook import build, SETUP, DATA
from stage1_notebooks import kernel, ROOT

STRATEGIES = ("uncertainty", "random")


def cells(strategy):
    return [
        ("md", f"# Stage 7: Active Learning ({strategy} selection)"),
        ("md", r"""
Can the ensemble's disagreement choose which simulations to run next? We start from a small labelled set,
train a $K=3$ ensemble, add the samples the ensemble picks, and repeat. The same loop with **random** picks
is the control (run in a separate kernel with identical settings).

| setting | value |
|---|---|
| pool | the 8,000 training samples (labels revealed only when picked) |
| initial set | 1,000 random samples (seed 0), the same for both strategies |
| rounds | labelled set 1,000 → 1,500 → 2,000 → 2,500 → 3,000 (500 added per round) |
| ensemble | 3 FNOs, seeds 0, 1, 2, retrained from scratch every round |
| optimizer | low-rank Adam, fixed rank $r=16$ (Stage 2), lr $10^{-3}$, refresh 200 |
| training | 20 epochs, batch 32, MSE; best epoch by validation loss (fixed 2,000-sample validation set) |
| uncertainty score | member spread: $\sqrt{\text{mean over pixels of the variance across members}}$ |
| `uncertainty` picks | the 500 pool samples with the largest spread |
| `random` picks | 500 pool samples drawn at random (seed = round) |
| metric | ensemble test relative $L^2$ on the 2,000 test samples, raw units |
"""),
        ("md", "## Setup"),
        ("code", SETUP + "from optim import LowRankAdam\nfrom scipy.stats import spearmanr\nimport time\n"),
        ("md", "## Data"),
        ("code", DATA),
        ("md", "## Settings"),
        ("code", f'''
STRATEGY = "{strategy}"
SMOKE = os.environ.get("AL_SMOKE") == "1"          # tiny local check only
N0, ADD, ROUNDS = (64, 32, 2) if SMOKE else (1000, 500, 5)
EPOCHS = 1 if SMOKE else 20
SEEDS, LR, RANK, REFRESH, BATCH = (0, 1, 2), 1e-3, 16, 200, 32

x_pool, y_pool = D.train_loader.dataset.tensors     # standardized, [8000, 1, 128, 128]
if SMOKE:
    x_pool, y_pool = x_pool[:256], y_pool[:256]
    D.val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*[t[:64] for t in D.val_loader.dataset.tensors]), batch_size=BATCH)
    D.x_test, D.y_test = D.x_test[:64], D.y_test[:64]
    D.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(D.x_test, D.y_test), batch_size=BATCH)
labelled = torch.randperm(len(x_pool), generator=torch.Generator().manual_seed(0))[:N0].tolist()
pd.DataFrame([{{"strategy": STRATEGY, "pool": len(x_pool), "initial": len(labelled), "added per round": ADD,
               "rounds": ROUNDS, "epochs": EPOCHS, "rank": RANK}}])
'''),
        ("md", "## Helpers"),
        ("code", '''
@torch.no_grad()
def predict(model, x, batch=64):
    model.eval()
    return torch.cat([model(x[i:i + batch].to(device)).float().cpu() for i in range(0, len(x), batch)])

def spread(P):                     # P: [K, N, 1, n, n] -> per-sample member spread, standardized units
    return P.var(0, unbiased=True).flatten(1).mean(1).sqrt()

def rel_l2(pred, y):               # Trainer convention: summed error norms / summed true norms, raw units
    p, t = pred * D.y_std + D.y_mean, y * D.y_std + D.y_mean
    return ((p - t).flatten(1).norm(dim=1).sum() / t.flatten(1).norm(dim=1).sum()).item()

def per_sample_rel(pred, y):
    p, t = pred * D.y_std + D.y_mean, y * D.y_std + D.y_mean
    return (p - t).flatten(1).norm(dim=1) / t.flatten(1).norm(dim=1)
'''),
        ("md", "## Active-learning loop"),
        ("code", '''
rounds, members, picks = [], [], []
picks += [{"index": i, "round_added": 0} for i in labelled]
for rnd in range(ROUNDS):
    t0 = time.time()
    ds = torch.utils.data.TensorDataset(x_pool[labelled], y_pool[labelled])
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)
    test_preds, models = [], []
    for s in SEEDS:
        tr = Trainer(nsbench.FourierNeuralOperator,
                     lambda p: LowRankAdam(p, lr=LR, refresh=REFRESH, rank_mode="fixed", rank=RANK),
                     loader, D.val_loader, D.test_loader, D.y_mean, D.y_std, device,
                     seed=s, label=f"{STRATEGY} round {rnd} n={len(labelled)} seed {s}").fit(EPOCHS)
        summ = tr.summary()                                     # loads the best-validation weights
        members.append(run_record(summ, strategy=STRATEGY, round=rnd, n_labelled=len(labelled), seed=s,
                                  optimizer=f"GaLore r={RANK}", lr=LR, epochs=EPOCHS, batch=BATCH))
        test_preds.append(predict(tr.model, D.x_test))
        models.append(tr.model)
    P = torch.stack(test_preds)
    ens = P.mean(0)
    err, sprd = per_sample_rel(ens, D.y_test), spread(P)
    row = {"strategy": STRATEGY, "round": rnd, "n_labelled": len(labelled),
           "ensemble_test_rel_L2": rel_l2(ens, D.y_test),
           "member_test_rel_L2_mean": float(np.mean([rel_l2(p, D.y_test) for p in P])),
           "spearman_spread_error_test": spearmanr(sprd.numpy(), err.numpy()).correlation}

    # choose the next samples from the unlabelled pool
    if rnd < ROUNDS - 1:
        unl = sorted(set(range(len(x_pool))) - set(labelled))
        pool_spread = spread(torch.stack([predict(m, x_pool[unl]) for m in models]))
        if STRATEGY == "uncertainty":
            chosen = [unl[i] for i in pool_spread.argsort(descending=True)[:ADD].tolist()]
        else:
            g = torch.Generator().manual_seed(rnd + 1)
            chosen = [unl[i] for i in torch.randperm(len(unl), generator=g)[:ADD].tolist()]
        pos = {u: i for i, u in enumerate(unl)}
        row.update(pool_spread_mean=pool_spread.mean().item(),
                   chosen_spread_mean=pool_spread[[pos[c] for c in chosen]].mean().item())
        labelled += chosen
        picks += [{"index": i, "round_added": rnd + 1} for i in chosen]
    row["round_minutes"] = (time.time() - t0) / 60
    rounds.append(row)
    del models, tr; torch.cuda.empty_cache()
    # save after every round so a timeout keeps the finished rounds
    save_csv(pd.DataFrame(rounds), f"stage7_{STRATEGY}")
    save_csv(pd.concat(members, ignore_index=True), f"stage7_{STRATEGY}_members")
    save_csv(pd.DataFrame(picks), f"stage7_{STRATEGY}_picks")
pd.DataFrame(rounds)
'''),
        ("md", "## Members per round"),
        ("code", '''
M = pd.concat(members, ignore_index=True)
M[["round", "n_labelled", "seed", "best_epoch", "test_rel_L2", "opt_state_MB", "peak_gpu_MB", "sec_per_epoch"]]
'''),
        ("md", "## Test error vs labelled set size"),
        ("code", '''
R = pd.DataFrame(rounds)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(R.n_labelled, R.ensemble_test_rel_L2, "o-", label="ensemble")
ax.plot(R.n_labelled, R.member_test_rel_L2_mean, "s--", label="mean member")
ax.set(xlabel="labelled training samples", ylabel="test relative $L^2$", yscale="log", title=f"{STRATEGY} selection")
ax.legend(); fig.tight_layout()
save_fig(f"stage7_{STRATEGY}", fig)
'''),
    ]


if __name__ == "__main__":
    for s in STRATEGIES:
        nb = ROOT / f"notebooks/stage7_al_{s}.ipynb"
        build(cells(s), nb)
        kernel(f"stage7_al_{s}", f"ns-stage7-al-{s}", f"NS Stage7 AL {s}", nb)
