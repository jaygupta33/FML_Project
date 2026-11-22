"""Data loading and preparation, same steps as the main notebook (dataset + preparation sections):
load, seed-0 permutation, 80/20 train/validation split, train-only normalisation, channel dim, loaders."""
import glob
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

FILES = ("nsforcing_train_128.pt", "nsforcing_test_128.pt")
EXPECTED = {"train": ((10000, 128, 128), 0.708), "test": ((2000, 128, 128), 0.706)}


def find_data_dir():
    """First folder under /kaggle/input (or ./data locally) that holds both .pt files."""
    hits = sorted({os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/{FILES[0]}", recursive=True)})
    hits += ["data", "/kaggle/working/data"]
    for d in hits:
        if all(os.path.exists(f"{d}/{f}") for f in FILES):
            return d
    raise FileNotFoundError("nsforcing_*_128.pt not found; attach the ns-data kernel output as input")


def check(train, test):
    """Stop before training if the data differs from the team's dataset."""
    problems = []
    for split, ds in (("train", train), ("test", test)):
        shape, y_std = EXPECTED[split]
        for key in ("x", "y"):
            if tuple(ds[key].shape) != shape:
                problems.append(f"{split} {key} shape {tuple(ds[key].shape)}, expected {shape}")
        measured = ds["y"].float().std().item()
        if abs(measured - y_std) > 0.01:
            problems.append(f"{split} y std {measured:.4f}, expected {y_std} ± 0.01")
    if problems:
        raise RuntimeError("Data does not match the team's dataset:\n  " + "\n  ".join(problems))


def load(data_dir=None, batch_size=32):
    data_dir = data_dir or find_data_dir()
    train = torch.load(f"{data_dir}/{FILES[0]}")
    test = torch.load(f"{data_dir}/{FILES[1]}")
    check(train, test)

    torch.manual_seed(0)
    x_tr, y_tr = train["x"].float(), train["y"].float()
    x_te, y_te = test["x"].float(), test["y"].float()
    perm = torch.randperm(len(x_tr), generator=torch.Generator().manual_seed(0))
    x_tr, y_tr = x_tr[perm], y_tr[perm]
    cut = int(0.8 * len(x_tr))
    x_va, y_va = x_tr[cut:], y_tr[cut:]
    x_tr, y_tr = x_tr[:cut], y_tr[:cut]

    xm, ym, xs, ys = x_tr.mean(), y_tr.mean(), x_tr.std(), y_tr.std()   # train statistics only
    norm = lambda t, m, s: ((t - m) / s).unsqueeze(1)
    x_tr, x_va, x_te = (norm(t, xm, xs) for t in (x_tr, x_va, x_te))
    y_tr, y_va, y_te = (norm(t, ym, ys) for t in (y_tr, y_va, y_te))

    return SimpleNamespace(
        data_dir=data_dir, x_mean=xm, x_std=xs, y_mean=ym, y_std=ys,
        train_loader=DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(TensorDataset(x_va, y_va), batch_size=batch_size),
        test_loader=DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size),
        x_test=x_te, y_test=y_te,
    )
