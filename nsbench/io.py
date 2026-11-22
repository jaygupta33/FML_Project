"""Output helpers: CSVs to results/, figures to figures/, test predictions to preds/ (Kaggle: /kaggle/working)."""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".")
RESULTS_DIR, FIG_DIR, PRED_DIR = OUT_DIR / "results", OUT_DIR / "figures", OUT_DIR / "preds"
for d in (RESULTS_DIR, FIG_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).replace("λ", "lam")).strip("_")


def save_csv(df, name, index=False):
    p = RESULTS_DIR / f"{safe_name(name)}.csv"
    df.to_csv(p, index=index)
    print(f"[SAVED] {p}")
    return p


def save_fig(name, fig):
    p = FIG_DIR / f"{safe_name(name)}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {p}")
    return p


@torch.no_grad()
def save_predictions(trainer, x_test, name, batch=64):
    """Best-weights test predictions in standardized units, float16 .npy (for ensembles later)."""
    m = trainer.model.eval()
    out = [m(x_test[i:i + batch].to(trainer.device)).float().cpu() for i in range(0, len(x_test), batch)]
    pred = torch.cat(out).squeeze(1)
    p = PRED_DIR / f"{safe_name(name)}.npy"
    np.save(p, pred.numpy().astype(np.float16))
    print(f"[SAVED] {p}")
    return pred                                     # float32, standardized units


def run_record(summary_df, **config):
    """Summary row plus the run's config and environment, so every CSV row is self-describing."""
    env = dict(torch=torch.__version__, python=sys.version.split()[0],
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    return summary_df.assign(**{k: str(v) if isinstance(v, (list, tuple, dict)) else v
                                for k, v in {**config, **env}.items()})
