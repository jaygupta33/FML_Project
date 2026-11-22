"""Build a .ipynb from a simple Python spec: a list of ("md" | "code", source) pairs."""
import json
from pathlib import Path


def build(cells, path):
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
            if kind == "md" else
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
             "source": src.strip("\n").splitlines(keepends=True)}
            for kind, src in cells
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {path}")


SETUP = '''
import sys, glob, os, importlib.util
if importlib.util.find_spec("neuralop") is None:
    !pip install -q neuraloperator
# project code (nsbench, optim, solver) comes from the private ns-surrogate-code dataset
code_dir = next((os.path.dirname(os.path.dirname(p)) for p in
                 glob.glob("/kaggle/input/**/nsbench/__init__.py", recursive=True)), os.path.abspath(".."))
sys.path.insert(0, code_dir)
import torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import nsbench
from nsbench import Trainer, save_csv, save_fig, save_predictions, run_record
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.DataFrame([{"python": sys.version.split()[0], "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "code": code_dir}])
'''

DATA = '''
D = nsbench.load()
pd.DataFrame([{"data": D.data_dir, "train batches": len(D.train_loader), "val batches": len(D.val_loader),
               "test samples": len(D.x_test), "y_mean": D.y_mean.item(), "y_std": D.y_std.item()}])
'''
