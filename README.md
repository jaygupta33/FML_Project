# Rank-Adaptive Optimization for Memory-Efficient Neural Surrogates of Navier-Stokes Flow

FNO and U-Net surrogates for 2-D Navier-Stokes (forcing → vorticity at T = 5, Zenodo 12825163), trained with
Adam, fixed-rank GaLore and an adaptive-rank variant; three-member ensembles; and a pseudo-spectral solver that
reproduces the dataset, used to compare surrogate cost against solver cost at matched accuracy.

Report: `report/report.tex` (all numbers come from `report/generated/`, which `tools/make_final.py` builds).
Main results table: `results/FINAL.csv`.

## Layout
| Path | Contents |
|---|---|
| `navier-stokes.ipynb` | original team notebook (data, models, trainer, gradient study) |
| `notebooks/` | one notebook per stage (built by `tools/stage*_notebooks.py`) |
| `nsbench/` | models, `Trainer`, `SpectralLoss` (generated from the main notebook by `tools/build_nsbench.py`), data loading, output helpers |
| `optim/lowrank_adam.py` | complex-aware low-rank Adam, fixed or adaptive rank |
| `solver/ns2d.py` | batched pseudo-spectral NS solver (CPU/GPU) |
| `tests/` | 37 unit tests (`python -m pytest tests -q`) |
| `results/`, `figures/` | stage CSVs and figures; `figures/final/` for the report |
| `kaggle/`, `kaggle_out/` | Kaggle kernel folders and downloaded outputs/logs |
| `docs/dataset_notes.md` | what the dataset is, with sources and the Stage 4 confirmation |

## Stages
| Stage | Notebook | Where it ran |
|---|---|---|
| 1 Baselines: FNO + Adam (3 seeds), U-Net lr sweep + spectral loss | `stage1_fno_seeds`, `stage1_unet_control` | Kaggle T4 |
| 2 Optimizer screening (10 epochs) and finalists (20 epochs × 3 seeds) | `stage2_screen`, `stage2_final_a/b` | Kaggle T4 |
| 3 K = 3 ensembles from saved test predictions | `stage3_ensembles` | local CPU |
| 4 Solver validation against the dataset | `stage4_solver_validation` | local CPU |
| 5 Solver cost-accuracy frontier and surrogate inference timing | `stage5_timing` | Kaggle T4 |
| 8 Break-even, final tables and figures | `tools/make_final.py` | local |

Not done (out of time): active learning (Stage 7), multi-step rollout (Stage 6).

## Reproduce
```bash
python -m venv .venv && .venv/bin/pip install torch numpy pandas matplotlib scipy neuraloperator pytest nbclient ipykernel
.venv/bin/python -m pytest tests -q
# data: run kaggle/ns-data (downloads and md5-checks the Zenodo archive) or place nsforcing_*_128.pt in data/
# GPU stages: tools/stage_code.sh uploads nsbench/optim/solver as a private Kaggle dataset; push kaggle/<stage>/
# local stages: .venv/bin/python tools/run_local.py notebooks/stage4_solver_validation.ipynb notebooks/stage3_ensembles.ipynb
.venv/bin/python tools/make_final.py      # results/FINAL.csv, figures/final/, report/generated/
cd report && pdflatex report.tex && pdflatex report.tex
```
