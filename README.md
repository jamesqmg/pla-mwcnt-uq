# Bootstrap vs. Bayesian MCMC — Tensile Strength UQ for FDM PLA/MWCNT

Code for my thesis at Mapúa University (Dual ME/MSE, 5th year). The question is straightforward: when you only have ~26 data points, which uncertainty quantification method gives you better-calibrated prediction intervals — bootstrap resampling or Bayesian MCMC?

Short answer: Bayesian wins at n ≤ 10 (~6 percentage points higher PICP). By n ≥ 20 they're basically the same.

## What this does

The dataset is n=26 PLA+FDM+ASTM D638 tensile measurements pulled from published literature and filtered down to a single material system. Small sample size is the whole point — that's what makes this a real problem in materials research.

Three phases:

1. **Bootstrap** — 1000 resamples, prediction intervals at 90% nominal coverage
2. **Bayesian NUTS** — weakly informative priors, 4 chains × 1000 draws via PyMC, same coverage target
3. **Subsampling** — run both methods at n = 5, 8, 10, 15, 20, 26, 30, 35, 44 to map out where they diverge

The divergence curves (PICP vs. n) are the main output.

## Results

| Method | PICP@90% in-sample | PICP@90% LOO | MPIW | CRPS |
|---|---|---|---|---|
| Bootstrap | 0.962 | 0.846 | 25.9 MPa | 4.59 |
| Bayesian NUTS | 1.000 | 1.000 | 28.6 MPa | 4.57 |

Bayesian R-hat = 1.0007, ESS = 4073 (well converged). Divergence threshold n* = 10.

## Structure

```
src/uq/
├── data_loader.py      — load and filter the dataset
├── bootstrap.py        — Phase 1
├── bayesian.py         — Phase 2
└── subsampling.py      — Phase 3 divergence curves

results/                — saved JSON outputs from each phase
figures/                — figure generators (*.py) and outputs (*.png)
data/extraction/        — master_dataset.csv and per-paper CSVs
```

## Setup

```bash
conda create -n thesis python=3.13
conda activate thesis
pip install pymc==5.28.4 arviz==0.23.4 scikit-learn==1.8.0 \
    uncertainty-toolbox pandas matplotlib numpy
```

## Running tests

```bash
python -m pytest src/tests/ -q              # 106 tests, ~40s
python -m pytest src/tests/ -q -m "not slow"  # skip NUTS, ~8s
```

## Stack

Python 3.13 · PyMC 5.28.4 · arviz 0.23.4 · scikit-learn 1.8.0 · numpy 2.4
