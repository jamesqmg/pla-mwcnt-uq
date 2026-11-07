# Bootstrap vs Bayesian MCMC: Tensile Strength UQ for FDM PLA/MWCNT

Uncertainty quantification is straightforward with a thousand specimens. Materials research rarely has thirty. This is the code for my thesis at Mapúa University (Dual ME/MSE), and the question is narrow: at n=21, does bootstrap resampling or Bayesian MCMC give better calibrated prediction intervals?

Short answer: no difference is detected. Under a like-for-like leave-one-out test the two score the same within noise. They part only in how they fail.

## The honest result

**Correction, November 2026.** Earlier versions of this repository reported n=26. The strict filter in `data_loader.py` matched "FDM" inside "NOT FDM", which let five extruded-filament rows from Velu 2025 into the strict set. The filter is now negation-aware and tested; the strict set is n=21 (Hardani 2024, 15 rows; Vidakis 2021, 6 rows). The numbers below are from `src/uq/run_estcon_revision.py` and its `results/estcon_revision_*.json`. The older strict-set results (`phase*`, `subsampling_*`, `inversion_table.json`, `surrogate_validation.json`) were computed on the n=26 set and are kept as a record; do not quote them. Files ending `_n21` are the same runs redone on n=21. The `rve_*` and `hierarchical_*` files never used the strict filter.

An earlier version also claimed a large Bayesian advantage in NLPD and a 5.6x saving in specimen budget. Both were an estimator artefact: PSIS-LOO scored the Bayesian model against a different target than the bootstrap's brute-force LOO. Refitting both on every held-out point removes the mismatch, and the advantage goes with it.

| Metric (n=21, brute-force LOO) | Bootstrap | Bayesian NUTS |
|---|---|---|
| PICP@90% (held-out points covered) | 0.857 (18/21) | 0.952 (20/21) |
| MPIW@90% | 24.0 MPa | 27.3 MPa |
| CRPS | 4.78 | 4.71 |
| NLPD | 3.556 | 3.523 |
| ECE | 0.115 | 0.077 |

Bootstrap under-covers: it promises 90 percent and delivers 86. Bayesian catches 20 of 21 for 3.3 MPa of extra width. Paired per-fold tests find no difference in NLPD (bootstrap minus Bayes +0.033, 95% CI −0.036 to +0.117) or CRPS (+0.072, −0.076 to +0.255), and the two disagree on coverage in only 2 of 21 folds (exact McNemar p = 0.50). At n=21 that is "no difference detected", not "equivalent".

On a pool of 44 rows split by processing route (FDM 34, non-FDM 10), a hierarchical model leans toward a steeper CNT-loading response for FDM (standardised slope 0.37 against 0.16; posterior probability 0.78) without resolving it. Route and source study coincide in this pool, so the route effect cannot be separated from a study effect. What partial pooling does show is the non-FDM group borrowing strength: its unpooled slope of −0.03 becomes a positive, honestly wide estimate.

## Scope note

On held-out FDM rows (fit on n, score the remaining 34 − n), neither method holds 90 percent coverage at any training size from 10 to 30. The bootstrap under-covers by 4 to 10 points throughout; the Bayesian model is near nominal only where its intervals are wide. Pool variance, not algorithm choice, is what binds here.

## What this does

The dataset is n=21 PLA+FDM+ASTM D638 tensile measurements curated from published literature and filtered to a single material system. Small sample size is the experimental condition, not a weakness to apologise for.

Three phases on the statistical side:

1. **Bootstrap** (`bootstrap.py`) at 1000 resamples, prediction intervals at 90% nominal coverage
2. **Bayesian NUTS** (`bayesian.py`) with weakly informative priors, 4 chains via PyMC, same coverage target
3. **Subsampling** (`subsampling.py`) running both across an n-ladder to map where they diverge

Prediction is then checked against physics on a four-tier ladder. Tier 1 is the leave-one-out test above. Tier 2 is analytical: Halpin-Tsai and Mori-Tanaka surrogates, each validated against its isotropic-sphere and rule-of-mixtures limits. Tier 3 is finite-element homogenisation over a representative volume element, which is what most of `rve_*.py` implements. Tier 4 is in-house specimens and is future work.

## Layout

```
src/uq/
  data_loader.py            load and filter the dataset
  bootstrap.py              Phase 1
  bayesian.py               Phase 2
  subsampling.py            Phase 3 divergence curves
  hierarchical.py           partial pooling across FDM / non-FDM groups
  inversion.py              minimum-n inversion of the divergence curves
  micromechanics.py         Halpin-Tsai surrogate
  mori_tanaka.py            Tandon-Weng / Mori-Tanaka surrogate
  rve_geometry.py           periodic RSA capsule packing
  rve_mesh.py               tet mesh and phase tagging
  rve_solve.py              periodic-BC elasticity solve
  rve_fft.py                spectral solver
  rve_wavy_geometry.py      wavy-fibre geometry
  rve_orientation.py        orientation sampling
  rve_ensemble.py           Monte Carlo ensemble wrapper
  run_estcon_revision.py    n=21 analyses: LOO, specimen budget, hierarchical
  run_*.py                  drivers that write into results/

src/tests/                  444 tests across 28 files
results/                    JSON outputs, one per experiment
figures/                    generators (*.py) and outputs; style.py holds the shared palette
data/extraction/            master_dataset.csv and per-paper CSVs
References/                 references.bib
```

This repository carries the code, data and results. The manuscript, the conference papers and the defense material live separately and are not published here.

## Setup

```bash
conda create -n thesis python=3.13
conda activate thesis
pip install pymc==5.28.4 arviz==0.23.4 scikit-learn==1.8.0 scikit-fem==12.0.1 \
    uncertainty-toolbox pandas matplotlib numpy
```

## Tests

```bash
python -m pytest src/tests/ -q                 # 444 tests (3 skip without optional builders)
python -m pytest src/tests/ -q -m "not slow"   # skip NUTS and the FE solves
```

MCMC runs are gated on convergence: R-hat below 1.01 and ESS above 400, checked with arviz. A sampler run that fails either is treated as a failed run, not a result.

## Stack

Python 3.13 · PyMC 5.28.4 · arviz 0.23.4 · scikit-learn 1.8.0 · scikit-fem 12.0.1 · numpy 2.4

## License

MIT, see [LICENSE](LICENSE). The dataset under `data/extraction/` is curated from published
literature; cite the original papers listed in `References/references.bib` if you use it.
