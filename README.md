# Bootstrap vs Bayesian MCMC: Tensile Strength UQ for FDM PLA/MWCNT

Uncertainty quantification is straightforward with a thousand specimens. Materials research rarely has thirty. This is the code for my thesis at Mapúa University (Dual ME/MSE), and the question is narrow: at n=26, does bootstrap resampling or Bayesian MCMC give better calibrated prediction intervals?

Short answer: neither. Under a like-for-like leave-one-out test the two tie on score. They part only in how they fail.

## The honest result

An earlier version of this work claimed a large Bayesian advantage in NLPD and a 5.6x saving in specimen budget. Both were an estimator artefact. PSIS-LOO was scoring the Bayesian model against a different target than the bootstrap's brute-force LOO, so the two were never compared like for like. Refitting the sampler on each of the 26 held-out points removes the mismatch, and the advantage goes with it.

| Metric (n=26, brute-force LOO) | Bootstrap | Bayesian NUTS |
|---|---|---|
| PICP@90% | 0.846 | 1.000 |
| MPIW@90% | 25.9 MPa | 28.9 MPa |
| CRPS | 4.59 | 5.03 |
| NLPD | 3.591 | 3.576 |
| ECE | 0.105 | 0.099 |

Bootstrap under-covers. It promises 90 percent and delivers 85, which is overconfidence you would not catch without holding points out. Bayesian over-covers, catching every point but paying about 3 MPa of extra width for it. On NLPD they sit 0.015 nats apart. That is a tie, not a winner.

Where Bayesian does earn its cost is partial pooling. On the n=44 pool split into FDM and non-FDM groups, the hierarchical model reaches PICP 0.909 and recovers a CNT-loading response 2.2 times stronger in the FDM group than outside it (R-hat 1.0066, ESS 1092). Letting the data-starved group borrow strength from the larger one is the thing flat bootstrap cannot do at all.

## Scope note

Neither method reaches 90 percent coverage on the pooled data at any n tested. Both hold 88 percent from n=5, so they separate on interval width rather than coverage. That ceiling is itself the finding: pool variance, not algorithm choice, is what binds here.

## What this does

The dataset is n=26 PLA+FDM+ASTM D638 tensile measurements curated from published literature and filtered to a single material system. Small sample size is the experimental condition, not a weakness to apologise for.

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
  run_*.py                  drivers that write into results/

src/tests/                  442 tests across 28 files
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
python -m pytest src/tests/ -q                 # 442 tests
python -m pytest src/tests/ -q -m "not slow"   # skip NUTS and the FE solves
```

MCMC runs are gated on convergence: R-hat below 1.01 and ESS above 400, checked with arviz. A sampler run that fails either is treated as a failed run, not a result.

## Stack

Python 3.13 · PyMC 5.28.4 · arviz 0.23.4 · scikit-learn 1.8.0 · scikit-fem 12.0.1 · numpy 2.4

## License

MIT, see [LICENSE](LICENSE). The dataset under `data/extraction/` is curated from published
literature; cite the original papers listed in `References/references.bib` if you use it.
