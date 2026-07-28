"""
Tests for src/uq/bayesian.run_bayesian_loo — honest out-of-sample Bayesian LOO.

Replaces the in-sample Bayesian evaluation (PICP=1.000) with leave-one-out
posterior-predictive intervals computed from EXACT quantiles (no z-ratio
extrapolation). Addresses reviews C4/C-rec1 (in-sample vs LOO mismatch) and
P#6 (Gaussian z-ratio approximation in the reliability figure).

Run:  python -m pytest src/tests/test_bayesian_loo.py -v
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.bayesian import run_bayesian_loo
from src.uq.bootstrap import COVERAGE_LEVELS

# Fast MCMC settings for tests
KW = dict(draws=150, tune=150, chains=2)


def test_loo_returns_expected_keys_and_ranges():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 10)
    y = 3 * x + 10 + rng.normal(0, 2, size=10)
    res = run_bayesian_loo(x, y, seed=42, **KW)

    for k in ("picp_90", "mpiw_90", "picp_95", "mpiw_95", "crps", "nlpd",
              "ece", "reliability_curve", "loo_pred_mean", "loo_pred_std",
              "loo_y_true", "n"):
        assert k in res, f"missing key {k}"

    assert 0.0 <= res["picp_90"] <= 1.0
    assert 0.0 <= res["picp_95"] <= 1.0
    assert res["mpiw_90"] > 0
    assert res["mpiw_95"] >= res["mpiw_90"]  # 95% interval wider than 90%
    assert res["n"] == 10


def test_loo_arrays_match_n():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 5, 9)
    y = 2 * x + 5 + rng.normal(0, 3, size=9)
    res = run_bayesian_loo(x, y, seed=7, **KW)
    assert len(res["loo_pred_mean"]) == 9
    assert len(res["loo_pred_std"]) == 9
    assert len(res["loo_y_true"]) == 9


def test_reliability_curve_is_nominal_empirical_pairs():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 4, 10)
    y = 4 * x + 8 + rng.normal(0, 2, size=10)
    res = run_bayesian_loo(x, y, seed=3, **KW)
    curve = res["reliability_curve"]
    assert len(curve) == len(COVERAGE_LEVELS)
    nominals = [c[0] for c in curve]
    assert nominals == list(COVERAGE_LEVELS)  # ordered, all levels present
    for nominal, empirical in curve:
        assert 0.0 <= empirical <= 1.0


def test_loo_pred_std_strictly_positive():
    rng = np.random.default_rng(4)
    x = np.linspace(0, 5, 8)
    y = 3 * x + rng.normal(0, 2, size=8)
    res = run_bayesian_loo(x, y, seed=11, **KW)
    assert all(s > 0 for s in res["loo_pred_std"])
