"""
Tests for src/uq/run_subsampling_refit.py — reproducible subsampling refit runner.

The original generator that produced results/subsampling_*.json was a one-off and
was not committed. This runner replaces it so the 500-repeat refit (review P#1/#7,
smoothing the specimen-budget inversion cliff) is reproducible. The genuinely new
logic here is merge_subsampling_results (union bootstrap-only + bayesian-key records
by n); the experiment functions themselves are already covered by test_subsampling.py.

Run:  python -m pytest src/tests/test_subsampling_refit.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.run_subsampling_refit import merge_subsampling_results


def _boot(n, picp=0.9, mpiw=40.0):
    return {"n": n, "picp_90_bootstrap": picp, "mpiw_90_bootstrap": mpiw,
            "picp_90_bootstrap_std": 0.0}


def _bayes(n, picp=1.0, mpiw=50.0):
    return {"n": n, "picp_90_bayesian": picp, "mpiw_90_bayesian": mpiw,
            "bayesian_effective_repeats": 5}


def test_merge_unions_keys_by_n():
    boot = [_boot(5), _boot(10), _boot(20)]
    bayes = [_bayes(5), _bayes(20)]  # bayesian only at key n
    merged = merge_subsampling_results(boot, bayes)

    by_n = {r["n"]: r for r in merged}
    # all bootstrap n present
    assert sorted(by_n) == [5, 10, 20]
    # n with both methods carries both metric families
    assert "picp_90_bootstrap" in by_n[5] and "picp_90_bayesian" in by_n[5]
    assert by_n[5]["picp_90_bayesian"] == 1.0
    # n with bootstrap only has no bayesian keys
    assert "picp_90_bayesian" not in by_n[10]


def test_merge_is_sorted_ascending_by_n():
    boot = [_boot(44), _boot(5), _boot(10)]
    merged = merge_subsampling_results(boot, [])
    assert [r["n"] for r in merged] == [5, 10, 44]


def test_merge_does_not_mutate_inputs():
    boot = [_boot(5)]
    bayes = [_bayes(5)]
    merge_subsampling_results(boot, bayes)
    assert "picp_90_bayesian" not in boot[0]  # input untouched


def test_merge_rejects_bayesian_n_absent_from_bootstrap():
    boot = [_boot(5)]
    bayes = [_bayes(99)]  # n not in bootstrap ladder
    with pytest.raises(ValueError, match="not present in bootstrap"):
        merge_subsampling_results(boot, bayes)
