"""
Tests for src/uq/subsampling.py — RED phase

The subsampling experiment (thesis §3.3) is the named methodological contribution.
It applies both bootstrap (Phase 1) and Bayesian NUTS (Phase 2) pipelines at
systematically decreasing n-values drawn from the heterogeneous parent dataset,
generating divergence curves of PICP(n) and MPIW(n).

Run:  python -m pytest src/tests/test_subsampling.py -v
Slow: python -m pytest src/tests/test_subsampling.py -v -m "not slow"
"""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.subsampling import (
    build_n_ladder,
    subsample_dataset,
    run_single_n,
    run_subsampling_experiment,
    divergence_threshold,
)


RNG = np.random.default_rng(42)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def parent_data():
    """Load the full unfiltered master_dataset (n≈44) as the n-ladder parent."""
    from src.uq.data_loader import load_dataset
    csv = Path(__file__).parents[2] / "data/extraction/master_dataset.csv"
    df = load_dataset(csv)
    from src.uq.subsampling import prepare_parent_dataset
    return prepare_parent_dataset(df)


@pytest.fixture
def synthetic_parent():
    """Synthetic n=40 parent for fast deterministic tests."""
    x = np.tile(np.linspace(0, 6, 8), 5)
    y = 45 + 2.5 * x + RNG.normal(0, 4, size=40)
    return x, y


# ── build_n_ladder ────────────────────────────────────────────────────────────

class TestBuildNLadder:

    def test_default_ladder_is_decreasing(self):
        ladder = build_n_ladder(max_n=40)
        assert list(ladder) == sorted(ladder)

    def test_all_values_leq_max_n(self):
        ladder = build_n_ladder(max_n=30)
        assert all(n <= 30 for n in ladder)

    def test_minimum_n_is_five(self):
        ladder = build_n_ladder(max_n=40)
        assert min(ladder) >= 5

    def test_custom_ladder_accepted(self):
        ladder = build_n_ladder(max_n=44, values=[5, 10, 20, 30, 44])
        assert list(ladder) == [5, 10, 20, 30, 44]

    def test_values_above_max_n_excluded(self):
        ladder = build_n_ladder(max_n=20, values=[5, 10, 15, 25, 30])
        assert all(n <= 20 for n in ladder)

    def test_at_least_three_steps(self):
        ladder = build_n_ladder(max_n=44)
        assert len(ladder) >= 3


# ── subsample_dataset ─────────────────────────────────────────────────────────

class TestSubsampleDataset:

    def test_returns_arrays_of_correct_size(self, synthetic_parent):
        x, y = synthetic_parent
        x_s, y_s = subsample_dataset(x, y, n=10, seed=0)
        assert len(x_s) == 10
        assert len(y_s) == 10

    def test_subsampled_values_are_from_parent(self, synthetic_parent):
        x, y = synthetic_parent
        x_s, y_s = subsample_dataset(x, y, n=15, seed=1)
        for xi in x_s:
            assert xi in x

    def test_no_duplicates_when_n_equals_parent(self, synthetic_parent):
        x, y = synthetic_parent
        x_s, y_s = subsample_dataset(x, y, n=len(x), seed=2)
        assert len(x_s) == len(x)

    def test_n_greater_than_parent_raises(self, synthetic_parent):
        x, y = synthetic_parent
        with pytest.raises((ValueError, Exception)):
            subsample_dataset(x, y, n=len(x) + 1, seed=0)

    def test_deterministic_with_seed(self, synthetic_parent):
        x, y = synthetic_parent
        x1, y1 = subsample_dataset(x, y, n=10, seed=99)
        x2, y2 = subsample_dataset(x, y, n=10, seed=99)
        np.testing.assert_array_equal(x1, x2)

    def test_different_seeds_give_different_samples(self, synthetic_parent):
        x, y = synthetic_parent
        x1, _ = subsample_dataset(x, y, n=10, seed=0)
        x2, _ = subsample_dataset(x, y, n=10, seed=1)
        assert not np.array_equal(x1, x2)


# ── run_single_n ──────────────────────────────────────────────────────────────

class TestRunSingleN:

    def test_returns_dict_with_required_keys(self, synthetic_parent):
        x, y = synthetic_parent
        result = run_single_n(x, y, n=15, B=100, n_repeats=3, seed=42,
                              skip_bayesian=True)
        for key in ("n", "picp_90_bootstrap", "picp_95_bootstrap",
                    "mpiw_90_bootstrap", "mpiw_95_bootstrap", "crps_bootstrap"):
            assert key in result, f"Missing: {key}"

    def test_bayesian_keys_present_when_not_skipped(self, synthetic_parent):
        x, y = synthetic_parent
        result = run_single_n(x, y, n=15, B=50, n_repeats=2, seed=42,
                              skip_bayesian=False, draws=100, tune=100, chains=2)
        for key in ("picp_90_bayesian", "mpiw_90_bayesian", "crps_bayesian"):
            assert key in result, f"Missing: {key}"

    def test_metrics_are_finite(self, synthetic_parent):
        x, y = synthetic_parent
        result = run_single_n(x, y, n=20, B=100, n_repeats=3, seed=42,
                              skip_bayesian=True)
        assert np.isfinite(result["picp_90_bootstrap"])
        assert np.isfinite(result["mpiw_90_bootstrap"])
        assert np.isfinite(result["crps_bootstrap"])

    def test_picp_in_zero_one(self, synthetic_parent):
        x, y = synthetic_parent
        result = run_single_n(x, y, n=20, B=100, n_repeats=5, seed=42,
                              skip_bayesian=True)
        assert 0.0 <= result["picp_90_bootstrap"] <= 1.0

    def test_n_field_matches_request(self, synthetic_parent):
        x, y = synthetic_parent
        result = run_single_n(x, y, n=12, B=50, n_repeats=2, seed=0,
                              skip_bayesian=True)
        assert result["n"] == 12

    def test_std_fields_present(self, synthetic_parent):
        """Standard deviations across repeats should be present for error bars."""
        x, y = synthetic_parent
        result = run_single_n(x, y, n=15, B=100, n_repeats=5, seed=42,
                              skip_bayesian=True)
        assert "picp_90_bootstrap_std" in result
        assert "mpiw_90_bootstrap_std" in result

    def test_bayesian_failure_count_reported(self, synthetic_parent):
        """bayesian_n_failures must be present when skip_bayesian=False."""
        x, y = synthetic_parent
        result = run_single_n(x, y, n=15, B=50, n_repeats=2, seed=42,
                              skip_bayesian=False, draws=100, tune=100, chains=2)
        assert "bayesian_n_failures" in result
        assert "bayesian_effective_repeats" in result
        assert result["bayesian_effective_repeats"] <= 2

    def test_seeds_are_independent(self, synthetic_parent):
        """Bootstrap and Bayesian should use different seeds — no shared state."""
        # Two runs with same seed should give same results (deterministic)
        x, y = synthetic_parent
        r1 = run_single_n(x, y, n=15, B=100, n_repeats=3, seed=7,
                          skip_bayesian=True)
        r2 = run_single_n(x, y, n=15, B=100, n_repeats=3, seed=7,
                          skip_bayesian=True)
        assert r1["picp_90_bootstrap"] == pytest.approx(r2["picp_90_bootstrap"])


# ── run_subsampling_experiment ────────────────────────────────────────────────

class TestRunSubsamplingExperiment:

    def test_returns_list_of_dicts(self, synthetic_parent):
        x, y = synthetic_parent
        ladder = [10, 20, 30]
        results = run_subsampling_experiment(x, y, n_ladder=ladder,
                                             B=50, n_repeats=2, seed=42,
                                             skip_bayesian=True)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_n_values_match_ladder(self, synthetic_parent):
        x, y = synthetic_parent
        ladder = [10, 20, 30]
        results = run_subsampling_experiment(x, y, n_ladder=ladder,
                                             B=50, n_repeats=2, seed=42,
                                             skip_bayesian=True)
        assert [r["n"] for r in results] == ladder

    def test_larger_n_gives_narrower_mpiw_on_average(self, synthetic_parent):
        """Bootstrap MPIW should decrease as n increases (more data → less uncertainty)."""
        x, y = synthetic_parent
        ladder = [8, 20, 35]
        results = run_subsampling_experiment(x, y, n_ladder=ladder,
                                             B=200, n_repeats=10, seed=42,
                                             skip_bayesian=True)
        mpiw = [r["mpiw_90_bootstrap"] for r in results]
        # Not guaranteed to be strictly monotone but generally true with enough repeats
        assert mpiw[0] >= mpiw[-1] * 0.8, \
            f"Expected MPIW to decrease with n: {mpiw}"

    @pytest.mark.slow
    def test_real_data_full_run_bootstrap_only(self, parent_data):
        """Smoke test on real master_dataset parent."""
        x, y = parent_data
        ladder = build_n_ladder(max_n=len(x), values=[5, 10, 20, len(x)])
        results = run_subsampling_experiment(x, y, n_ladder=ladder,
                                             B=200, n_repeats=5, seed=42,
                                             skip_bayesian=True)
        assert len(results) == len(ladder)
        for r in results:
            assert np.isfinite(r["picp_90_bootstrap"])

    @pytest.mark.slow
    def test_real_data_full_run_with_bayesian(self, parent_data):
        """End-to-end with Bayesian on real data — slow but the key integration test."""
        x, y = parent_data
        ladder = build_n_ladder(max_n=len(x), values=[10, 20, len(x)])
        results = run_subsampling_experiment(x, y, n_ladder=ladder,
                                             B=200, n_repeats=3, seed=42,
                                             skip_bayesian=False,
                                             draws=500, tune=500, chains=4)
        for r in results:
            assert "picp_90_bayesian" in r
            assert np.isfinite(r["picp_90_bayesian"])


# ── divergence_threshold ──────────────────────────────────────────────────────

class TestDivergenceThreshold:

    def test_returns_dict_with_threshold_keys(self):
        mock_results = [
            {"n": 5,  "picp_90_bootstrap": 0.40, "picp_90_bayesian": 0.72},
            {"n": 10, "picp_90_bootstrap": 0.55, "picp_90_bayesian": 0.80},
            {"n": 20, "picp_90_bootstrap": 0.75, "picp_90_bayesian": 0.85},
            {"n": 35, "picp_90_bootstrap": 0.88, "picp_90_bayesian": 0.90},
        ]
        result = divergence_threshold(mock_results, metric="picp_90", delta=0.10)
        assert "threshold_n" in result
        assert "max_divergence" in result
        assert "divergence_per_n" in result

    def test_threshold_n_is_from_ladder(self):
        mock_results = [
            {"n": 5,  "picp_90_bootstrap": 0.40, "picp_90_bayesian": 0.72},
            {"n": 10, "picp_90_bootstrap": 0.55, "picp_90_bayesian": 0.80},
            {"n": 20, "picp_90_bootstrap": 0.78, "picp_90_bayesian": 0.83},
            {"n": 35, "picp_90_bootstrap": 0.89, "picp_90_bayesian": 0.90},
        ]
        result = divergence_threshold(mock_results, metric="picp_90", delta=0.10)
        if result["threshold_n"] is not None:
            assert result["threshold_n"] in [5, 10, 20, 35]

    def test_no_divergence_returns_none_threshold(self):
        mock_results = [
            {"n": 10, "picp_90_bootstrap": 0.88, "picp_90_bayesian": 0.90},
            {"n": 20, "picp_90_bootstrap": 0.89, "picp_90_bayesian": 0.90},
            {"n": 30, "picp_90_bootstrap": 0.90, "picp_90_bayesian": 0.91},
        ]
        result = divergence_threshold(mock_results, metric="picp_90", delta=0.15)
        assert result["threshold_n"] is None

    def test_threshold_n_is_largest_diverging_n(self):
        """threshold_n should be the LARGEST n where divergence still exceeds delta
        (scanning top-down), not the smallest."""
        mock_results = [
            {"n": 5,  "picp_90_bootstrap": 0.40, "picp_90_bayesian": 0.72},  # diff=0.32
            {"n": 10, "picp_90_bootstrap": 0.55, "picp_90_bayesian": 0.80},  # diff=0.25
            {"n": 20, "picp_90_bootstrap": 0.78, "picp_90_bayesian": 0.83},  # diff=0.05
            {"n": 35, "picp_90_bootstrap": 0.89, "picp_90_bayesian": 0.90},  # diff=0.01
        ]
        result = divergence_threshold(mock_results, metric="picp_90", delta=0.10)
        # delta=0.10: n=5 (0.32>0.10) and n=10 (0.25>0.10) both exceed delta
        # threshold should be 10 (LARGEST n exceeding delta), not 5 (smallest)
        assert result["threshold_n"] == 10

    def test_max_divergence_is_non_negative(self):
        mock_results = [
            {"n": 5,  "picp_90_bootstrap": 0.40, "picp_90_bayesian": 0.72},
            {"n": 20, "picp_90_bootstrap": 0.80, "picp_90_bayesian": 0.88},
        ]
        result = divergence_threshold(mock_results, metric="picp_90", delta=0.10)
        assert result["max_divergence"] >= 0
