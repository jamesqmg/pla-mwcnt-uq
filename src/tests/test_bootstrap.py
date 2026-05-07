"""
Tests for src/uq/bootstrap.py  — RED phase (all should fail before implementation)

Run:  python -m pytest src/tests/test_bootstrap.py -v
"""
import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.bootstrap import (
    fit_regression,
    bootstrap_predict,
    compute_picp,
    compute_mpiw,
    compute_crps,
    run_bootstrap_pipeline,
)

RNG = np.random.default_rng(42)

# ── Synthetic data fixtures ────────────────────────────────────────────────────

@pytest.fixture
def linear_data():
    """Clean linear relationship: y = 50 + 3*x + noise. n=20."""
    x = np.linspace(0, 5, 20)
    y = 50 + 3 * x + RNG.normal(0, 2, size=20)
    return x, y


@pytest.fixture
def small_data():
    """Minimal dataset: n=8."""
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    y = np.array([44.0, 46.0, 49.0, 51.0, 55.0, 53.0, 57.0, 58.0])
    return x, y


# ── fit_regression ─────────────────────────────────────────────────────────────

class TestFitRegression:

    def test_returns_fitted_model(self, linear_data):
        x, y = linear_data
        model = fit_regression(x, y, degree=1)
        assert hasattr(model, "predict")

    def test_linear_predictions_reasonable(self, linear_data):
        x, y = linear_data
        model = fit_regression(x, y, degree=1)
        preds = model.predict(x.reshape(-1, 1))
        # R² should be high for clean linear data
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.85

    def test_degree_2_polynomial_accepted(self, linear_data):
        x, y = linear_data
        model = fit_regression(x, y, degree=2)
        preds = model.predict(x.reshape(-1, 1))
        assert preds.shape == (len(x),)

    def test_invalid_degree_raises(self, linear_data):
        x, y = linear_data
        with pytest.raises((ValueError, Exception)):
            fit_regression(x, y, degree=0)


# ── bootstrap_predict ─────────────────────────────────────────────────────────

class TestBootstrapPredict:

    def test_returns_three_arrays(self, linear_data):
        x, y = linear_data
        x_pred = np.linspace(0, 5, 10)
        point, lower, upper = bootstrap_predict(x, y, x_pred, B=100, alpha=0.10)
        assert point.shape == (10,)
        assert lower.shape == (10,)
        assert upper.shape == (10,)

    def test_lower_leq_point_leq_upper(self, linear_data):
        x, y = linear_data
        x_pred = np.linspace(0, 5, 10)
        point, lower, upper = bootstrap_predict(x, y, x_pred, B=200, alpha=0.10)
        assert (lower <= point).all()
        assert (point <= upper).all()

    def test_interval_width_positive(self, linear_data):
        x, y = linear_data
        x_pred = np.array([1.0, 2.5, 4.0])
        _, lower, upper = bootstrap_predict(x, y, x_pred, B=200, alpha=0.10)
        assert (upper - lower > 0).all()

    def test_b1000_runs_without_error(self, linear_data):
        x, y = linear_data
        x_pred = np.array([2.5])
        point, lower, upper = bootstrap_predict(x, y, x_pred, B=1000, alpha=0.10)
        assert np.isfinite(point[0])
        assert np.isfinite(lower[0])
        assert np.isfinite(upper[0])

    def test_90pct_interval_wider_than_50pct(self, linear_data):
        x, y = linear_data
        x_pred = np.array([2.5])
        _, lo90, up90 = bootstrap_predict(x, y, x_pred, B=500, alpha=0.10)
        _, lo50, up50 = bootstrap_predict(x, y, x_pred, B=500, alpha=0.50)
        width90 = (up90 - lo90)[0]
        width50 = (up50 - lo50)[0]
        assert width90 > width50

    def test_deterministic_with_seed(self, linear_data):
        x, y = linear_data
        x_pred = np.array([2.5])
        _, lo1, up1 = bootstrap_predict(x, y, x_pred, B=200, alpha=0.10, seed=99)
        _, lo2, up2 = bootstrap_predict(x, y, x_pred, B=200, alpha=0.10, seed=99)
        np.testing.assert_allclose(lo1, lo2)
        np.testing.assert_allclose(up1, up2)

    def test_small_dataset_does_not_crash(self, small_data):
        x, y = small_data
        x_pred = np.array([1.0, 3.0])
        point, lower, upper = bootstrap_predict(x, y, x_pred, B=200, alpha=0.10)
        assert np.isfinite(lower).all()
        assert np.isfinite(upper).all()


# ── compute_picp ──────────────────────────────────────────────────────────────

class TestComputePicp:

    def test_perfect_coverage_returns_one(self):
        y_true = np.array([50.0, 55.0, 60.0])
        lower  = np.array([45.0, 50.0, 55.0])
        upper  = np.array([55.0, 60.0, 65.0])
        assert compute_picp(y_true, lower, upper) == pytest.approx(1.0)

    def test_zero_coverage_returns_zero(self):
        y_true = np.array([50.0, 55.0, 60.0])
        lower  = np.array([60.0, 65.0, 70.0])
        upper  = np.array([70.0, 75.0, 80.0])
        assert compute_picp(y_true, lower, upper) == pytest.approx(0.0)

    def test_partial_coverage(self):
        y_true = np.array([50.0, 75.0])  # first in interval, second out
        lower  = np.array([45.0, 45.0])
        upper  = np.array([55.0, 55.0])
        assert compute_picp(y_true, lower, upper) == pytest.approx(0.5)

    def test_output_in_zero_one(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(50, 5, 20)
        lower  = rng.normal(45, 5, 20)
        upper  = lower + 10
        picp = compute_picp(y_true, lower, upper)
        assert 0.0 <= picp <= 1.0

    def test_boundary_inclusion(self):
        """Values exactly on interval boundary count as covered."""
        y_true = np.array([50.0, 60.0])
        lower  = np.array([50.0, 55.0])
        upper  = np.array([55.0, 60.0])
        assert compute_picp(y_true, lower, upper) == pytest.approx(1.0)


# ── compute_mpiw ──────────────────────────────────────────────────────────────

class TestComputeMpiw:

    def test_uniform_width(self):
        lower = np.array([40.0, 45.0, 50.0])
        upper = np.array([50.0, 55.0, 60.0])
        assert compute_mpiw(lower, upper) == pytest.approx(10.0)

    def test_variable_width_mean(self):
        lower = np.array([40.0, 45.0])
        upper = np.array([50.0, 65.0])   # widths 10 and 20
        assert compute_mpiw(lower, upper) == pytest.approx(15.0)

    def test_mpiw_is_positive(self):
        lower = np.array([40.0, 45.0])
        upper = np.array([50.0, 55.0])
        assert compute_mpiw(lower, upper) > 0

    def test_zero_width_interval(self):
        lower = np.array([50.0, 50.0])
        upper = np.array([50.0, 50.0])
        assert compute_mpiw(lower, upper) == pytest.approx(0.0)


# ── compute_crps ──────────────────────────────────────────────────────────────

class TestComputeCrps:

    def test_perfect_point_forecast_near_zero(self):
        """CRPS → 0 as forecast distribution concentrates on true value."""
        rng = np.random.default_rng(0)
        y_true = np.array([50.0])
        # Bootstrap samples all very close to truth
        boot_samples = rng.normal(50.0, 0.001, (1, 500))
        crps = compute_crps(y_true, boot_samples)
        assert crps < 0.5  # near-zero for near-perfect forecast

    def test_poor_forecast_higher_crps(self):
        """CRPS increases when forecast distribution is far from truth."""
        rng = np.random.default_rng(0)
        y_true = np.array([50.0])
        # Good: samples near 50
        boot_good = rng.normal(50.0, 1.0, (1, 500))
        # Bad: samples near 80
        boot_bad  = rng.normal(80.0, 1.0, (1, 500))
        crps_good = compute_crps(y_true, boot_good)
        crps_bad  = compute_crps(y_true, boot_bad)
        assert crps_bad > crps_good

    def test_output_is_positive(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(50, 5, 5)
        boot_samples = rng.normal(50, 5, (5, 200))
        crps = compute_crps(y_true, boot_samples)
        assert crps >= 0

    def test_shape_mismatch_raises(self):
        y_true = np.array([50.0, 55.0])
        boot_samples = np.random.normal(50, 5, (3, 100))  # wrong n
        with pytest.raises((ValueError, Exception)):
            compute_crps(y_true, boot_samples)


# ── run_bootstrap_pipeline (integration) ──────────────────────────────────────

class TestRunBootstrapPipeline:

    def test_returns_dict_with_required_keys(self, linear_data):
        x, y = linear_data
        result = run_bootstrap_pipeline(x, y, B=200, alpha=0.10, seed=42)
        for key in ("picp_90", "picp_95", "mpiw_90", "mpiw_95", "crps",
                    "lower_90", "upper_90", "lower_95", "upper_95", "point_pred"):
            assert key in result, f"Missing key: {key}"

    def test_picp_in_range(self, linear_data):
        x, y = linear_data
        result = run_bootstrap_pipeline(x, y, B=500, alpha=0.10, seed=42)
        assert 0.0 <= result["picp_90"] <= 1.0
        assert 0.0 <= result["picp_95"] <= 1.0

    def test_95pct_interval_wider_than_90pct(self, linear_data):
        x, y = linear_data
        result = run_bootstrap_pipeline(x, y, B=500, alpha=0.10, seed=42)
        assert result["mpiw_95"] >= result["mpiw_90"]

    def test_crps_is_finite_positive(self, linear_data):
        x, y = linear_data
        result = run_bootstrap_pipeline(x, y, B=500, alpha=0.10, seed=42)
        assert np.isfinite(result["crps"])
        assert result["crps"] >= 0

    def test_leave_one_out_evaluation(self, linear_data):
        """Pipeline should use held-out evaluation, not in-sample."""
        x, y = linear_data
        result = run_bootstrap_pipeline(x, y, B=200, alpha=0.10, seed=42,
                                        eval_mode="loo")
        assert "picp_90" in result

    def test_real_dataset_runs_end_to_end(self):
        """Smoke test on the actual filtered thesis dataset."""
        from src.uq.data_loader import load_dataset, filter_pla_fdm
        from pathlib import Path
        csv = Path(__file__).parents[2] / "data/extraction/master_dataset_strict.csv"
        df = filter_pla_fdm(load_dataset(csv))
        x = df["cnt_wt_pct"].values.astype(float)
        y = df["tensile_mpa"].values.astype(float)
        result = run_bootstrap_pipeline(x, y, B=200, alpha=0.10, seed=42)
        assert np.isfinite(result["picp_90"])
        assert result["mpiw_90"] > 0
