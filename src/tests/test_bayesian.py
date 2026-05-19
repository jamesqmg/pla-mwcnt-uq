"""
Tests for src/uq/bayesian.py  — RED phase (all should fail before implementation)

Run:  python -m pytest src/tests/test_bayesian.py -v
Note: NUTS tests are marked slow; skip with -m "not slow"
"""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from unittest.mock import MagicMock, patch

import arviz as az

from src.uq.bayesian import (
    standardize,
    unstandardize_predictions,
    build_model,
    run_nuts,
    check_convergence,
    posterior_predict,
    prior_predictive_check,
    run_bayesian_pipeline,
    compute_bayesian_nlpd,
)

RNG = np.random.default_rng(42)


@pytest.fixture
def linear_data():
    """Clean linear data: y = 50 + 3x + noise, n=20."""
    x = np.linspace(0, 5, 20)
    y = 50 + 3 * x + RNG.normal(0, 2, size=20)
    return x, y


@pytest.fixture
def thesis_data():
    """Real filtered thesis dataset (PLA+FDM+D638)."""
    from src.uq.data_loader import load_dataset, filter_pla_fdm
    csv = Path(__file__).parents[2] / "data/extraction/master_dataset_strict.csv"
    df = filter_pla_fdm(load_dataset(csv))
    return df["cnt_wt_pct"].values.astype(float), df["tensile_mpa"].values.astype(float)


# ── standardize / unstandardize ───────────────────────────────────────────────

class TestStandardize:

    def test_returns_two_arrays_and_params(self, linear_data):
        x, y = linear_data
        x_std, y_std, params = standardize(x, y)
        assert x_std.shape == x.shape
        assert y_std.shape == y.shape
        assert "x_mean" in params and "x_std" in params
        assert "y_mean" in params and "y_std" in params

    def test_standardized_x_has_zero_mean(self, linear_data):
        x, y = linear_data
        x_std, _, _ = standardize(x, y)
        assert abs(x_std.mean()) < 1e-10

    def test_standardized_x_has_unit_std(self, linear_data):
        x, y = linear_data
        x_std, _, _ = standardize(x, y)
        assert abs(x_std.std() - 1.0) < 1e-10

    def test_standardized_y_has_zero_mean(self, linear_data):
        x, y = linear_data
        _, y_std, _ = standardize(x, y)
        assert abs(y_std.mean()) < 1e-10

    def test_roundtrip_unstandardize(self, linear_data):
        x, y = linear_data
        x_std, y_std, params = standardize(x, y)
        y_recovered = unstandardize_predictions(y_std, params)
        np.testing.assert_allclose(y_recovered, y, rtol=1e-10)

    def test_constant_x_raises(self):
        x = np.ones(10)
        y = np.random.normal(50, 5, 10)
        with pytest.raises((ValueError, Exception)):
            standardize(x, y)


# ── build_model ───────────────────────────────────────────────────────────────

class TestBuildModel:

    def test_returns_pymc_model(self, linear_data):
        import pymc as pm
        x, y = linear_data
        x_std, y_std, _ = standardize(x, y)
        model = build_model(x_std, y_std)
        assert isinstance(model, pm.Model)

    def test_model_has_beta_prior(self, linear_data):
        x, y = linear_data
        x_std, y_std, _ = standardize(x, y)
        model = build_model(x_std, y_std)
        var_names = [v.name for v in model.free_RVs]
        assert any("beta" in n.lower() or "b" in n.lower() for n in var_names)

    def test_model_has_sigma_prior(self, linear_data):
        x, y = linear_data
        x_std, y_std, _ = standardize(x, y)
        model = build_model(x_std, y_std)
        var_names = [v.name for v in model.free_RVs]
        assert any("sigma" in n.lower() or "s" in n.lower() for n in var_names)

    def test_degree_2_model_accepted(self, linear_data):
        import pymc as pm
        x, y = linear_data
        x_std, y_std, _ = standardize(x, y)
        model = build_model(x_std, y_std, degree=2)
        assert isinstance(model, pm.Model)


# ── prior_predictive_check ────────────────────────────────────────────────────

class TestPriorPredictiveCheck:

    def test_returns_samples_in_physical_range(self, linear_data):
        """Prior predictive samples should cover plausible PLA/MWCNT tensile range."""
        x, y = linear_data
        x_std, y_std, params = standardize(x, y)
        model = build_model(x_std, y_std)
        samples_std = prior_predictive_check(model, n_samples=200, seed=42)
        # Unstandardize to get MPa
        samples_mpa = unstandardize_predictions(samples_std, params)
        # PLA/MWCNT tensile should be between 1 and 300 MPa for weakly informative priors
        p5, p95 = np.percentile(samples_mpa, [5, 95])
        assert p5 > 1.0,   f"5th pct = {p5:.1f} MPa — prior too wide (below 1 MPa)"
        assert p95 < 300.0, f"95th pct = {p95:.1f} MPa — prior too wide (above 300 MPa)"

    def test_returns_numpy_array(self, linear_data):
        x, y = linear_data
        x_std, y_std, _ = standardize(x, y)
        model = build_model(x_std, y_std)
        result = prior_predictive_check(model, n_samples=100, seed=42)
        assert isinstance(result, np.ndarray)


# ── run_nuts ──────────────────────────────────────────────────────────────────

class TestRunNuts:

    @pytest.mark.slow
    def test_returns_inference_data(self, linear_data):
        import arviz as az
        x, y = linear_data
        x_std, y_std, _ = standardize(x, y)
        model = build_model(x_std, y_std)
        idata = run_nuts(model, draws=200, tune=200, chains=2, seed=42)
        assert isinstance(idata, az.InferenceData)

    @pytest.mark.slow
    def test_posterior_has_expected_vars(self, linear_data):
        x, y = linear_data
        x_std, y_std, _ = standardize(x, y)
        model = build_model(x_std, y_std)
        idata = run_nuts(model, draws=200, tune=200, chains=2, seed=42)
        var_names = list(idata.posterior.data_vars)
        assert len(var_names) >= 2  # at least beta and sigma

    @pytest.mark.slow
    def test_chain_count_matches_request(self, linear_data):
        x, y = linear_data
        x_std, y_std, _ = standardize(x, y)
        model = build_model(x_std, y_std)
        idata = run_nuts(model, draws=100, tune=100, chains=2, seed=42)
        # Each variable should have 2 chains
        first_var = list(idata.posterior.data_vars)[0]
        assert idata.posterior[first_var].shape[0] == 2


# ── check_convergence ─────────────────────────────────────────────────────────

class TestCheckConvergence:

    @pytest.mark.slow
    def test_returns_dict_with_required_keys(self, linear_data):
        x, y = linear_data
        x_std, y_std, _ = standardize(x, y)
        model = build_model(x_std, y_std)
        idata = run_nuts(model, draws=500, tune=500, chains=4, seed=42)
        result = check_convergence(idata)
        for key in ("rhat_max", "ess_min", "passed", "rhat_per_var", "ess_per_var"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.slow
    def test_converged_chain_passes(self, linear_data):
        """Well-specified linear model should converge with draws=500."""
        x, y = linear_data
        x_std, y_std, _ = standardize(x, y)
        model = build_model(x_std, y_std)
        idata = run_nuts(model, draws=500, tune=500, chains=4, seed=42)
        result = check_convergence(idata)
        assert result["rhat_max"] < 1.01, \
            f"R-hat = {result['rhat_max']:.4f} — chain did not converge"
        assert result["ess_min"] > 400, \
            f"ESS = {result['ess_min']:.0f} — too few effective samples"

    @pytest.mark.slow
    def test_passed_flag_reflects_thresholds(self, linear_data):
        x, y = linear_data
        x_std, y_std, _ = standardize(x, y)
        model = build_model(x_std, y_std)
        idata = run_nuts(model, draws=500, tune=500, chains=4, seed=42)
        result = check_convergence(idata)
        # passed should be True iff both thresholds met
        expected = result["rhat_max"] < 1.01 and result["ess_min"] > 400
        assert result["passed"] == expected


# ── posterior_predict ─────────────────────────────────────────────────────────

class TestPosteriorPredict:

    @pytest.mark.slow
    def test_returns_three_arrays(self, linear_data):
        x, y = linear_data
        x_std, y_std, params = standardize(x, y)
        model = build_model(x_std, y_std)
        idata = run_nuts(model, draws=300, tune=300, chains=2, seed=42)
        x_pred_std = (np.linspace(0, 5, 5) - params["x_mean"]) / params["x_std"]
        point, lower, upper = posterior_predict(idata, x_pred_std, params, alpha=0.10)
        assert point.shape == (5,)
        assert lower.shape == (5,)
        assert upper.shape == (5,)

    @pytest.mark.slow
    def test_lower_leq_point_leq_upper(self, linear_data):
        x, y = linear_data
        x_std, y_std, params = standardize(x, y)
        model = build_model(x_std, y_std)
        idata = run_nuts(model, draws=300, tune=300, chains=2, seed=42)
        x_pred_std = (x - params["x_mean"]) / params["x_std"]
        point, lower, upper = posterior_predict(idata, x_pred_std, params, alpha=0.10)
        assert (lower <= point).all()
        assert (point <= upper).all()

    @pytest.mark.slow
    def test_95pct_wider_than_90pct(self, linear_data):
        x, y = linear_data
        x_std, y_std, params = standardize(x, y)
        model = build_model(x_std, y_std)
        idata = run_nuts(model, draws=300, tune=300, chains=2, seed=42)
        x_pred_std = (x - params["x_mean"]) / params["x_std"]
        _, lo90, up90 = posterior_predict(idata, x_pred_std, params, alpha=0.10)
        _, lo95, up95 = posterior_predict(idata, x_pred_std, params, alpha=0.05)
        mpiw_90 = (up90 - lo90).mean()
        mpiw_95 = (up95 - lo95).mean()
        assert mpiw_95 >= mpiw_90

    @pytest.mark.slow
    def test_predictions_in_mpa_range(self, linear_data):
        """Predictions should be in plausible MPa range after unstandardizing."""
        x, y = linear_data
        x_std, y_std, params = standardize(x, y)
        model = build_model(x_std, y_std)
        idata = run_nuts(model, draws=300, tune=300, chains=2, seed=42)
        x_pred_std = (x - params["x_mean"]) / params["x_std"]
        point, lower, upper = posterior_predict(idata, x_pred_std, params, alpha=0.10)
        assert (point > 0).all()
        assert (point < 500).all()


# ── run_bayesian_pipeline (integration) ───────────────────────────────────────

class TestRunBayesianPipeline:

    @pytest.mark.slow
    def test_returns_dict_with_required_keys(self, linear_data):
        x, y = linear_data
        result = run_bayesian_pipeline(
            x, y, draws=300, tune=300, chains=2, seed=42
        )
        for key in ("picp_90", "picp_95", "mpiw_90", "mpiw_95", "crps",
                    "lower_90", "upper_90", "lower_95", "upper_95",
                    "point_pred", "convergence", "n", "draws", "chains"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.slow
    def test_picp_in_range(self, linear_data):
        x, y = linear_data
        result = run_bayesian_pipeline(x, y, draws=500, tune=500, chains=4, seed=42)
        assert 0.0 <= result["picp_90"] <= 1.0
        assert 0.0 <= result["picp_95"] <= 1.0

    @pytest.mark.slow
    def test_95pct_wider_than_90pct(self, linear_data):
        x, y = linear_data
        result = run_bayesian_pipeline(x, y, draws=500, tune=500, chains=4, seed=42)
        assert result["mpiw_95"] >= result["mpiw_90"]

    @pytest.mark.slow
    def test_convergence_reported(self, linear_data):
        x, y = linear_data
        result = run_bayesian_pipeline(x, y, draws=500, tune=500, chains=4, seed=42)
        assert "passed" in result["convergence"]
        assert "rhat_max" in result["convergence"]
        assert "ess_min" in result["convergence"]

    @pytest.mark.slow
    def test_real_thesis_data_runs_end_to_end(self, thesis_data):
        """Smoke test on actual PLA+FDM+D638 dataset (n=26)."""
        x, y = thesis_data
        result = run_bayesian_pipeline(
            x, y, draws=500, tune=500, chains=4, seed=42
        )
        assert np.isfinite(result["picp_90"])
        assert result["mpiw_90"] > 0
        assert np.isfinite(result["crps"])
        # Report convergence status for thesis
        conv = result["convergence"]
        print(f"\nThesis data convergence: R-hat={conv['rhat_max']:.4f}, "
              f"ESS={conv['ess_min']:.0f}, passed={conv['passed']}")
        print(f"PICP@90%={result['picp_90']:.3f}, MPIW@90%={result['mpiw_90']:.2f} MPa, "
              f"CRPS={result['crps']:.4f}")


# ── compute_bayesian_nlpd ─────────────────────────────────────────────────────

class TestComputeBayesianNlpd:

    def _mock_idata(self):
        return MagicMock(spec=az.InferenceData)

    def test_returns_negative_elpd_over_n(self):
        mock_loo = MagicMock()
        mock_loo.elpd_loo = -50.0   # sum over 10 obs → per-obs = -5.0 → NLPD = 5.0
        with patch("src.uq.bayesian.az.loo", return_value=mock_loo):
            result = compute_bayesian_nlpd(self._mock_idata(), n=10)
        assert result == pytest.approx(5.0)

    def test_result_is_float(self):
        mock_loo = MagicMock()
        mock_loo.elpd_loo = -30.0
        with patch("src.uq.bayesian.az.loo", return_value=mock_loo):
            result = compute_bayesian_nlpd(self._mock_idata(), n=10)
        assert isinstance(result, float)

    def test_higher_elpd_lower_nlpd(self):
        idata_mock = self._mock_idata()
        for elpd, expected in [(-20.0, 2.0), (-40.0, 4.0)]:
            mock_loo = MagicMock()
            mock_loo.elpd_loo = elpd
            with patch("src.uq.bayesian.az.loo", return_value=mock_loo):
                nlpd = compute_bayesian_nlpd(idata_mock, n=10)
            assert nlpd == pytest.approx(expected)

    def test_positive_nlpd(self):
        # elpd_loo should always be negative (log probabilities < 1) → NLPD > 0
        mock_loo = MagicMock()
        mock_loo.elpd_loo = -25.3
        with patch("src.uq.bayesian.az.loo", return_value=mock_loo):
            result = compute_bayesian_nlpd(self._mock_idata(), n=5)
        assert result > 0.0
