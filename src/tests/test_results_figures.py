"""
Numeric-output + render smoke tests for the Track B results and figures.

Per the project convention, figures are validated on their numeric outputs (the result
JSONs) rather than pixel content; the build() calls are exercised only as a render
smoke test to catch import/plotting breakage.

Run:  python -m pytest src/tests/test_results_figures.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    return json.loads(path.read_text())


class TestPhase2InformedPriorResult:

    def test_flat_prior_reproduces_canonical(self):
        d = _load("phase2_informed_prior.json")["flat_prior"]
        # Byte-stable reproduction of the canonical Phase 2 numbers.
        assert d["mpiw_90_mpa"] == pytest.approx(28.6, abs=0.2)
        assert d["nlpd_psis_loo"] == pytest.approx(1.49, abs=0.02)
        assert d["rhat_max"] < 1.01
        assert d["converged"] is True

    def test_informed_prior_nudges_slope_positive_but_data_dominates(self):
        d = _load("phase2_informed_prior.json")
        flat = d["flat_prior"]["slope_raw_mpa_per_wt"]
        informed = d["informed_prior"]["slope_raw_mpa_per_wt"]
        prior_mean = d["informed_prior_spec"]["slope_mean_mpa_per_wt"]
        # informed slope sits between the flat estimate and the prior mean (data dominates).
        assert flat < informed < prior_mean

    def test_predictive_metrics_essentially_unchanged(self):
        d = _load("phase2_informed_prior.json")
        f, i = d["flat_prior"], d["informed_prior"]
        assert abs(f["mpiw_90_mpa"] - i["mpiw_90_mpa"]) < 1.0
        assert abs(f["nlpd_psis_loo"] - i["nlpd_psis_loo"]) < 0.1


class TestPhase2cPaperInterceptResult:

    def test_converged_and_tau_alpha_positive(self):
        d = _load("hierarchical_phase2c_paper_intercept.json")
        assert d["convergence"]["passed"] is True
        assert d["tau_alpha_posterior_mean"] > 0.0

    def test_between_study_variation_exceeds_slope_split(self):
        # The headline B3 finding: tau_alpha (study baseline) > tau_beta (process slope).
        d = _load("hierarchical_phase2c_paper_intercept.json")
        assert d["tau_alpha_posterior_mean"] > d["tau_beta_posterior_mean"]

    def test_six_papers_and_finite_intercepts(self):
        d = _load("hierarchical_phase2c_paper_intercept.json")
        alpha = d["alpha_paper_posterior_mean"]
        assert d["n_papers"] == 6
        assert len(alpha) == 6
        assert all(np.isfinite(v) for v in alpha.values())


class TestSurrogateValidationResult:

    def test_directional_agreement_and_modulus_strength_gap(self):
        d = _load("surrogate_validation.json")
        assert d["trend_sign_agreement"] is True
        assert d["independent_of_tensile_data"] is True
        assert d["magnitude_ratio_modulus_to_strength"] > 10.0

    def test_uses_flat_prior(self):
        assert _load("surrogate_validation.json")["prior"] == "flat"


class TestFiguresRender:
    """Smoke test: each results figure builds and writes a PNG without error."""

    @pytest.mark.parametrize("module_name, out_png", [
        ("figures.build_fig_prior_sensitivity", "fig_prior_sensitivity.png"),
        ("figures.build_fig_paper_intercept", "fig_paper_intercept.png"),
        ("figures.build_fig_surrogate_crosscheck", "fig_surrogate_crosscheck.png"),
    ])
    def test_figure_builds(self, module_name, out_png):
        import importlib
        out = ROOT / "figures" / out_png
        json_needed = {
            "figures.build_fig_prior_sensitivity": "phase2_informed_prior.json",
            "figures.build_fig_paper_intercept": "hierarchical_phase2c_paper_intercept.json",
            "figures.build_fig_surrogate_crosscheck": "surrogate_validation.json",
        }[module_name]
        if not (RESULTS / json_needed).exists():
            pytest.skip(f"{json_needed} not generated yet")
        if importlib.util.find_spec(module_name) is None:
            pytest.skip(f"{module_name} not written yet")
        mod = importlib.import_module(module_name)
        mod.build()
        assert out.exists() and out.stat().st_size > 0
