"""
Tests for src/uq/hierarchical.py — RED phase (all should fail before implementation).

Per v6_design.md §4.1 (LOCKED):
- Dataset: master_dataset.csv (n=45), NOT the n=26 strict filter
- Grouping (Option B, binary): FDM (n=35) vs non-FDM (n=10)
  - FDM = FFF (21) + "3D Printing" (13) + "Melt mixing + single-screw filament extrusion (for FDM/FFF use)" (1)
  - non-FDM = "twin-screw extrusion + injection molding" (5) + "Filament (Melt Mixing + Single-Screw Extrusion — NOT FDM)" (5)
- Priors: β_pop ~ N(0,1), σ_pop ~ HalfN(1), τ_β ~ HalfN(0.5),
          β_g ~ N(β_pop, τ_β), σ_obs ~ HalfN(1)
- Sampling: 4 chains × 1000 draws, 1000 tune
- Gates: R-hat < 1.01, ESS > 400; abort if any chain > 30 min wall-clock

Run:  python -m pytest src/tests/test_hierarchical.py -v
Skip slow:  python -m pytest src/tests/test_hierarchical.py -v -m "not slow"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.hierarchical import (
    GROUP_FDM,
    GROUP_NON_FDM,
    PROCESS_TO_GROUP,
    assign_groups,
    assign_papers,
    build_hierarchical_model,
    build_hierarchical_model_v2,
    load_hierarchical_data,
    load_hierarchical_data_with_papers,
    posterior_predict_hierarchical,
    run_hierarchical_pipeline,
    run_hierarchical_v2_pipeline,
)

RNG = np.random.default_rng(42)

MASTER_CSV = Path(__file__).parents[2] / "data/extraction/master_dataset.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Group assignment (locked Option B mapping — verbatim strings from spec)
# ──────────────────────────────────────────────────────────────────────────────

class TestGroupConstants:

    def test_group_codes_are_distinct_integers(self) -> None:
        assert isinstance(GROUP_FDM, int)
        assert isinstance(GROUP_NON_FDM, int)
        assert GROUP_FDM != GROUP_NON_FDM

    def test_group_codes_are_zero_and_one(self) -> None:
        assert {GROUP_FDM, GROUP_NON_FDM} == {0, 1}

    def test_mapping_covers_all_five_spec_strings(self) -> None:
        expected = {
            "FFF": GROUP_FDM,
            "3D Printing": GROUP_FDM,
            "Melt mixing + single-screw filament extrusion (for FDM/FFF use)": GROUP_FDM,
            "twin-screw extrusion + injection molding": GROUP_NON_FDM,
            "Filament (Melt Mixing + Single-Screw Extrusion — NOT FDM)": GROUP_NON_FDM,
        }
        for k, v in expected.items():
            assert PROCESS_TO_GROUP[k] == v, f"mismatch for {k!r}"


class TestAssignGroups:

    def test_returns_int_array_matching_input_length(self) -> None:
        series = pd.Series(["FFF", "twin-screw extrusion + injection molding", "3D Printing"])
        out = assign_groups(series)
        assert isinstance(out, np.ndarray)
        assert out.dtype.kind == "i"
        assert out.shape == (3,)

    def test_known_strings_map_correctly(self) -> None:
        series = pd.Series([
            "FFF",
            "3D Printing",
            "Melt mixing + single-screw filament extrusion (for FDM/FFF use)",
            "twin-screw extrusion + injection molding",
            "Filament (Melt Mixing + Single-Screw Extrusion — NOT FDM)",
        ])
        out = assign_groups(series)
        np.testing.assert_array_equal(
            out, np.array([GROUP_FDM, GROUP_FDM, GROUP_FDM, GROUP_NON_FDM, GROUP_NON_FDM])
        )

    def test_unknown_string_raises_keyerror(self) -> None:
        series = pd.Series(["compression molding"])
        with pytest.raises((KeyError, ValueError)):
            assign_groups(series)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (real master_dataset.csv, n=45)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not MASTER_CSV.exists(), reason="master_dataset.csv not present")
class TestLoadHierarchicalData:

    def test_returns_x_y_group_idx_arrays(self) -> None:
        x, y, group_idx = load_hierarchical_data(MASTER_CSV)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(group_idx, np.ndarray)
        assert x.shape == y.shape == group_idx.shape

    def test_n_equals_44(self) -> None:
        """Spec said 45 but one Sankar 2025 row is unparseable "NR [GRAPHICAL ONLY]" → n=44."""
        x, y, group_idx = load_hierarchical_data(MASTER_CSV)
        assert len(x) == 44, f"expected n=44 (45 raw − 1 NR row), got {len(x)}"

    def test_group_counts_match_spec(self) -> None:
        """After dropping the one NR row (FDM-group Sankar): FDM=34, non-FDM=10."""
        _, _, group_idx = load_hierarchical_data(MASTER_CSV)
        n_fdm = int((group_idx == GROUP_FDM).sum())
        n_non = int((group_idx == GROUP_NON_FDM).sum())
        assert n_fdm == 34, f"FDM count {n_fdm} != 34"
        assert n_non == 10, f"non-FDM count {n_non} != 10"

    def test_no_nan_in_x_or_y(self) -> None:
        x, y, _ = load_hierarchical_data(MASTER_CSV)
        assert not np.isnan(x).any()
        assert not np.isnan(y).any()

    def test_group_idx_values_only_zero_and_one(self) -> None:
        _, _, group_idx = load_hierarchical_data(MASTER_CSV)
        assert set(np.unique(group_idx).tolist()) <= {GROUP_FDM, GROUP_NON_FDM}


# ──────────────────────────────────────────────────────────────────────────────
# Model construction
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synth_two_group_data():
    """Synthetic two-group data with known group-level slopes."""
    n_per = 30
    x_a = RNG.uniform(-2, 2, n_per)
    x_b = RNG.uniform(-2, 2, n_per)
    # FDM group: y = 0.8 x + noise; non-FDM: y = 0.4 x + noise
    y_a = 0.8 * x_a + RNG.normal(0, 0.3, n_per)
    y_b = 0.4 * x_b + RNG.normal(0, 0.3, n_per)
    x = np.concatenate([x_a, x_b])
    y = np.concatenate([y_a, y_b])
    g = np.array([GROUP_FDM] * n_per + [GROUP_NON_FDM] * n_per, dtype=int)
    return x, y, g


class TestBuildHierarchicalModel:

    def test_returns_pymc_model(self, synth_two_group_data) -> None:
        import pymc as pm
        x, y, g = synth_two_group_data
        model = build_hierarchical_model(x, y, g, n_groups=2)
        assert isinstance(model, pm.Model)

    def test_model_has_required_variables(self, synth_two_group_data) -> None:
        x, y, g = synth_two_group_data
        model = build_hierarchical_model(x, y, g, n_groups=2)
        names = {rv.name for rv in model.unobserved_RVs}
        for var in ("beta_pop", "tau_beta", "beta_g", "sigma_obs"):
            assert var in names, f"missing variable: {var} (have {names})"

    def test_beta_g_has_group_dimension(self, synth_two_group_data) -> None:
        """β_g should be shape (n_groups, degree+1) for degree=1 → (2, 2)."""
        x, y, g = synth_two_group_data
        model = build_hierarchical_model(x, y, g, n_groups=2, degree=1)
        beta_g = next(rv for rv in model.unobserved_RVs if rv.name == "beta_g")
        # Shape should include n_groups as a dimension
        shape = tuple(beta_g.shape.eval())
        assert 2 in shape, f"β_g shape {shape} does not include n_groups=2"

    def test_observed_data_attached(self, synth_two_group_data) -> None:
        x, y, g = synth_two_group_data
        model = build_hierarchical_model(x, y, g, n_groups=2)
        observed_names = {rv.name for rv in model.observed_RVs}
        assert "y_obs" in observed_names


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline (slow — NUTS sampling)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestHierarchicalPipelineSynthetic:
    """End-to-end NUTS run on small synthetic data — must converge quickly."""

    def test_pipeline_returns_dict_with_required_keys(self, synth_two_group_data) -> None:
        x, y, g = synth_two_group_data
        result = run_hierarchical_pipeline(
            x, y, g, n_groups=2, draws=400, tune=400, chains=2, seed=0,
        )
        for key in (
            "picp_90", "picp_95", "mpiw_90", "mpiw_95", "crps",
            "convergence", "chain_runtime_sec", "n", "draws", "chains",
            "beta_g_posterior_mean", "beta_pop_posterior_mean",
        ):
            assert key in result, f"missing key: {key}"

    def test_pipeline_converges_on_clean_data(self, synth_two_group_data) -> None:
        """Use the spec sampling config (4 chains × 1000 draws) — anything less is borderline."""
        x, y, g = synth_two_group_data
        result = run_hierarchical_pipeline(
            x, y, g, n_groups=2, draws=1000, tune=1000, chains=4, seed=0,
        )
        assert result["convergence"]["rhat_max"] < 1.01, (
            f"R-hat {result['convergence']['rhat_max']:.4f} >= 1.01"
        )
        assert result["convergence"]["ess_min"] > 400, (
            f"ESS {result['convergence']['ess_min']:.1f} <= 400"
        )

    def test_pipeline_recovers_group_difference(self, synth_two_group_data) -> None:
        """FDM slope (0.8) should be greater than non-FDM slope (0.4) in posterior."""
        x, y, g = synth_two_group_data
        result = run_hierarchical_pipeline(
            x, y, g, n_groups=2, draws=500, tune=500, chains=2, seed=0,
        )
        beta_g_mean = result["beta_g_posterior_mean"]
        # beta_g_mean is shape (n_groups, degree+1); slope is index 1
        slope_fdm = beta_g_mean[GROUP_FDM, 1]
        slope_non = beta_g_mean[GROUP_NON_FDM, 1]
        assert slope_fdm > slope_non, (
            f"expected FDM slope > non-FDM; got fdm={slope_fdm:.3f} non={slope_non:.3f}"
        )

    def test_chain_runtime_recorded(self, synth_two_group_data) -> None:
        x, y, g = synth_two_group_data
        result = run_hierarchical_pipeline(
            x, y, g, n_groups=2, draws=300, tune=300, chains=2, seed=0,
        )
        runtimes = result["chain_runtime_sec"]
        assert isinstance(runtimes, (list, np.ndarray))
        assert len(runtimes) >= 1
        for t in runtimes:
            assert t > 0, "chain runtime must be positive"
            assert t < 1800, "chain runtime exceeded 30-min budget (spec gate)"


# ──────────────────────────────────────────────────────────────────────────────
# Real thesis data (slow — NUTS sampling on n=45)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(not MASTER_CSV.exists(), reason="master_dataset.csv not present")
class TestHierarchicalPipelineThesisData:

    def test_pipeline_runs_on_real_n44(self) -> None:
        x, y, g = load_hierarchical_data(MASTER_CSV)
        result = run_hierarchical_pipeline(
            x, y, g, n_groups=2, draws=500, tune=500, chains=2, seed=0,
        )
        assert result["n"] == 44
        assert 0.0 <= result["picp_90"] <= 1.0
        assert result["mpiw_90"] > 0

    def test_real_data_meets_runtime_budget(self) -> None:
        """Per §4.1: abort if any chain > 30 min on local hardware."""
        x, y, g = load_hierarchical_data(MASTER_CSV)
        t0 = time.perf_counter()
        result = run_hierarchical_pipeline(
            x, y, g, n_groups=2, draws=500, tune=500, chains=2, seed=0,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 1800, f"total pipeline took {elapsed:.0f}s (budget: per-chain 30 min)"
        for t in result["chain_runtime_sec"]:
            assert t < 1800, f"chain runtime {t:.0f}s exceeded 30-min budget"


# ──────────────────────────────────────────────────────────────────────────────
# Posterior predict
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestPosteriorPredictHierarchical:

    def test_returns_point_lower_upper_in_mpa(self, synth_two_group_data) -> None:
        """Posterior predict must return three arrays of equal length."""
        from src.uq.bayesian import standardize
        x, y, g = synth_two_group_data
        x_std, y_std, params = standardize(x, y)

        result = run_hierarchical_pipeline(
            x, y, g, n_groups=2, draws=300, tune=300, chains=2, seed=0,
        )
        idata = result["idata"]

        point, lo, up = posterior_predict_hierarchical(
            idata, x_pred_std=x_std, group_idx_pred=g, params=params, alpha=0.10,
        )
        assert point.shape == lo.shape == up.shape == x.shape
        assert (up >= lo).all(), "upper must be >= lower elementwise"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2c — per-paper random intercept (additive; v1 must stay green)
# ──────────────────────────────────────────────────────────────────────────────

class TestAssignPapers:

    def test_returns_codes_and_names(self) -> None:
        s = pd.Series(["a", "b", "a", "c", "b"])
        codes, names = assign_papers(s)
        assert codes.dtype == int
        assert len(codes) == len(s)
        assert names == ["a", "b", "c"]

    def test_codes_are_contiguous_from_zero(self) -> None:
        codes, names = assign_papers(pd.Series(["p", "q", "r", "p"]))
        assert set(codes) == set(range(len(names)))

    def test_codes_index_back_to_names(self) -> None:
        s = pd.Series(["x", "y", "x"])
        codes, names = assign_papers(s)
        assert [names[c] for c in codes] == ["x", "y", "x"]


class TestLoadWithPapers:

    def test_returns_five_arrays_and_consistent_lengths(self) -> None:
        x, y, g, p, names = load_hierarchical_data_with_papers(MASTER_CSV)
        assert len(x) == len(y) == len(g) == len(p) == 44
        assert len(names) == len(set(p))

    def test_six_papers_in_complete_pool(self) -> None:
        # The n=44 modelled pool has 6 papers; the 7th source contributes only an
        # incomplete row dropped by the (wt%, strength) mask.
        _, _, _, p, names = load_hierarchical_data_with_papers(MASTER_CSV)
        assert len(names) == 6
        assert int(p.max()) + 1 == 6


@pytest.fixture
def synth_paper_data():
    """Two groups x four papers (two per group) with distinct intercept offsets.

    Each group holds two papers so the group-level and paper-level intercepts stay
    jointly identifiable; a single paper per group would confound the two intercepts
    and induce a hierarchical funnel.
    """
    rng = np.random.default_rng(7)
    blocks, gs, ps = [], [], []
    # (paper, group, intercept offset): monotone offsets, papers split across groups.
    spec = [(0, GROUP_FDM, -1.5), (1, GROUP_FDM, -0.5),
            (2, GROUP_NON_FDM, 0.5), (3, GROUP_NON_FDM, 1.5)]
    for paper, grp, off in spec:
        xb = rng.uniform(-2, 2, 20)
        yb = off + 0.6 * xb + rng.normal(0, 0.3, 20)
        blocks.append((xb, yb))
        gs += [grp] * 20
        ps += [paper] * 20
    x = np.concatenate([b[0] for b in blocks])
    y = np.concatenate([b[1] for b in blocks])
    return x, y, np.array(gs, dtype=int), np.array(ps, dtype=int)


class TestBuildHierarchicalModelV2:

    def test_returns_model_with_paper_intercept_vars(self, synth_paper_data) -> None:
        import pymc as pm
        x, y, g, p = synth_paper_data
        model = build_hierarchical_model_v2(x, y, g, p, n_groups=2, n_papers=4)
        assert isinstance(model, pm.Model)
        names = {rv.name for rv in model.unobserved_RVs}
        for var in ("beta_pop", "tau_beta", "beta_g", "tau_alpha", "z_paper",
                    "alpha_paper", "sigma_obs"):
            assert var in names, f"missing {var} (have {names})"

    def test_alpha_paper_has_paper_dimension(self, synth_paper_data) -> None:
        x, y, g, p = synth_paper_data
        model = build_hierarchical_model_v2(x, y, g, p, n_groups=2, n_papers=4)
        alpha = next(rv for rv in model.unobserved_RVs if rv.name == "alpha_paper")
        assert 4 in tuple(alpha.shape.eval())

    def test_slope_pooling_preserved(self, synth_paper_data) -> None:
        # beta_g must still be present so v2 is strictly additive over v1.
        x, y, g, p = synth_paper_data
        model = build_hierarchical_model_v2(x, y, g, p, n_groups=2, n_papers=4)
        assert "beta_g" in {rv.name for rv in model.unobserved_RVs}


class TestRunHierarchicalV2Pipeline:

    @pytest.mark.slow
    def test_recovers_paper_offsets_and_reports_tau_alpha(self, synth_paper_data) -> None:
        x, y, g, p = synth_paper_data
        res = run_hierarchical_v2_pipeline(
            x, y, g, p, n_groups=2, n_papers=4, draws=1000, tune=1000, chains=4, seed=42
        )
        assert res["convergence"]["passed"], res["convergence"]
        assert "tau_alpha_posterior_mean" in res
        assert res["tau_alpha_posterior_mean"] > 0.0
        # paper intercepts should recover the -1.5/-0.5/+0.5/+1.5 ordering (monotone).
        alpha = res["alpha_paper_posterior_mean"]
        assert alpha[0] < alpha[1] < alpha[2] < alpha[3]

    def test_noncontiguous_paper_idx_raises(self, synth_paper_data) -> None:
        # A gapped paper index (missing level 1) must fail loud, not silently
        # allocate a phantom paper level.
        x, y, g, _ = synth_paper_data
        gapped = np.array([0, 2] * (len(x) // 2), dtype=int)[: len(x)]
        with pytest.raises(ValueError):
            run_hierarchical_v2_pipeline(
                x, y, g, gapped, n_groups=2, draws=50, tune=50, chains=2, seed=1
            )

    @pytest.mark.slow
    def test_infers_n_papers_when_omitted(self, synth_paper_data) -> None:
        x, y, g, p = synth_paper_data
        res = run_hierarchical_v2_pipeline(
            x, y, g, p, n_groups=2, draws=400, tune=400, chains=2, seed=1
        )
        assert res["n_papers"] == 4
