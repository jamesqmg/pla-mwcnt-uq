"""
Tests for src/uq/rve_ensemble.py — RED phase (all should fail before implementation)

Tier 3 / Phase 4 Track B4: Monte Carlo ensemble wrapper around the
geometry -> mesh -> solve pipeline (rve_geometry.py, rve_mesh.py,
rve_solve.py). Each realization is an independent random RVE draw at the
same target volume fraction; the ensemble of effective moduli is what
turns a single point-solve into the Fig 4 prediction band.

Run:  python -m pytest src/tests/test_rve_ensemble.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_ensemble import RVEEnsembleConfig, run_mc_ensemble  # noqa: E402

E_MATRIX, NU_MATRIX = 3.0, 0.35
E_INCLUSION, NU_INCLUSION = 1000.0, 0.2
UNIAXIAL_STRAIN = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

TARGET_VF = 0.02
CONFIG = RVEEnsembleConfig(
    aspect_ratio=8.0, domain_size=15.0, n_elements_per_side=8,
    matrix_props=(E_MATRIX, NU_MATRIX), inclusion_props=(E_INCLUSION, NU_INCLUSION),
    macro_strain=UNIAXIAL_STRAIN,
)


class TestRunMcEnsemble:
    def test_output_shape(self):
        out = run_mc_ensemble(TARGET_VF, CONFIG, n_realizations=3, base_seed=0)
        assert out.shape == (3,)

    def test_deterministic_given_same_base_seed(self):
        out1 = run_mc_ensemble(TARGET_VF, CONFIG, n_realizations=3, base_seed=42)
        out2 = run_mc_ensemble(TARGET_VF, CONFIG, n_realizations=3, base_seed=42)
        np.testing.assert_allclose(out1, out2)

    def test_different_base_seed_gives_different_results(self):
        out1 = run_mc_ensemble(TARGET_VF, CONFIG, n_realizations=3, base_seed=1)
        out2 = run_mc_ensemble(TARGET_VF, CONFIG, n_realizations=3, base_seed=2)
        assert not np.allclose(out1, out2)

    def test_single_realization(self):
        out = run_mc_ensemble(TARGET_VF, CONFIG, n_realizations=1, base_seed=0)
        assert out.shape == (1,)

    def test_stiff_filler_increases_mean_effective_modulus(self):
        out = run_mc_ensemble(TARGET_VF, CONFIG, n_realizations=3, base_seed=0)
        assert out.mean() > E_MATRIX

    def test_zero_realizations_raises(self):
        with pytest.raises(ValueError):
            run_mc_ensemble(TARGET_VF, CONFIG, n_realizations=0, base_seed=0)

    def test_negative_realizations_raises(self):
        with pytest.raises(ValueError):
            run_mc_ensemble(TARGET_VF, CONFIG, n_realizations=-1, base_seed=0)

    def test_zero_loading_strain_raises(self):
        zero_strain_config = RVEEnsembleConfig(
            aspect_ratio=CONFIG.aspect_ratio, domain_size=CONFIG.domain_size,
            n_elements_per_side=CONFIG.n_elements_per_side,
            matrix_props=CONFIG.matrix_props, inclusion_props=CONFIG.inclusion_props,
            macro_strain=np.zeros((3, 3)),
        )
        with pytest.raises(ValueError):
            run_mc_ensemble(TARGET_VF, zero_strain_config, n_realizations=1, base_seed=0)
