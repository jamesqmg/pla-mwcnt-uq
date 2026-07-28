"""Tests for the directional (anisotropic) MC ensemble -- RED phase.

Production counterpart to test_rve_ensemble.py. The directional ensemble draws
random transverse packings of continuous straight fibers at a fixed target Vf
and returns the transversely-isotropic engineering moduli (E_axial, E_transverse)
per realization -- the spread is the MC band for Fig 3 / Fig 4. Meshes are kept
small so the six-load-case extraction stays fast in CI.

Run:  python -m pytest src/tests/test_rve_directional_ensemble.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_directional_ensemble import (  # noqa: E402
    DirectionalEnsembleConfig,
    run_directional_mc_ensemble,
)

MATRIX = (3.023, 0.35)
INCLUSION = (1000.0, 0.2)


def _config(solver="direct"):
    # Deliberately coarse/small: these are integration smoke tests for the
    # ensemble plumbing + physics ordering (E1>E2), not accuracy tests. Keep the
    # six-load-case extractions cheap so the fast suite stays fast.
    return DirectionalEnsembleConfig(
        domain_size=6.0, n_elements_per_side=10,
        matrix_props=MATRIX, inclusion_props=INCLUSION, axis=2, solver=solver,
    )


class TestDirectionalEnsemble:
    def test_shapes_and_length(self):
        res = run_directional_mc_ensemble(0.12, _config(), n_realizations=3, base_seed=2026)
        assert res.e_axial.shape == (3,)
        assert res.e_transverse.shape == (3,)
        assert res.achieved_vf.shape == (3,)

    def test_axial_stiffer_than_transverse_and_above_matrix(self):
        res = run_directional_mc_ensemble(0.12, _config(), n_realizations=2, base_seed=7)
        assert np.all(res.e_axial > res.e_transverse)   # aligned fibers stiffen the axis
        assert np.all(res.e_axial > MATRIX[0])          # stiffer than the neat matrix
        assert np.all(res.e_transverse > 0.0)

    def test_reproducible(self):
        a = run_directional_mc_ensemble(0.12, _config(), 2, base_seed=99)
        b = run_directional_mc_ensemble(0.12, _config(), 2, base_seed=99)
        np.testing.assert_array_equal(a.e_axial, b.e_axial)
        np.testing.assert_array_equal(a.e_transverse, b.e_transverse)

    def test_different_seed_differs(self):
        a = run_directional_mc_ensemble(0.12, _config(), 2, base_seed=1)
        b = run_directional_mc_ensemble(0.12, _config(), 2, base_seed=2)
        assert not np.array_equal(a.e_axial, b.e_axial)

    def test_invalid_n_realizations_raises(self):
        with pytest.raises(ValueError):
            run_directional_mc_ensemble(0.12, _config(), 0, base_seed=1)

    def test_iterative_matches_direct(self):
        """config.solver is forwarded; iterative reproduces direct moduli."""
        d = run_directional_mc_ensemble(0.12, _config("direct"), 1, base_seed=5)
        it = run_directional_mc_ensemble(0.12, _config("iterative"), 1, base_seed=5)
        assert abs(it.e_axial[0] - d.e_axial[0]) / d.e_axial[0] < 1e-3
        assert abs(it.e_transverse[0] - d.e_transverse[0]) / d.e_transverse[0] < 1e-3
