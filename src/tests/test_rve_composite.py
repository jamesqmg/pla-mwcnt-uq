"""Tests for composite-voxel (partial-volume) material assignment -- RED phase.

The binary voxel tagger over-stiffens thin wavy fibers at feasible resolution
(Phase-3 finding: E_axial still dropping 14% at n_side 20->24). The standard fix
is to give each boundary element a fiber-volume-FRACTION-weighted modulus instead
of all-or-nothing. `classify_wavy_fiber_fractions` returns the per-element fiber
volume fraction phi in [0,1] (sub-sampled on a barycentric grid, only for elements
straddling a fiber surface -- interior/exterior elements are 1/0 for free), and
`composite_lame_fields` mixes the Lame parameters by phi. `solve_homogenization`
accepts a precomputed (lambda, mu) field via `lame_fields`.

Run:  python -m pytest src/tests/test_rve_composite.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from skfem.models.elasticity import lame_parameters

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_mesh import build_background_mesh, classify_wavy_fiber_fractions  # noqa: E402
from src.uq.rve_solve import composite_lame_fields, solve_homogenization  # noqa: E402
from src.uq.rve_wavy_geometry import generate_wavy_fiber_geometry  # noqa: E402

MATRIX = (3.0, 0.35)
INCLUSION = (1000.0, 0.2)
UNIAXIAL = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


class TestClassifyWavyFiberFractions:
    def _geom(self):
        return generate_wavy_fiber_geometry(
            target_vf=0.12, domain_size=10.0, rng=np.random.default_rng(0),
            waviness_ratio=0.2, n_waves=2, n_samples=32,
        )

    def test_fractions_in_unit_interval(self):
        mesh = build_background_mesh(10.0, 16)
        phi = classify_wavy_fiber_fractions(mesh, self._geom())
        assert phi.shape == (mesh.t.shape[1],)
        assert phi.min() >= 0.0
        assert phi.max() <= 1.0

    def test_has_interior_and_boundary_and_exterior(self):
        mesh = build_background_mesh(10.0, 24)
        phi = classify_wavy_fiber_fractions(mesh, self._geom())
        assert (phi == 0.0).any()                       # exterior matrix
        assert ((phi > 0.0) & (phi < 1.0)).any()        # partial (boundary) elements
        assert phi.max() > 0.8                          # some element nearly full fiber

    def test_mean_fraction_tracks_volume_fraction(self):
        mesh = build_background_mesh(10.0, 20)
        geom = self._geom()
        phi = classify_wavy_fiber_fractions(mesh, geom)
        assert 0.4 * geom.achieved_vf < phi.mean() < 2.5 * geom.achieved_vf

    def test_deterministic(self):
        mesh = build_background_mesh(10.0, 14)
        geom = self._geom()
        np.testing.assert_array_equal(
            classify_wavy_fiber_fractions(mesh, geom),
            classify_wavy_fiber_fractions(mesh, geom),
        )


class TestCompositeLameFields:
    def test_pure_phases_match_endpoints(self):
        phi = np.array([0.0, 1.0])
        lam, mu = composite_lame_fields(phi, MATRIX, INCLUSION)
        lam_m, mu_m = lame_parameters(*MATRIX)
        lam_i, mu_i = lame_parameters(*INCLUSION)
        assert lam[0, 0] == pytest.approx(lam_m)
        assert mu[0, 0] == pytest.approx(mu_m)
        assert lam[1, 0] == pytest.approx(lam_i)
        assert mu[1, 0] == pytest.approx(mu_i)

    def test_half_fraction_is_average(self):
        lam, mu = composite_lame_fields(np.array([0.5]), MATRIX, INCLUSION)
        lam_m, mu_m = lame_parameters(*MATRIX)
        lam_i, mu_i = lame_parameters(*INCLUSION)
        assert lam[0, 0] == pytest.approx(0.5 * (lam_m + lam_i))
        assert mu[0, 0] == pytest.approx(0.5 * (mu_m + mu_i))

    def test_shape(self):
        lam, mu = composite_lame_fields(np.array([0.1, 0.9, 0.5]), MATRIX, INCLUSION)
        assert lam.shape == (3, 1)
        assert mu.shape == (3, 1)


class TestSolveWithLameOverride:
    def test_single_phase_recovers_isotropic(self):
        mesh = build_background_mesh(10.0, 4)
        n_el = mesh.t.shape[1]
        lam_m, mu_m = lame_parameters(*MATRIX)
        lame_fields = (np.full((n_el, 1), lam_m), np.full((n_el, 1), mu_m))
        result = solve_homogenization(
            mesh, None, 10.0, MATRIX, INCLUSION, UNIAXIAL, lame_fields=lame_fields
        )
        e11 = UNIAXIAL[0, 0]
        expected = np.array([
            [(lam_m + 2 * mu_m) * e11, 0.0, 0.0],
            [0.0, lam_m * e11, 0.0],
            [0.0, 0.0, lam_m * e11],
        ])
        np.testing.assert_allclose(result.macro_stress, expected, atol=1e-6)
