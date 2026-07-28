"""
Tests for src/uq/rve_solve.py — RED phase (all should fail before implementation)

Tier 3 / Phase 4 Track B4: periodic-BC linear elasticity solve for the
FE-RVE homogenization. Builds on rve_geometry.py (inclusion placement) and
rve_mesh.py (background mesh + phase tagging). Validates the per-element
Lame-parameter field construction, the DOF-elimination periodic constraint
assembly, and the solve itself -- anchored by an exact analytical check:
a single-phase (homogeneous) domain under imposed macro strain must
recover the imposed strain EXACTLY (zero fluctuation field) and the
textbook isotropic stress-strain relation, since there is no heterogeneity
to require any deviation.

Run:  python -m pytest src/tests/test_rve_solve.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_geometry import generate_rve_geometry  # noqa: E402
from src.uq.rve_mesh import INCLUSION_PHASE, build_background_mesh, classify_element_phase  # noqa: E402
from src.uq.rve_solve import (  # noqa: E402
    assemble_periodic_constraints,
    phase_lame_fields,
    solve_homogenization,
)

E_MATRIX, NU_MATRIX = 3.0, 0.35
E_INCLUSION, NU_INCLUSION = 1000.0, 0.2
UNIAXIAL_STRAIN = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


class TestPhaseLameFields:
    def test_output_shapes(self):
        tags = np.array([0, 1, 0, 1, 1])
        lam, mu = phase_lame_fields(tags, (E_MATRIX, NU_MATRIX), (E_INCLUSION, NU_INCLUSION))
        assert lam.shape == (5, 1)
        assert mu.shape == (5, 1)

    def test_matrix_and_inclusion_elements_get_different_values(self):
        tags = np.array([0, 1])
        lam, mu = phase_lame_fields(tags, (E_MATRIX, NU_MATRIX), (E_INCLUSION, NU_INCLUSION))
        assert lam[0, 0] != pytest.approx(lam[1, 0])
        assert mu[0, 0] != pytest.approx(mu[1, 0])

    def test_all_matrix_when_no_inclusion_elements(self):
        tags = np.zeros(4, dtype=int)
        lam, mu = phase_lame_fields(tags, (E_MATRIX, NU_MATRIX), (E_INCLUSION, NU_INCLUSION))
        assert np.all(lam == lam[0])
        assert np.all(mu == mu[0])

    def test_nonpositive_modulus_raises(self):
        with pytest.raises(ValueError):
            phase_lame_fields(np.array([0]), (0.0, 0.35), (E_INCLUSION, NU_INCLUSION))

    def test_poisson_ratio_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            phase_lame_fields(np.array([0]), (E_MATRIX, 0.5), (E_INCLUSION, NU_INCLUSION))
        with pytest.raises(ValueError):
            phase_lame_fields(np.array([0]), (E_MATRIX, -1.0), (E_INCLUSION, NU_INCLUSION))


class TestAssemblePeriodicConstraints:
    def test_expansion_matrix_shape(self):
        import skfem
        mesh = build_background_mesh(10.0, 4)
        basis = skfem.Basis(mesh, skfem.ElementVector(skfem.ElementTetP1()))
        T, b = assemble_periodic_constraints(mesh, basis, 10.0, UNIAXIAL_STRAIN)
        assert T.shape[0] == basis.N
        assert b.shape == (basis.N,)
        # one reference node (origin) is pinned out entirely
        assert T.shape[1] == basis.N - 3 - 3 * int(np.isclose(mesh.p, 10.0).any(axis=0).sum())

    def test_non_square_macro_strain_raises(self):
        import skfem
        mesh = build_background_mesh(10.0, 4)
        basis = skfem.Basis(mesh, skfem.ElementVector(skfem.ElementTetP1()))
        with pytest.raises(ValueError):
            assemble_periodic_constraints(mesh, basis, 10.0, np.zeros((2, 2)))

    def test_asymmetric_macro_strain_raises(self):
        import skfem
        mesh = build_background_mesh(10.0, 4)
        basis = skfem.Basis(mesh, skfem.ElementVector(skfem.ElementTetP1()))
        asym = np.array([[0.0, 0.01, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            assemble_periodic_constraints(mesh, basis, 10.0, asym)


class TestSolveHomogenizationHomogeneousCase:
    """Single-phase sanity check: exact recovery of the imposed strain field."""

    def _solve(self, domain_size=10.0, n_side=4):
        mesh = build_background_mesh(domain_size, n_side)
        tags = np.zeros(mesh.t.shape[1], dtype=int)  # all matrix, no inclusions
        return solve_homogenization(
            mesh, tags, domain_size,
            matrix_props=(E_MATRIX, NU_MATRIX),
            inclusion_props=(E_INCLUSION, NU_INCLUSION),  # irrelevant: no inclusion elements
            macro_strain=UNIAXIAL_STRAIN,
        )

    def test_zero_fluctuation(self):
        result = self._solve()
        assert result.max_fluctuation == pytest.approx(0.0, abs=1e-8)

    def test_recovers_isotropic_stress_strain_relation(self):
        result = self._solve()
        lam, mu = 2.5925925925925917, 1.111111111111111  # lame_parameters(3.0, 0.35)
        e11 = UNIAXIAL_STRAIN[0, 0]
        expected = np.array([
            [(lam + 2 * mu) * e11, 0.0, 0.0],
            [0.0, lam * e11, 0.0],
            [0.0, 0.0, lam * e11],
        ])
        np.testing.assert_allclose(result.macro_stress, expected, atol=1e-6)

    def test_deterministic(self):
        r1, r2 = self._solve(), self._solve()
        np.testing.assert_allclose(r1.macro_stress, r2.macro_stress)


class TestSolveHomogenizationHeterogeneousCase:
    def _solve(self, seed=0, target_vf=0.05, domain_size=15.0, n_side=12):
        geom = generate_rve_geometry(
            target_vf=target_vf, aspect_ratio=8.0, domain_size=domain_size,
            rng=np.random.default_rng(seed),
        )
        mesh = build_background_mesh(domain_size, n_side)
        tags = classify_element_phase(mesh, geom)
        assert (tags == INCLUSION_PHASE).any(), "test fixture must actually contain inclusion elements"
        return solve_homogenization(
            mesh, tags, domain_size,
            matrix_props=(E_MATRIX, NU_MATRIX),
            inclusion_props=(E_INCLUSION, NU_INCLUSION),
            macro_strain=UNIAXIAL_STRAIN,
        )

    def test_fluctuation_is_nonzero(self):
        result = self._solve()
        assert result.max_fluctuation > 1e-4

    def test_stiff_filler_increases_effective_modulus(self):
        result = self._solve()
        e11 = UNIAXIAL_STRAIN[0, 0]
        e_eff = result.macro_stress[0, 0] / e11
        assert e_eff > E_MATRIX

    def test_effective_modulus_stays_below_rule_of_mixtures_bound(self):
        # Sanity ceiling: even a perfectly-aligned continuous-fiber upper
        # bound (Voigt/rule of mixtures) caps the achievable stiffening at
        # this volume fraction -- a much higher result would signal a bug.
        result = self._solve()
        e11 = UNIAXIAL_STRAIN[0, 0]
        e_eff = result.macro_stress[0, 0] / e11
        rom_upper_bound = 0.95 * E_MATRIX + 0.05 * E_INCLUSION
        assert e_eff < rom_upper_bound
