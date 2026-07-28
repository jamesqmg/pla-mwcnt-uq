"""Tests for the OOM-safe iterative (CG + AMG/ILU) solve path -- RED phase.

The direct sparse solve (spsolve/SuperLU) exhausts memory via LU fill-in at
production mesh sizes (n_side>=32, ~600k DOFs -> OOM kill; observed 2026-07-11).
A conjugate-gradient solve with an algebraic-multigrid preconditioner keeps
memory ~O(N) and is the right Krylov method for the SPD reduced stiffness.
It must reproduce the direct solve's homogenized response to solver tolerance;
these tests are that correctness anchor at small scale, so a converged result
at n=32/36 (where no direct answer exists) is trustworthy by extension.

Run:  python -m pytest src/tests/test_rve_iterative_solver.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from skfem.models.elasticity import lame_parameters

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_mesh import build_background_mesh, classify_wavy_fiber_fractions  # noqa: E402
from src.uq.rve_solve import composite_lame_fields, solve_homogenization  # noqa: E402
from src.uq.rve_stiffness import engineering_constants, full_stiffness_tensor  # noqa: E402
from src.uq.rve_wavy_geometry import generate_wavy_fiber_geometry  # noqa: E402

MATRIX = (3.0, 0.35)
INCLUSION = (1000.0, 0.2)
UNIAXIAL = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def _composite_field(mesh):
    geom = generate_wavy_fiber_geometry(
        target_vf=0.12, domain_size=10.0, rng=np.random.default_rng(0),
        waviness_ratio=0.2, n_waves=2, n_samples=32,
    )
    phi = classify_wavy_fiber_fractions(mesh, geom)
    return composite_lame_fields(phi, MATRIX, INCLUSION)


class TestIterativeMatchesDirect:
    def test_single_phase_iterative_exact(self):
        """Single-phase: the iterative path must still recover the affine field exactly."""
        mesh = build_background_mesh(10.0, 4)
        n_el = mesh.t.shape[1]
        lam_m, mu_m = lame_parameters(*MATRIX)
        lame_fields = (np.full((n_el, 1), lam_m), np.full((n_el, 1), mu_m))
        result = solve_homogenization(
            mesh, None, 10.0, MATRIX, INCLUSION, UNIAXIAL,
            lame_fields=lame_fields, solver="iterative",
        )
        e11 = UNIAXIAL[0, 0]
        expected = np.array([
            [(lam_m + 2 * mu_m) * e11, 0.0, 0.0],
            [0.0, lam_m * e11, 0.0],
            [0.0, 0.0, lam_m * e11],
        ])
        np.testing.assert_allclose(result.macro_stress, expected, atol=1e-6)

    def test_iterative_matches_direct_composite(self):
        """High-contrast composite: iterative macro-stress == direct to CG tolerance."""
        mesh = build_background_mesh(10.0, 12)
        lame = _composite_field(mesh)
        direct = solve_homogenization(
            mesh, None, 10.0, MATRIX, INCLUSION, UNIAXIAL, lame_fields=lame, solver="direct",
        )
        iterative = solve_homogenization(
            mesh, None, 10.0, MATRIX, INCLUSION, UNIAXIAL, lame_fields=lame, solver="iterative",
        )
        np.testing.assert_allclose(
            iterative.macro_stress, direct.macro_stress, rtol=1e-5, atol=1e-7,
        )

    def test_full_stiffness_and_moduli_match(self):
        """The 6x6 C and the extracted E1/E2 must agree across solvers (the E1 anchor)."""
        mesh = build_background_mesh(10.0, 10)
        lame = _composite_field(mesh)
        c_direct = full_stiffness_tensor(
            mesh, None, 10.0, MATRIX, INCLUSION, lame_fields=lame, solver="direct",
        )
        c_iter = full_stiffness_tensor(
            mesh, None, 10.0, MATRIX, INCLUSION, lame_fields=lame, solver="iterative",
        )
        np.testing.assert_allclose(c_iter, c_direct, rtol=1e-5, atol=1e-6)
        e1_d, e2_d = engineering_constants(c_direct, axial=2)
        e1_i, e2_i = engineering_constants(c_iter, axial=2)
        assert abs(e1_i - e1_d) / e1_d < 1e-4
        assert abs(e2_i - e2_d) / e2_d < 1e-4

    def test_batched_matches_percase_direct(self):
        """The batched assembly (solve_macro_strains) must equal 6 independent solves."""
        from src.uq.rve_solve import solve_homogenization
        from src.uq.rve_stiffness import _stress_to_voigt, _voigt_unit_strains
        mesh = build_background_mesh(10.0, 8)
        lame = _composite_field(mesh)
        c_ref = np.zeros((6, 6))
        for k, eps in enumerate(_voigt_unit_strains()):
            r = solve_homogenization(
                mesh, None, 10.0, MATRIX, INCLUSION, eps, lame_fields=lame, solver="direct",
            )
            c_ref[:, k] = _stress_to_voigt(r.macro_stress)
        c_batched = full_stiffness_tensor(
            mesh, None, 10.0, MATRIX, INCLUSION, lame_fields=lame, solver="direct",
        )
        np.testing.assert_allclose(c_batched, c_ref, rtol=1e-9, atol=1e-10)

    def test_unknown_solver_raises(self):
        mesh = build_background_mesh(10.0, 4)
        n_el = mesh.t.shape[1]
        lame = composite_lame_fields(np.zeros(n_el), MATRIX, INCLUSION)
        with pytest.raises(ValueError):
            solve_homogenization(
                mesh, None, 10.0, MATRIX, INCLUSION, UNIAXIAL,
                lame_fields=lame, solver="bogus",
            )
