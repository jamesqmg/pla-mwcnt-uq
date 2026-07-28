"""Tests for src/uq/rve_stiffness.py -- RED phase (fail before implementation).

Full anisotropic homogenization: run the six unit Voigt-strain load cases
through solve_homogenization, assemble the 6x6 stiffness matrix C, and derive
directional engineering constants. Correctness anchor: a single-phase isotropic
domain must reproduce the textbook isotropic stiffness matrix exactly
(C11 = lam + 2mu, C12 = lam, C44 = mu) and give E_axial = E_transverse = E.

The aligned continuous-fiber case is the physics check the whole redesign turns
on: stiff fibers along the axis must make the axial modulus exceed the
transverse one (transverse isotropy), bounded by the constituent moduli.

Run:  python -m pytest src/tests/test_rve_stiffness.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from skfem.models.elasticity import lame_parameters

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_continuous_geometry import generate_continuous_fiber_geometry  # noqa: E402
from src.uq.rve_mesh import (  # noqa: E402
    INCLUSION_PHASE,
    build_background_mesh,
    classify_continuous_fiber_phase,
)
from src.uq.rve_stiffness import engineering_constants, full_stiffness_tensor  # noqa: E402

E_MATRIX, NU_MATRIX = 3.0, 0.35
E_INCLUSION, NU_INCLUSION = 1000.0, 0.2


def _isotropic_voigt_stiffness(E: float, nu: float) -> np.ndarray:
    lam, mu = lame_parameters(E, nu)
    C = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            C[i, j] = lam
        C[i, i] = lam + 2 * mu
    for k in range(3, 6):
        C[k, k] = mu
    return C


class TestFullStiffnessTensor:
    def test_single_phase_matches_isotropic_analytic(self):
        mesh = build_background_mesh(10.0, 4)
        tags = np.zeros(mesh.t.shape[1], dtype=int)  # all matrix
        C = full_stiffness_tensor(
            mesh, tags, 10.0, (E_MATRIX, NU_MATRIX), (E_INCLUSION, NU_INCLUSION)
        )
        np.testing.assert_allclose(C, _isotropic_voigt_stiffness(E_MATRIX, NU_MATRIX), atol=1e-6)

    def test_shape_is_6x6(self):
        mesh = build_background_mesh(10.0, 4)
        tags = np.zeros(mesh.t.shape[1], dtype=int)
        C = full_stiffness_tensor(
            mesh, tags, 10.0, (E_MATRIX, NU_MATRIX), (E_INCLUSION, NU_INCLUSION)
        )
        assert C.shape == (6, 6)

    def test_aligned_fibers_are_transversely_isotropic_and_bounded(self):
        # One 6-solve extraction, several assertions on it (kept small for speed).
        domain, axis = 10.0, 2
        mesh = build_background_mesh(domain, 10)
        geom = generate_continuous_fiber_geometry(
            target_vf=0.12, domain_size=domain, rng=np.random.default_rng(0), axis=axis
        )
        tags = classify_continuous_fiber_phase(mesh, geom)
        assert (tags == INCLUSION_PHASE).any(), "fixture must contain fiber elements"

        C = full_stiffness_tensor(
            mesh, tags, domain, (E_MATRIX, NU_MATRIX), (E_INCLUSION, NU_INCLUSION)
        )
        # Stiffness matrix must be symmetric.
        np.testing.assert_allclose(C, C.T, atol=1e-6 * np.abs(C).max())

        e_axial, e_transverse = engineering_constants(C, axial=axis)
        # Aligned stiff fibers: axial modulus exceeds transverse (the signature
        # of the continuous-fiber redesign), and both stay between the two
        # constituent moduli.
        assert e_axial > e_transverse
        assert e_axial > E_MATRIX
        assert e_transverse >= E_MATRIX * 0.99
        assert e_axial < E_INCLUSION


class TestEngineeringConstants:
    def test_isotropic_recovers_youngs_modulus(self):
        C = _isotropic_voigt_stiffness(E_MATRIX, NU_MATRIX)
        e_axial, e_transverse = engineering_constants(C, axial=2)
        assert e_axial == pytest.approx(E_MATRIX, rel=1e-9)
        assert e_transverse == pytest.approx(E_MATRIX, rel=1e-9)

    def test_axial_index_selection(self):
        # Hand-built transversely-isotropic compliance with a distinct axial
        # entry on axis 0: engineering_constants must read E from the right index.
        S = np.diag([1 / 200.0, 1 / 5.0, 1 / 5.0, 1.0, 1.0, 1.0])
        C = np.linalg.inv(S)
        e_axial, e_transverse = engineering_constants(C, axial=0)
        assert e_axial == pytest.approx(200.0, rel=1e-6)
        assert e_transverse == pytest.approx(5.0, rel=1e-6)
