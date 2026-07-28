"""Tests for the interphase third phase (Phase 2) -- RED phase.

Adds a coaxial interphase shell (modified-modulus polymer) around each fiber:
Phase.INTERPHASE tags for the shell r < d <= r + t, a 3-way phase_lame_fields,
and interphase_props threaded through the solve and the stiffness extractor.
All additions are backward-compatible: interphase_thickness=0 / interphase_props=None
must reproduce the existing two-phase behavior exactly, which the plumbing test
below checks by setting interphase props equal to the matrix.

Run:  python -m pytest src/tests/test_rve_interphase.py -v
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
    INTERPHASE_PHASE,
    MATRIX_PHASE,
    build_background_mesh,
    classify_continuous_fiber_phase,
    classify_wavy_fiber_phase,
)
from src.uq.rve_solve import phase_lame_fields  # noqa: E402
from src.uq.rve_stiffness import engineering_constants, full_stiffness_tensor  # noqa: E402
from src.uq.rve_wavy_geometry import generate_wavy_fiber_geometry  # noqa: E402

MATRIX = (3.0, 0.35)
INCLUSION = (1000.0, 0.2)
INTERPHASE = (30.0, 0.3)


class TestPhaseLameFieldsThreePhase:
    def test_interphase_elements_get_interphase_values(self):
        tags = np.array([MATRIX_PHASE, INCLUSION_PHASE, INTERPHASE_PHASE])
        lam, mu = phase_lame_fields(tags, MATRIX, INCLUSION, interphase_props=INTERPHASE)
        lam_p, mu_p = lame_parameters(*INTERPHASE)
        assert lam[2, 0] == pytest.approx(lam_p)
        assert mu[2, 0] == pytest.approx(mu_p)

    def test_three_distinct_values(self):
        tags = np.array([MATRIX_PHASE, INCLUSION_PHASE, INTERPHASE_PHASE])
        lam, mu = phase_lame_fields(tags, MATRIX, INCLUSION, interphase_props=INTERPHASE)
        assert len({round(float(x), 6) for x in mu[:, 0]}) == 3

    def test_interphase_tag_without_props_raises(self):
        tags = np.array([MATRIX_PHASE, INTERPHASE_PHASE])
        with pytest.raises(ValueError):
            phase_lame_fields(tags, MATRIX, INCLUSION, interphase_props=None)

    def test_two_phase_unchanged_when_no_interphase(self):
        tags = np.array([MATRIX_PHASE, INCLUSION_PHASE, MATRIX_PHASE])
        lam, mu = phase_lame_fields(tags, MATRIX, INCLUSION)  # no interphase arg
        lam_m, mu_m = lame_parameters(*MATRIX)
        lam_i, mu_i = lame_parameters(*INCLUSION)
        assert mu[0, 0] == pytest.approx(mu_m)
        assert mu[1, 0] == pytest.approx(mu_i)


class TestContinuousClassifierInterphase:
    def _setup(self):
        mesh = build_background_mesh(10.0, 40)
        geom = generate_continuous_fiber_geometry(
            target_vf=0.10, domain_size=10.0, rng=np.random.default_rng(0), axis=2
        )
        return mesh, geom

    def test_no_interphase_by_default(self):
        mesh, geom = self._setup()
        tags = classify_continuous_fiber_phase(mesh, geom)
        assert not (tags == INTERPHASE_PHASE).any()

    def test_shell_tagged_and_core_unchanged(self):
        mesh, geom = self._setup()
        tags0 = classify_continuous_fiber_phase(mesh, geom)
        tagsi = classify_continuous_fiber_phase(mesh, geom, interphase_thickness=0.4)
        assert (tagsi == INTERPHASE_PHASE).any()
        # fiber cores identical
        assert (tags0 == INCLUSION_PHASE).sum() == (tagsi == INCLUSION_PHASE).sum()
        # interphase elements were matrix without the shell
        shell = tagsi == INTERPHASE_PHASE
        assert np.all(tags0[shell] == MATRIX_PHASE)


class TestWavyClassifierInterphase:
    def test_shell_tagged(self):
        mesh = build_background_mesh(10.0, 30)
        geom = generate_wavy_fiber_geometry(
            target_vf=0.10, domain_size=10.0, rng=np.random.default_rng(1),
            waviness_ratio=0.2, n_waves=2, n_samples=24,
        )
        tags0 = classify_wavy_fiber_phase(mesh, geom)
        tagsi = classify_wavy_fiber_phase(mesh, geom, interphase_thickness=0.4)
        assert not (tags0 == INTERPHASE_PHASE).any()
        assert (tagsi == INTERPHASE_PHASE).any()
        assert (tags0 == INCLUSION_PHASE).sum() == (tagsi == INCLUSION_PHASE).sum()


class TestStiffnessWithInterphase:
    def _geom_tags(self, thickness):
        mesh = build_background_mesh(8.0, 12)
        geom = generate_continuous_fiber_geometry(
            target_vf=0.12, domain_size=8.0, rng=np.random.default_rng(7), axis=2
        )
        return mesh, classify_continuous_fiber_phase(mesh, geom, interphase_thickness=thickness)

    def test_interphase_equal_to_matrix_reproduces_two_phase(self):
        mesh, tags0 = self._geom_tags(0.0)
        c_two = full_stiffness_tensor(mesh, tags0, 8.0, MATRIX, INCLUSION)
        mesh_i, tags_i = self._geom_tags(0.5)
        c_three = full_stiffness_tensor(mesh_i, tags_i, 8.0, MATRIX, INCLUSION, interphase_props=MATRIX)
        np.testing.assert_allclose(c_three, c_two, rtol=1e-9, atol=1e-9)

    def test_stiff_interphase_raises_axial_modulus(self):
        mesh, tags0 = self._geom_tags(0.0)
        c_two = full_stiffness_tensor(mesh, tags0, 8.0, MATRIX, INCLUSION)
        mesh_i, tags_i = self._geom_tags(0.5)
        c_stiff = full_stiffness_tensor(mesh_i, tags_i, 8.0, MATRIX, INCLUSION, interphase_props=(100.0, 0.3))
        assert engineering_constants(c_stiff, axial=2)[0] > engineering_constants(c_two, axial=2)[0]
