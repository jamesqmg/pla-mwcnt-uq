"""Tests for classify_continuous_fiber_phase in src/uq/rve_mesh.py -- RED phase.

Continuous-through-fiber tagging: an element is INCLUSION if its transverse-plane
centroid falls within a fiber radius of any (periodically-imaged) fiber center.
Anchored by tying the tagged element fraction back to the geometry's achieved
area fraction, so the tagger and the geometry can't silently disagree.

Run:  python -m pytest src/tests/test_rve_continuous_mesh.py -v
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_continuous_geometry import (  # noqa: E402
    ContinuousFiberGeometry,
    generate_continuous_fiber_geometry,
)
from src.uq.rve_mesh import (  # noqa: E402
    INCLUSION_PHASE,
    MATRIX_PHASE,
    build_background_mesh,
    classify_continuous_fiber_phase,
)


class TestClassifyContinuousFiberPhase:
    def test_all_matrix_when_no_fibers(self):
        mesh = build_background_mesh(10.0, 8)
        geom = ContinuousFiberGeometry(
            domain_size=10.0, fiber_centers=np.empty((0, 2)), axis=2,
            radius=0.5, target_vf=0.0, achieved_vf=0.0,
        )
        tags = classify_continuous_fiber_phase(mesh, geom)
        assert tags.shape == (mesh.t.shape[1],)
        assert np.all(tags == MATRIX_PHASE)

    def test_some_elements_tagged_inclusion(self):
        mesh = build_background_mesh(10.0, 40)
        geom = generate_continuous_fiber_geometry(
            target_vf=0.10, domain_size=10.0, rng=np.random.default_rng(0)
        )
        tags = classify_continuous_fiber_phase(mesh, geom)
        assert (tags == INCLUSION_PHASE).any()
        assert (tags == MATRIX_PHASE).any()

    def test_tagged_fraction_tracks_area_fraction(self):
        # Voxel/staircase error at finite resolution means this won't be exact,
        # but it must be the right order of magnitude -- within a factor of two.
        mesh = build_background_mesh(10.0, 40)
        geom = generate_continuous_fiber_geometry(
            target_vf=0.10, domain_size=10.0, rng=np.random.default_rng(1)
        )
        tags = classify_continuous_fiber_phase(mesh, geom)
        frac = float((tags == INCLUSION_PHASE).mean())
        assert 0.5 * geom.achieved_vf < frac < 2.0 * geom.achieved_vf

    def test_element_at_fiber_center_is_inclusion(self):
        mesh = build_background_mesh(10.0, 10)
        centroids = mesh.p[:, mesh.t].mean(axis=1)  # (3, n_elements)
        # Pick a real element centroid, put a fiber exactly at its transverse
        # coordinate (axis=2 -> transverse axes 0,1).
        e0 = 0
        center_xy = centroids[[0, 1], e0]
        geom = ContinuousFiberGeometry(
            domain_size=10.0, fiber_centers=center_xy.reshape(1, 2), axis=2,
            radius=0.5, target_vf=0.0, achieved_vf=0.0,
        )
        tags = classify_continuous_fiber_phase(mesh, geom)
        assert tags[e0] == INCLUSION_PHASE

    def test_periodic_wrap_tags_opposite_face(self):
        # A fiber centered on the low transverse face (u=0) must, by periodicity,
        # tag elements near the high transverse face (u=domain).
        mesh = build_background_mesh(10.0, 40)
        centroids = mesh.p[:, mesh.t].mean(axis=1)
        v_mid = 5.0
        geom = ContinuousFiberGeometry(
            domain_size=10.0, fiber_centers=np.array([[0.0, v_mid]]), axis=2,
            radius=0.5, target_vf=0.0, achieved_vf=0.0,
        )
        tags = classify_continuous_fiber_phase(mesh, geom)
        u_centroid = centroids[0, :]
        tagged_high_face = (tags == INCLUSION_PHASE) & (u_centroid > 9.0)
        assert tagged_high_face.any()

    def test_axis_zero_orientation(self):
        # Fibers along x (axis=0): transverse plane is (y, z).
        mesh = build_background_mesh(10.0, 30)
        geom = generate_continuous_fiber_geometry(
            target_vf=0.08, domain_size=10.0, rng=np.random.default_rng(2), axis=0
        )
        tags = classify_continuous_fiber_phase(mesh, geom)
        assert (tags == INCLUSION_PHASE).any()

    def test_deterministic(self):
        mesh = build_background_mesh(10.0, 24)
        geom = generate_continuous_fiber_geometry(
            target_vf=0.10, domain_size=10.0, rng=np.random.default_rng(3)
        )
        t1 = classify_continuous_fiber_phase(mesh, geom)
        t2 = classify_continuous_fiber_phase(mesh, geom)
        np.testing.assert_array_equal(t1, t2)
