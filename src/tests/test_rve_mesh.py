"""
Tests for src/uq/rve_mesh.py — RED phase (all should fail before implementation)

Tier 3 / Phase 4 Track B4: periodic background mesh + matrix/inclusion phase
tagging for the FE-RVE homogenization solve, built on top of the RVEGeometry
from rve_geometry.py. Validates the structured tensor mesh, voxel-style
phase classification (periodic-aware, reusing the minimum-image distance
from step 1), and the periodic face-node correspondence needed for PBC in
the solve step (corner/edge nodes are returned separately -- their
constraint hierarchy belongs to the solve step, not this one).

Run:  python -m pytest src/tests/test_rve_mesh.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_geometry import Inclusion, RVEGeometry  # noqa: E402
from src.uq.rve_mesh import (  # noqa: E402
    INCLUSION_PHASE,
    MATRIX_PHASE,
    build_background_mesh,
    classify_element_phase,
    periodic_face_node_pairs,
)


class TestBuildBackgroundMesh:
    def test_spans_the_full_domain(self):
        mesh = build_background_mesh(domain_size=10.0, n_elements_per_side=4)
        assert mesh.p.min() == pytest.approx(0.0)
        assert mesh.p.max() == pytest.approx(10.0)

    def test_node_count_matches_tensor_grid(self):
        mesh = build_background_mesh(domain_size=10.0, n_elements_per_side=4)
        assert mesh.p.shape[1] == 5 ** 3  # (n+1) grid points per side

    def test_nonpositive_domain_size_raises(self):
        with pytest.raises(ValueError):
            build_background_mesh(domain_size=0.0, n_elements_per_side=4)

    def test_zero_elements_per_side_raises(self):
        with pytest.raises(ValueError):
            build_background_mesh(domain_size=10.0, n_elements_per_side=0)


class TestClassifyElementPhase:
    def test_all_matrix_when_no_inclusions(self):
        mesh = build_background_mesh(10.0, 4)
        geom = RVEGeometry(domain_size=10.0, inclusions=(), target_vf=0.0, achieved_vf=0.0)
        tags = classify_element_phase(mesh, geom)
        assert np.all(tags == MATRIX_PHASE)

    def test_element_at_inclusion_center_is_tagged_inclusion(self):
        mesh = build_background_mesh(10.0, 6)
        inc = Inclusion(center=np.array([5.0, 5.0, 5.0]), orientation=np.array([0., 0., 1.]),
                         length=2.0, radius=2.0)
        geom = RVEGeometry(domain_size=10.0, inclusions=(inc,), target_vf=0.02, achieved_vf=0.02)
        tags = classify_element_phase(mesh, geom)
        centroids = mesh.p[:, mesh.t].mean(axis=1)
        near_center = np.argmin(np.linalg.norm(centroids - inc.center[:, None], axis=0))
        assert tags[near_center] == INCLUSION_PHASE

    def test_far_element_stays_matrix(self):
        mesh = build_background_mesh(10.0, 6)
        inc = Inclusion(center=np.array([5.0, 5.0, 5.0]), orientation=np.array([0., 0., 1.]),
                         length=1.0, radius=0.5)
        geom = RVEGeometry(domain_size=10.0, inclusions=(inc,), target_vf=0.001, achieved_vf=0.001)
        tags = classify_element_phase(mesh, geom)
        centroids = mesh.p[:, mesh.t].mean(axis=1)
        target = np.array([[0.3], [0.3], [0.3]])
        corner_element = np.argmin(np.linalg.norm(centroids - target, axis=0))
        assert tags[corner_element] == MATRIX_PHASE

    def test_periodic_wrap_is_respected(self):
        # An inclusion centered right at the x=0 face should also tag
        # elements near the opposite (x=domain_size) face as inclusion.
        mesh = build_background_mesh(10.0, 6)
        inc = Inclusion(center=np.array([0.1, 5.0, 5.0]), orientation=np.array([0., 0., 1.]),
                         length=1.0, radius=1.5)
        geom = RVEGeometry(domain_size=10.0, inclusions=(inc,), target_vf=0.01, achieved_vf=0.01)
        tags = classify_element_phase(mesh, geom)
        centroids = mesh.p[:, mesh.t].mean(axis=1)
        target = np.array([[9.9], [5.0], [5.0]])
        near_far_face = np.argmin(np.linalg.norm(centroids - target, axis=0))
        assert tags[near_far_face] == INCLUSION_PHASE


class TestPeriodicFaceNodePairs:
    def test_pair_counts_match_grid(self):
        mesh = build_background_mesh(10.0, 4)
        pairs = periodic_face_node_pairs(mesh, domain_size=10.0)
        assert len(pairs.x_pairs) == 5 * 5
        assert len(pairs.y_pairs) == 5 * 5
        assert len(pairs.z_pairs) == 5 * 5

    def test_x_pairs_differ_only_in_x_by_domain_size(self):
        mesh = build_background_mesh(10.0, 4)
        pairs = periodic_face_node_pairs(mesh, domain_size=10.0)
        for lo, hi in pairs.x_pairs:
            np.testing.assert_allclose(mesh.p[1:, lo], mesh.p[1:, hi])
            assert mesh.p[0, hi] - mesh.p[0, lo] == pytest.approx(10.0)

    def test_y_pairs_differ_only_in_y_by_domain_size(self):
        mesh = build_background_mesh(10.0, 4)
        pairs = periodic_face_node_pairs(mesh, domain_size=10.0)
        for lo, hi in pairs.y_pairs:
            np.testing.assert_allclose(mesh.p[[0, 2], lo], mesh.p[[0, 2], hi])
            assert mesh.p[1, hi] - mesh.p[1, lo] == pytest.approx(10.0)

    def test_corner_count_is_eight(self):
        mesh = build_background_mesh(10.0, 4)
        pairs = periodic_face_node_pairs(mesh, domain_size=10.0)
        assert len(pairs.corner_nodes) == 8

    def test_edge_count_matches_cube_edges(self):
        mesh = build_background_mesh(10.0, 4)
        pairs = periodic_face_node_pairs(mesh, domain_size=10.0)
        # 12 edges, each with (n_elements_per_side - 1) non-corner nodes
        assert len(pairs.edge_nodes) == 12 * (4 - 1)

    def test_corners_and_edges_are_disjoint(self):
        mesh = build_background_mesh(10.0, 4)
        pairs = periodic_face_node_pairs(mesh, domain_size=10.0)
        assert len(set(pairs.corner_nodes.tolist()) & set(pairs.edge_nodes.tolist())) == 0

    def test_non_tensor_mesh_raises(self):
        import skfem
        mesh = skfem.MeshTet()  # the default unit-cube mesh, not our tensor grid
        with pytest.raises(RuntimeError):
            periodic_face_node_pairs(mesh, domain_size=2.0)
