"""Tests for classify_wavy_fiber_phase in src/uq/rve_mesh.py -- RED phase.

Voxel tagging for wavy continuous fibers: an element is INCLUSION if its 3D
centroid lies within a fiber radius of any fiber's sampled centerline polyline,
under the transverse periodic-image convention. Anchored by tying the tagged
element fraction back to the geometry's estimated volume fraction.

Run:  python -m pytest src/tests/test_rve_wavy_mesh.py -v
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_mesh import (  # noqa: E402
    INCLUSION_PHASE,
    MATRIX_PHASE,
    build_background_mesh,
    classify_wavy_fiber_phase,
)
from src.uq.rve_wavy_geometry import (  # noqa: E402
    WavyFiber,
    WavyFiberGeometry,
    generate_wavy_fiber_geometry,
)


class TestClassifyWavyFiberPhase:
    def test_all_matrix_when_no_fibers(self):
        mesh = build_background_mesh(10.0, 8)
        geom = WavyFiberGeometry(
            domain_size=10.0, fibers=(), axis=2, radius=0.5,
            waviness_ratio=0.0, n_waves=1, n_samples=24, target_vf=0.0, achieved_vf=0.0,
        )
        tags = classify_wavy_fiber_phase(mesh, geom)
        assert tags.shape == (mesh.t.shape[1],)
        assert np.all(tags == MATRIX_PHASE)

    def test_some_elements_tagged(self):
        mesh = build_background_mesh(10.0, 36)
        geom = generate_wavy_fiber_geometry(
            target_vf=0.12, domain_size=10.0, rng=np.random.default_rng(0),
            waviness_ratio=0.2, n_waves=2, n_samples=32,
        )
        tags = classify_wavy_fiber_phase(mesh, geom)
        assert (tags == INCLUSION_PHASE).any()
        assert (tags == MATRIX_PHASE).any()

    def test_tagged_fraction_tracks_estimated_vf(self):
        # thin-tube estimate vs voxel truth: same order of magnitude
        mesh = build_background_mesh(10.0, 36)
        geom = generate_wavy_fiber_geometry(
            target_vf=0.12, domain_size=10.0, rng=np.random.default_rng(1),
            waviness_ratio=0.2, n_waves=2, n_samples=32,
        )
        tags = classify_wavy_fiber_phase(mesh, geom)
        frac = float((tags == INCLUSION_PHASE).mean())
        assert 0.4 * geom.achieved_vf < frac < 2.5 * geom.achieved_vf

    def test_periodic_wrap_tags_opposite_face(self):
        # Straight (w=0) fiber on the low transverse face must tag the high face.
        mesh = build_background_mesh(10.0, 36)
        centroids = mesh.p[:, mesh.t].mean(axis=1)
        fiber = WavyFiber(
            base_center=np.array([0.0, 5.0]), wave_dir=np.array([1.0, 0.0]), phase=0.0,
            amplitude=0.0, wavelength=10.0, axis=2, radius=0.5,
        )
        geom = WavyFiberGeometry(
            domain_size=10.0, fibers=(fiber,), axis=2, radius=0.5,
            waviness_ratio=0.0, n_waves=1, n_samples=32, target_vf=0.0, achieved_vf=0.0,
        )
        tags = classify_wavy_fiber_phase(mesh, geom)
        u_centroid = centroids[0, :]
        assert ((tags == INCLUSION_PHASE) & (u_centroid > 9.0)).any()

    def test_deterministic(self):
        mesh = build_background_mesh(10.0, 24)
        geom = generate_wavy_fiber_geometry(
            target_vf=0.10, domain_size=10.0, rng=np.random.default_rng(3),
            waviness_ratio=0.25, n_waves=2, n_samples=32,
        )
        t1 = classify_wavy_fiber_phase(mesh, geom)
        t2 = classify_wavy_fiber_phase(mesh, geom)
        np.testing.assert_array_equal(t1, t2)
