"""
Tests for src/uq/rve_geometry.py — RED phase (all should fail before implementation)

Tier 3 / Phase 4 Track B4: periodic RVE geometry generator for the FE-RVE
homogenization solve. Validates the purity de-rating of nominal volume
fraction, the segment-segment / periodic-minimum-image distance
primitives used for overlap testing, and the RSA inclusion-placement
generator itself.

Run:  python -m pytest src/tests/test_rve_geometry.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_geometry import (  # noqa: E402
    closest_points_segment_segment,
    effective_volume_fraction,
    generate_rve_geometry,
    min_image_capsule_distance,
    random_unit_vector,
)


class TestEffectiveVolumeFraction:
    """Purity de-rating: V_f_eff = V_f_nominal * purity (mass-purity approximated as volume-proportional)."""

    def test_full_purity_is_identity(self):
        assert effective_volume_fraction(0.02, purity=1.0) == pytest.approx(0.02)

    def test_derates_proportionally(self):
        assert effective_volume_fraction(0.02, purity=0.95) == pytest.approx(0.019)

    def test_zero_purity_raises(self):
        with pytest.raises(ValueError):
            effective_volume_fraction(0.02, purity=0.0)

    def test_purity_above_one_raises(self):
        with pytest.raises(ValueError):
            effective_volume_fraction(0.02, purity=1.01)

    def test_negative_vf_raises(self):
        with pytest.raises(ValueError):
            effective_volume_fraction(-0.01, purity=1.0)

    def test_vf_of_one_raises(self):
        with pytest.raises(ValueError):
            effective_volume_fraction(1.0, purity=1.0)


class TestClosestPointsSegmentSegment:
    """Minimum distance between two finite line segments in 3D (Ericson's robust algorithm)."""

    def test_parallel_segments_perpendicular_offset(self):
        p1, q1 = np.array([0., 0., 0.]), np.array([1., 0., 0.])
        p2, q2 = np.array([0., 1., 0.]), np.array([1., 1., 0.])
        assert closest_points_segment_segment(p1, q1, p2, q2) == pytest.approx(1.0)

    def test_intersecting_segments_zero_distance(self):
        p1, q1 = np.array([-1., 0., 0.]), np.array([1., 0., 0.])
        p2, q2 = np.array([0., -1., 0.]), np.array([0., 1., 0.])
        assert closest_points_segment_segment(p1, q1, p2, q2) == pytest.approx(0.0, abs=1e-9)

    def test_skew_segments(self):
        p1, q1 = np.array([0., 0., 0.]), np.array([1., 0., 0.])
        p2, q2 = np.array([0., 0., 1.]), np.array([0., 1., 1.])
        assert closest_points_segment_segment(p1, q1, p2, q2) == pytest.approx(1.0)

    def test_endpoint_to_endpoint_gap(self):
        p1, q1 = np.array([0., 0., 0.]), np.array([1., 0., 0.])
        p2, q2 = np.array([2., 0., 0.]), np.array([3., 0., 0.])
        assert closest_points_segment_segment(p1, q1, p2, q2) == pytest.approx(1.0)

    def test_degenerate_point_segments(self):
        # Both "segments" collapse to points -- must not divide by zero.
        p1 = q1 = np.array([0., 0., 0.])
        p2 = q2 = np.array([3., 4., 0.])
        assert closest_points_segment_segment(p1, q1, p2, q2) == pytest.approx(5.0)

    def test_one_degenerate_one_real_segment(self):
        p1 = q1 = np.array([0., 0., 0.])   # a point
        p2, q2 = np.array([1., -1., 0.]), np.array([1., 1., 0.])
        assert closest_points_segment_segment(p1, q1, p2, q2) == pytest.approx(1.0)


class TestMinImageCapsuleDistance:
    """Segment-segment distance under periodic (minimum-image) wrap."""

    def test_direct_distance_when_no_wrap_needed(self):
        c1, c2 = np.array([1., 5., 5.]), np.array([3., 5., 5.])
        u = np.array([0., 0., 1.])
        d = min_image_capsule_distance(c1, u, 0.0, c2, u, 0.0, domain_size=10.0)
        assert d == pytest.approx(2.0)

    def test_wrap_gives_shorter_distance_than_direct(self):
        # Two point-inclusions near opposite faces of a size-10 periodic box
        # are actually close through the wrap, though far apart directly.
        c1, c2 = np.array([0.5, 5., 5.]), np.array([9.5, 5., 5.])
        u = np.array([0., 0., 1.])
        d = min_image_capsule_distance(c1, u, 0.0, c2, u, 0.0, domain_size=10.0)
        assert d == pytest.approx(1.0)

    def test_symmetric_in_argument_order(self):
        c1, c2 = np.array([0.5, 5., 5.]), np.array([9.5, 5., 5.])
        u = np.array([0., 0., 1.])
        d_ab = min_image_capsule_distance(c1, u, 0.0, c2, u, 0.0, domain_size=10.0)
        d_ba = min_image_capsule_distance(c2, u, 0.0, c1, u, 0.0, domain_size=10.0)
        assert d_ab == pytest.approx(d_ba)


class TestRandomUnitVector:
    def test_has_unit_norm(self):
        v = random_unit_vector(np.random.default_rng(0))
        assert np.linalg.norm(v) == pytest.approx(1.0)

    def test_deterministic_given_same_seed(self):
        v1 = random_unit_vector(np.random.default_rng(5))
        v2 = random_unit_vector(np.random.default_rng(5))
        np.testing.assert_allclose(v1, v2)


class TestGenerateRveGeometry:
    """RSA placement of non-overlapping capsule inclusions at a target V_f.

    Test-scale aspect ratios (~8) are used throughout -- NOT literature MWCNT
    aspect ratios (500-3000+), which are far too extreme for tractable explicit
    dispersed-fiber RSA packing. That gap is a real, unresolved modeling
    decision (see the module docstring), not a test-data shortcut.
    """

    def test_achieves_target_volume_fraction_within_tolerance(self):
        rng = np.random.default_rng(0)
        geom = generate_rve_geometry(target_vf=0.02, aspect_ratio=8.0, domain_size=15.0, rng=rng)
        assert geom.achieved_vf == pytest.approx(geom.target_vf, rel=0.15)

    def test_no_two_inclusions_overlap(self):
        rng = np.random.default_rng(1)
        geom = generate_rve_geometry(target_vf=0.02, aspect_ratio=8.0, domain_size=15.0, rng=rng)
        for i, a in enumerate(geom.inclusions):
            for b in geom.inclusions[i + 1:]:
                d = min_image_capsule_distance(
                    a.center, a.orientation, a.length,
                    b.center, b.orientation, b.length,
                    geom.domain_size,
                )
                assert d >= 2 * a.radius - 1e-6

    def test_deterministic_given_same_seed(self):
        geom_a = generate_rve_geometry(0.02, 8.0, 15.0, np.random.default_rng(42))
        geom_b = generate_rve_geometry(0.02, 8.0, 15.0, np.random.default_rng(42))
        assert geom_a.achieved_vf == pytest.approx(geom_b.achieved_vf)
        assert len(geom_a.inclusions) == len(geom_b.inclusions)
        for a, b in zip(geom_a.inclusions, geom_b.inclusions):
            np.testing.assert_allclose(a.center, b.center)
            np.testing.assert_allclose(a.orientation, b.orientation)

    def test_different_seeds_give_different_placement(self):
        geom_a = generate_rve_geometry(0.02, 8.0, 15.0, np.random.default_rng(1))
        geom_b = generate_rve_geometry(0.02, 8.0, 15.0, np.random.default_rng(2))
        assert not np.allclose(geom_a.inclusions[0].center, geom_b.inclusions[0].center)

    def test_inclusion_centers_lie_within_domain(self):
        rng = np.random.default_rng(3)
        geom = generate_rve_geometry(0.02, 8.0, 15.0, rng)
        for inc in geom.inclusions:
            assert np.all(inc.center >= 0.0) and np.all(inc.center < geom.domain_size)

    def test_zero_target_vf_gives_no_inclusions(self):
        geom = generate_rve_geometry(0.0, 8.0, 15.0, np.random.default_rng(0))
        assert len(geom.inclusions) == 0
        assert geom.achieved_vf == pytest.approx(0.0)

    def test_negative_target_vf_raises(self):
        with pytest.raises(ValueError):
            generate_rve_geometry(-0.01, 8.0, 15.0, np.random.default_rng(0))

    def test_target_vf_too_high_raises(self):
        with pytest.raises(ValueError):
            generate_rve_geometry(0.9, 8.0, 15.0, np.random.default_rng(0))

    def test_nonpositive_aspect_ratio_raises(self):
        with pytest.raises(ValueError):
            generate_rve_geometry(0.02, 0.0, 15.0, np.random.default_rng(0))

    def test_nonpositive_domain_size_raises(self):
        with pytest.raises(ValueError):
            generate_rve_geometry(0.02, 8.0, 0.0, np.random.default_rng(0))

    def test_infeasible_packing_raises_runtime_error(self):
        # Near-point inclusions (tiny aspect ratio) in a domain barely larger
        # than the exclusion radius: the theoretical maximum periodic
        # separation (domain_size * sqrt(3)/2 ~= 0.87) is less than the
        # required clearance (2*radius = 1.0), so no RNG draw can ever place
        # a second inclusion -- deterministic failure, not a flaky one.
        with pytest.raises(RuntimeError):
            generate_rve_geometry(
                target_vf=0.02, aspect_ratio=0.01, domain_size=1.0,
                rng=np.random.default_rng(0), max_attempts_per_inclusion=20,
            )
