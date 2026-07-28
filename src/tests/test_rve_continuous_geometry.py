"""Tests for src/uq/rve_continuous_geometry.py -- RED phase (fail before implementation).

Tier 3 / Phase 4 continuous-through-fiber redesign. Models aligned MWCNTs as
continuous fibers threading the full RVE, placed as non-overlapping disks in the
transverse plane via periodic 2D Random Sequential Addition. This sidesteps the
finite-rod RSA jamming wall (rve_geometry.py) at realistic nanotube slenderness.

The overlap check here is computed independently of the module under test (a
direct minimum-image pairwise distance) so the non-overlap assertion isn't a
tautology against the module's own distance function.

Run:  python -m pytest src/tests/test_rve_continuous_geometry.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_continuous_geometry import (  # noqa: E402
    ContinuousFiberGeometry,
    generate_continuous_fiber_geometry,
    min_image_disk_distance,
)

RADIUS = 0.5


def _all_pairs_min_image_ok(centers: np.ndarray, domain_size: float, radius: float) -> bool:
    """Independent overlap check: every pair of periodic disks is at least 2r apart."""
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.abs(centers[i] - centers[j])
            d = np.minimum(d, domain_size - d)  # minimum image per transverse axis
            if np.hypot(d[0], d[1]) < 2 * radius - 1e-9:
                return False
    return True


class TestMinImageDiskDistance:
    def test_direct_distance_no_wrap(self):
        c1 = np.array([2.0, 2.0])
        c2 = np.array([5.0, 6.0])
        assert min_image_disk_distance(c1, c2, 100.0) == pytest.approx(5.0)

    def test_wrap_is_shorter(self):
        # Across a domain of size 10, points at 0.5 and 9.5 are 1.0 apart via wrap,
        # not 9.0 apart directly.
        c1 = np.array([0.5, 3.0])
        c2 = np.array([9.5, 3.0])
        assert min_image_disk_distance(c1, c2, 10.0) == pytest.approx(1.0)

    def test_symmetric(self):
        c1 = np.array([1.0, 7.0])
        c2 = np.array([8.0, 2.0])
        assert min_image_disk_distance(c1, c2, 10.0) == pytest.approx(
            min_image_disk_distance(c2, c1, 10.0)
        )


class TestGenerateContinuousFiberGeometry:
    def test_returns_geometry_dataclass(self):
        geom = generate_continuous_fiber_geometry(
            target_vf=0.05, domain_size=20.0, rng=np.random.default_rng(0)
        )
        assert isinstance(geom, ContinuousFiberGeometry)
        assert geom.domain_size == 20.0
        assert geom.radius == RADIUS
        assert geom.axis == 2

    def test_centers_shape_and_in_domain(self):
        geom = generate_continuous_fiber_geometry(
            target_vf=0.08, domain_size=20.0, rng=np.random.default_rng(1)
        )
        assert geom.fiber_centers.ndim == 2
        assert geom.fiber_centers.shape[1] == 2
        assert geom.fiber_centers.shape[0] >= 1
        assert (geom.fiber_centers >= 0.0).all()
        assert (geom.fiber_centers <= 20.0).all()

    def test_no_overlaps(self):
        geom = generate_continuous_fiber_geometry(
            target_vf=0.10, domain_size=20.0, rng=np.random.default_rng(2)
        )
        assert _all_pairs_min_image_ok(geom.fiber_centers, geom.domain_size, geom.radius)

    def test_achieved_vf_matches_area_fraction(self):
        geom = generate_continuous_fiber_geometry(
            target_vf=0.10, domain_size=20.0, rng=np.random.default_rng(3)
        )
        expected = len(geom.fiber_centers) * np.pi * RADIUS**2 / 20.0**2
        assert geom.achieved_vf == pytest.approx(expected)

    def test_achieved_vf_close_to_target(self):
        # Discretisation is one fiber's worth of area; achieved should be within that.
        target = 0.10
        geom = generate_continuous_fiber_geometry(
            target_vf=target, domain_size=25.0, rng=np.random.default_rng(4)
        )
        one_fiber_vf = np.pi * RADIUS**2 / 25.0**2
        assert abs(geom.achieved_vf - target) <= one_fiber_vf + 1e-12

    def test_deterministic_for_same_seed(self):
        g1 = generate_continuous_fiber_geometry(
            target_vf=0.08, domain_size=20.0, rng=np.random.default_rng(7)
        )
        g2 = generate_continuous_fiber_geometry(
            target_vf=0.08, domain_size=20.0, rng=np.random.default_rng(7)
        )
        np.testing.assert_array_equal(g1.fiber_centers, g2.fiber_centers)

    def test_zero_target_vf_gives_no_fibers(self):
        geom = generate_continuous_fiber_geometry(
            target_vf=0.0, domain_size=20.0, rng=np.random.default_rng(0)
        )
        assert geom.fiber_centers.shape == (0, 2)
        assert geom.achieved_vf == 0.0

    def test_axis_parameter_respected(self):
        geom = generate_continuous_fiber_geometry(
            target_vf=0.05, domain_size=20.0, rng=np.random.default_rng(0), axis=0
        )
        assert geom.axis == 0

    def test_higher_vf_packs_more_fibers(self):
        low = generate_continuous_fiber_geometry(
            target_vf=0.05, domain_size=25.0, rng=np.random.default_rng(5)
        )
        high = generate_continuous_fiber_geometry(
            target_vf=0.15, domain_size=25.0, rng=np.random.default_rng(5)
        )
        assert len(high.fiber_centers) > len(low.fiber_centers)


class TestGenerateContinuousFiberValidation:
    def test_negative_vf_raises(self):
        with pytest.raises(ValueError):
            generate_continuous_fiber_geometry(-0.01, 20.0, np.random.default_rng(0))

    def test_vf_above_packing_ceiling_raises(self):
        with pytest.raises(ValueError):
            generate_continuous_fiber_geometry(0.75, 20.0, np.random.default_rng(0))

    def test_nonpositive_domain_raises(self):
        with pytest.raises(ValueError):
            generate_continuous_fiber_geometry(0.05, 0.0, np.random.default_rng(0))

    def test_nonpositive_radius_raises(self):
        with pytest.raises(ValueError):
            generate_continuous_fiber_geometry(0.05, 20.0, np.random.default_rng(0), radius=0.0)

    def test_bad_axis_raises(self):
        with pytest.raises(ValueError):
            generate_continuous_fiber_geometry(0.05, 20.0, np.random.default_rng(0), axis=3)
