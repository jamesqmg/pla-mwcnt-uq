"""Tests for src/uq/rve_wavy_geometry.py -- RED phase (fail before implementation).

Wavy continuous through-fibers: each aligned fiber's centerline is a sinusoidal
transverse displacement along the axial coordinate (Berhan et al. 2004 sinusoidal
fiber model; shao_2009), sampled as a piecewise-linear polyline (doh_2020's
piecewise-segment construction) so the existing segment-distance primitives can
be reused for overlap and voxel tagging.

This sub-step (1a) covers only the representation, the sampler, and contour
length -- placement and tagging are separate steps. Holding the volume fraction
constant across waviness levels (via contour length) is what makes the eventual
waviness-knockdown study isolate tortuosity rather than filler volume, so
contour length is tested directly here.

Run:  python -m pytest src/tests/test_rve_wavy_geometry.py -v
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_geometry import closest_points_segment_segment  # noqa: E402
from src.uq.rve_wavy_geometry import (  # noqa: E402
    WavyFiber,
    WavyFiberGeometry,
    amplitude_for_ratio,
    contour_length,
    fiber_polyline,
    generate_wavy_fiber_geometry,
    wavelength_for_n_waves,
)

DOMAIN = 12.0


def _straight_fiber(amplitude=0.0, axis=2, phase=0.0):
    return WavyFiber(
        base_center=np.array([4.0, 7.0]),
        wave_dir=np.array([1.0, 0.0]),
        phase=phase,
        amplitude=amplitude,
        wavelength=DOMAIN / 3,
        axis=axis,
        radius=0.5,
    )


class TestHelpers:
    def test_wavelength_for_n_waves_divides_domain(self):
        assert wavelength_for_n_waves(DOMAIN, 3) == pytest.approx(4.0)
        assert wavelength_for_n_waves(DOMAIN, 1) == pytest.approx(DOMAIN)

    def test_wavelength_bad_n_waves_raises(self):
        with pytest.raises(ValueError):
            wavelength_for_n_waves(DOMAIN, 0)

    def test_amplitude_for_ratio(self):
        # a = w * lambda
        assert amplitude_for_ratio(0.1, 4.0) == pytest.approx(0.4)
        assert amplitude_for_ratio(0.0, 4.0) == pytest.approx(0.0)

    def test_amplitude_negative_ratio_raises(self):
        with pytest.raises(ValueError):
            amplitude_for_ratio(-0.1, 4.0)


class TestFiberPolyline:
    def test_shape_and_axial_coordinate(self):
        f = _straight_fiber(amplitude=0.5)
        pts = fiber_polyline(f, DOMAIN, n_samples=48)
        assert pts.shape == (49, 3)
        # axial coordinate runs 0..DOMAIN along the fiber axis
        np.testing.assert_allclose(pts[:, f.axis], np.linspace(0, DOMAIN, 49))

    def test_zero_amplitude_is_straight(self):
        f = _straight_fiber(amplitude=0.0)
        pts = fiber_polyline(f, DOMAIN, n_samples=32)
        transverse_axes = [a for a in range(3) if a != f.axis]
        np.testing.assert_allclose(pts[:, transverse_axes[0]], f.base_center[0])
        np.testing.assert_allclose(pts[:, transverse_axes[1]], f.base_center[1])

    def test_axial_periodicity_when_wavelength_divides_domain(self):
        # wavelength = DOMAIN/3 -> transverse position matches at z=0 and z=DOMAIN
        f = _straight_fiber(amplitude=0.4, phase=1.1)
        pts = fiber_polyline(f, DOMAIN, n_samples=60)
        transverse_axes = [a for a in range(3) if a != f.axis]
        np.testing.assert_allclose(
            pts[0, transverse_axes], pts[-1, transverse_axes], atol=1e-9
        )

    def test_amplitude_controls_transverse_excursion(self):
        f = _straight_fiber(amplitude=0.6)
        pts = fiber_polyline(f, DOMAIN, n_samples=200)
        u = pts[:, [a for a in range(3) if a != f.axis][0]]
        assert np.max(np.abs(u - f.base_center[0])) == pytest.approx(0.6, abs=2e-3)

    def test_axis_zero(self):
        f = _straight_fiber(amplitude=0.3, axis=0)
        pts = fiber_polyline(f, DOMAIN, n_samples=40)
        np.testing.assert_allclose(pts[:, 0], np.linspace(0, DOMAIN, 41))


class TestContourLength:
    def test_straight_contour_equals_axial_length(self):
        f = _straight_fiber(amplitude=0.0)
        assert contour_length(f, DOMAIN, n_samples=64) == pytest.approx(DOMAIN)

    def test_wavy_contour_exceeds_axial_length(self):
        f = _straight_fiber(amplitude=0.5)
        assert contour_length(f, DOMAIN, n_samples=200) > DOMAIN

    def test_contour_increases_with_amplitude(self):
        low = contour_length(_straight_fiber(amplitude=0.2), DOMAIN, n_samples=200)
        high = contour_length(_straight_fiber(amplitude=0.6), DOMAIN, n_samples=200)
        assert high > low

    def test_contour_converges_with_sampling(self):
        # Refining the sampling should change the length only marginally.
        coarse = contour_length(_straight_fiber(amplitude=0.4), DOMAIN, n_samples=100)
        fine = contour_length(_straight_fiber(amplitude=0.4), DOMAIN, n_samples=800)
        assert abs(fine - coarse) / fine < 0.01


def _min_fiber_pair_distance(geom: WavyFiberGeometry) -> float:
    """Independent min centerline distance across all fiber pairs (no band prune)."""
    ta = [a for a in range(3) if a != geom.axis]
    polys = [fiber_polyline(f, geom.domain_size, geom.n_samples) for f in geom.fibers]
    best = np.inf
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            for su, sv in itertools.product((-1, 0, 1), repeat=2):
                q = polys[j].copy()
                q[:, ta[0]] += su * geom.domain_size
                q[:, ta[1]] += sv * geom.domain_size
                for k in range(len(polys[i]) - 1):
                    for m in range(len(q) - 1):
                        d = closest_points_segment_segment(polys[i][k], polys[i][k + 1], q[m], q[m + 1])
                        if d < best:
                            best = d
    return float(best)


class TestGenerateWavyFiberGeometry:
    def test_returns_geometry_with_fibers(self):
        geom = generate_wavy_fiber_geometry(
            target_vf=0.10, domain_size=8.0, rng=np.random.default_rng(0),
            waviness_ratio=0.2, n_waves=2, n_samples=24,
        )
        assert isinstance(geom, WavyFiberGeometry)
        assert len(geom.fibers) >= 2
        assert geom.waviness_ratio == 0.2

    def test_no_overlap_wavy(self):
        geom = generate_wavy_fiber_geometry(
            target_vf=0.10, domain_size=8.0, rng=np.random.default_rng(1),
            waviness_ratio=0.2, n_waves=2, n_samples=24,
        )
        assert _min_fiber_pair_distance(geom) >= 2 * geom.radius - 1e-6

    def test_no_overlap_straight_limit(self):
        geom = generate_wavy_fiber_geometry(
            target_vf=0.10, domain_size=8.0, rng=np.random.default_rng(2),
            waviness_ratio=0.0, n_waves=2, n_samples=24,
        )
        assert _min_fiber_pair_distance(geom) >= 2 * geom.radius - 1e-6

    def test_deterministic(self):
        kw = dict(target_vf=0.10, domain_size=8.0, waviness_ratio=0.2, n_waves=2, n_samples=24)
        g1 = generate_wavy_fiber_geometry(rng=np.random.default_rng(5), **kw)
        g2 = generate_wavy_fiber_geometry(rng=np.random.default_rng(5), **kw)
        assert len(g1.fibers) == len(g2.fibers)
        np.testing.assert_array_equal(
            np.array([f.base_center for f in g1.fibers]),
            np.array([f.base_center for f in g2.fibers]),
        )

    def test_achieved_vf_near_target(self):
        target = 0.08
        geom = generate_wavy_fiber_geometry(
            target_vf=target, domain_size=12.0, rng=np.random.default_rng(3),
            waviness_ratio=0.15, n_waves=3, n_samples=48,
        )
        # thin-tube estimate: within ~2 fibers' worth of the target
        two_fiber_vf = 2 * np.pi * 0.5**2 * geom.domain_size / geom.domain_size**3 * 3.0
        assert abs(geom.achieved_vf - target) < two_fiber_vf

    def test_waviness_reduces_fiber_count_at_fixed_vf(self):
        # Longer contour length per fiber -> fewer fibers for the same volume.
        kw = dict(target_vf=0.10, domain_size=12.0, n_waves=3, n_samples=48)
        straight = generate_wavy_fiber_geometry(
            rng=np.random.default_rng(4), waviness_ratio=0.0, **kw
        )
        wavy = generate_wavy_fiber_geometry(
            rng=np.random.default_rng(4), waviness_ratio=0.3, **kw
        )
        assert len(wavy.fibers) <= len(straight.fibers)


class TestGenerateWavyFiberValidation:
    def test_negative_vf_raises(self):
        with pytest.raises(ValueError):
            generate_wavy_fiber_geometry(-0.01, 8.0, np.random.default_rng(0), 0.2, 2)

    def test_bad_axis_raises(self):
        with pytest.raises(ValueError):
            generate_wavy_fiber_geometry(0.1, 8.0, np.random.default_rng(0), 0.2, 2, axis=3)

    def test_negative_waviness_raises(self):
        with pytest.raises(ValueError):
            generate_wavy_fiber_geometry(0.1, 8.0, np.random.default_rng(0), -0.2, 2)

    def test_bad_n_waves_raises(self):
        with pytest.raises(ValueError):
            generate_wavy_fiber_geometry(0.1, 8.0, np.random.default_rng(0), 0.2, 0)
