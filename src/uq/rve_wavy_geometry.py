"""Wavy continuous through-fiber geometry for Tier-3 FE-RVE homogenization.

Extends the straight aligned-fiber model (rve_continuous_geometry.py) with
sinusoidal waviness: each fiber still threads the full RVE along a fixed axis and
wraps periodically, but its centerline is displaced transversely by a sinusoid
of amplitude a and wavelength lambda. The dimensionless waviness ratio

    w = a / lambda

is the standard sinusoidal-fiber waviness parameter (Berhan, Yi & Sastry 2004,
J. Appl. Phys. 95(9):5027; Shao et al. 2009, Compos. Struct. 87(3):274). Holding
w = 0 recovers a straight fiber. The centerline is sampled as a piecewise-linear
polyline -- the same piecewise-segment construction Doh et al. (2020, Carbon 172)
use for explicit wavy CNTs -- so the segment-distance primitives already in
rve_geometry.py can be reused for overlap and voxel tagging.

Periodicity: choosing lambda = domain_size / n_waves (integer n_waves) makes the
transverse displacement close on itself across the axial boundary, so a wavy
fiber is genuinely periodic in all three directions.

Volume accounting: a wavy fiber of contour length L_c > L (axial length) sweeps a
tube of volume ~ pi r^2 L_c (thin-tube approximation; exact only when the
curvature radius stays well above r, which bounds the usable waviness range at
fixed wavelength). To isolate tortuosity from filler volume in a
waviness-knockdown study, the fiber count is set from the contour length so the
volume fraction is held fixed as w varies -- hence contour_length is a
first-class quantity here. The classifier's tagged element fraction is the
ground-truth volume fraction used by the solve; this estimate only sets targets.

This module (representation, sampler, contour length) is step 1a; placement and
tagging follow in later steps.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from src.uq.rve_geometry import closest_points_segment_segment


@dataclass(frozen=True)
class WavyFiber:
    """One wavy continuous fiber: a sinusoidal centerline along `axis`.

    `base_center` is the transverse-plane position the sinusoid oscillates about;
    `wave_dir` is the (unit) transverse direction of oscillation; `phase` offsets
    the sinusoid. All lengths are in units of fiber diameter d (radius = 0.5).
    """

    base_center: np.ndarray  # (2,) transverse-plane coordinates
    wave_dir: np.ndarray     # (2,) unit vector in the transverse plane
    phase: float
    amplitude: float         # a, in diameters
    wavelength: float        # lambda, in diameters
    axis: int = 2
    radius: float = 0.5


def wavelength_for_n_waves(domain_size: float, n_waves: int) -> float:
    """Wavelength that fits `n_waves` full periods in the domain (axial periodicity)."""
    if n_waves < 1:
        raise ValueError("n_waves must be a positive integer.")
    return domain_size / n_waves


def amplitude_for_ratio(waviness_ratio: float, wavelength: float) -> float:
    """Sinusoid amplitude a = w * lambda from the waviness ratio w."""
    if waviness_ratio < 0.0:
        raise ValueError("waviness_ratio must be non-negative.")
    return waviness_ratio * wavelength


def _transverse_axes(axis: int) -> list[int]:
    return [a for a in range(3) if a != axis]


def fiber_polyline(fiber: WavyFiber, domain_size: float, n_samples: int) -> np.ndarray:
    """Sample the fiber centerline as an (n_samples + 1, 3) polyline over [0, domain_size].

    The axial coordinate runs uniformly from 0 to domain_size along `fiber.axis`;
    the two transverse coordinates follow base_center + a*sin(2*pi*z/lambda + phase)
    along `fiber.wave_dir`.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    z = np.linspace(0.0, domain_size, n_samples + 1)
    offset = fiber.amplitude * np.sin(2.0 * np.pi * z / fiber.wavelength + fiber.phase)
    transverse = fiber.base_center[None, :] + offset[:, None] * fiber.wave_dir[None, :]

    pts = np.zeros((n_samples + 1, 3))
    ta = _transverse_axes(fiber.axis)
    pts[:, ta[0]] = transverse[:, 0]
    pts[:, ta[1]] = transverse[:, 1]
    pts[:, fiber.axis] = z
    return pts


def contour_length(fiber: WavyFiber, domain_size: float, n_samples: int) -> float:
    """Arc length of the sampled centerline polyline (>= domain_size; grows with w)."""
    pts = fiber_polyline(fiber, domain_size, n_samples)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


@dataclass(frozen=True)
class WavyFiberGeometry:
    """A generated periodic RVE of wavy continuous through-fibers.

    `fibers` carry the sinusoid parameters; re-sample with `fiber_polyline` at
    `n_samples` to recover the polylines used for placement and tagging.
    `achieved_vf` is the thin-tube geometry estimate (n_fibers * pi r^2 L_c /
    V); the classifier's tagged element fraction is the ground-truth volume
    fraction used by the solve.
    """

    domain_size: float
    fibers: tuple[WavyFiber, ...]
    axis: int
    radius: float
    waviness_ratio: float
    n_waves: int
    n_samples: int
    target_vf: float
    achieved_vf: float


def _fibers_overlap(
    pts_p: np.ndarray,
    pts_q: np.ndarray,
    domain_size: float,
    transverse_axes: list[int],
    radius: float,
    band: int,
) -> bool:
    """True if two sampled fiber tubes come within 2*radius anywhere (periodic).

    Compares segments under the 9 transverse periodic images of `pts_q`. Because
    both polylines are sampled at identical axial stations, only segment pairs
    whose axial indices are within `band` can be close enough to touch (segments
    farther apart in the axial index are separated in z by more than 2*radius);
    the axial periodic wrap needs no separate image because a wavy fiber is
    axially periodic, so the boundary configuration is already present in the
    interior. Early-exits on the first contact.
    """
    min_gap = 2.0 * radius
    n_seg = len(pts_p) - 1
    for su, sv in itertools.product((-1, 0, 1), repeat=2):
        q = pts_q.copy()
        q[:, transverse_axes[0]] += su * domain_size
        q[:, transverse_axes[1]] += sv * domain_size
        for k in range(n_seg):
            p1, p2 = pts_p[k], pts_p[k + 1]
            m_lo = max(0, k - band)
            m_hi = min(n_seg, k + band + 1)
            for m in range(m_lo, m_hi):
                if closest_points_segment_segment(p1, p2, q[m], q[m + 1]) < min_gap:
                    return True
    return False


def generate_wavy_fiber_geometry(
    target_vf: float,
    domain_size: float,
    rng: np.random.Generator,
    waviness_ratio: float,
    n_waves: int,
    radius: float = 0.5,
    axis: int = 2,
    n_samples: int = 48,
    max_attempts_per_fiber: int = 5000,
) -> WavyFiberGeometry:
    """Place non-overlapping wavy continuous fibers, holding volume fraction ~fixed as w varies.

    The fiber count is set from the contour length so that the (thin-tube)
    volume fraction stays near `target_vf` regardless of waviness ratio -- this
    is what lets a waviness-knockdown study vary tortuosity at fixed filler
    volume. Placement is periodic 2D base-RSA with a broad-phase base-distance
    prune and a segment-level narrow-phase overlap test (`_fibers_overlap`).
    Raises RuntimeError if a fiber cannot be placed within the attempt budget.
    """
    if target_vf < 0.0:
        raise ValueError("target_vf must be non-negative.")
    if domain_size <= 0.0:
        raise ValueError("domain_size must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}.")
    if waviness_ratio < 0.0:
        raise ValueError("waviness_ratio must be non-negative.")

    wavelength = wavelength_for_n_waves(domain_size, n_waves)
    amplitude = amplitude_for_ratio(waviness_ratio, wavelength)
    transverse_axes = _transverse_axes(axis)

    reference = WavyFiber(
        base_center=np.zeros(2), wave_dir=np.array([1.0, 0.0]), phase=0.0,
        amplitude=amplitude, wavelength=wavelength, axis=axis, radius=radius,
    )
    l_c = contour_length(reference, domain_size, n_samples)
    fiber_volume = np.pi * radius**2 * l_c
    domain_volume = domain_size**3
    n_target = round(target_vf * domain_volume / fiber_volume)

    seg_len = domain_size / n_samples
    band = int(np.ceil(2.0 * radius / seg_len)) + 1
    broad_reach = 2.0 * amplitude + 2.0 * radius

    fibers: list[WavyFiber] = []
    polylines: list[np.ndarray] = []
    for _ in range(n_target):
        placed = False
        for _attempt in range(max_attempts_per_fiber):
            base = rng.uniform(0.0, domain_size, size=2)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            wave_dir = np.array([np.cos(angle), np.sin(angle)])
            phase = rng.uniform(0.0, 2.0 * np.pi)
            candidate = WavyFiber(
                base_center=base, wave_dir=wave_dir, phase=phase,
                amplitude=amplitude, wavelength=wavelength, axis=axis, radius=radius,
            )
            candidate_poly = fiber_polyline(candidate, domain_size, n_samples)

            ok = True
            for other, other_poly in zip(fibers, polylines):
                delta = np.abs(base - other.base_center)
                delta = np.minimum(delta, domain_size - delta)
                if np.hypot(delta[0], delta[1]) > broad_reach:
                    continue  # footprints too far apart to touch
                if _fibers_overlap(other_poly, candidate_poly, domain_size, transverse_axes, radius, band):
                    ok = False
                    break
            if ok:
                fibers.append(candidate)
                polylines.append(candidate_poly)
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"Could not place wavy fiber {len(fibers) + 1}/{n_target} after "
                f"{max_attempts_per_fiber} attempts (target_vf={target_vf}, "
                f"domain_size={domain_size}, waviness_ratio={waviness_ratio}). "
                "Packing is likely infeasible -- try a larger domain or a lower target_vf."
            )

    achieved_vf = len(fibers) * fiber_volume / domain_volume
    return WavyFiberGeometry(
        domain_size=domain_size,
        fibers=tuple(fibers),
        axis=axis,
        radius=radius,
        waviness_ratio=waviness_ratio,
        n_waves=n_waves,
        n_samples=n_samples,
        target_vf=target_vf,
        achieved_vf=achieved_vf,
    )
