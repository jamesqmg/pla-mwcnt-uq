"""RVE geometry generator for Tier-3 FE-RVE homogenization (Phase 4 / Track B4).

Generates a periodic representative volume element (RVE): a cubic domain
containing randomly placed, randomly oriented, non-overlapping MWCNT
inclusions modelled as capsules (a cylindrical centerline segment swept by
a fixed radius), at a target volume fraction. Geometry only -- meshing and
the FE solve are separate modules.

All lengths are non-dimensionalised in units of CNT diameter d (so radius
= 0.5 and `aspect_ratio` directly gives each inclusion's centerline length).
Real diameter/length/purity come from the Tier-4 procurement spec, not from
generic literature values -- callers must supply `aspect_ratio` explicitly.

Volume accounting uses the plain-cylinder volume (pi r^2 * length), not the
full capsule volume (cylinder + hemispherical caps): at the aspect ratios
this module targets the cap contribution is a rounding error, and the caps
exist only so the overlap test (`closest_points_segment_segment`) has a
simple, robust well-defined minimum distance between two rods.

Placement uses periodic Random Sequential Addition (RSA): each inclusion is
tried against the 27 periodic images (`min_image_capsule_distance`) of every
already-placed inclusion, so overlaps that wrap across a domain face are
still caught. Placement can fail to converge -- this raises RuntimeError
rather than silently under-filling the domain, because elongated particles
have large excluded volume and jam at low nominal volume fractions.

Known open scope question (do not silently resolve): literature MWCNT
aspect ratios (length/diameter ~ 500-3000+) are far too extreme for
tractable explicit-fiber RSA packing at any reasonable domain size -- the
excluded volume of a single rod scales with aspect_ratio^2, so realistic
nanotube slenderness jams almost immediately. Before running this at
real values, that has to be resolved deliberately (a capped/reduced
effective aspect ratio, a continuous/periodic-fiber treatment, or
reconsidering whether explicit-fiber 3D FE is the right tool at all vs.
the mean-field Halpin-Tsai/Mori-Tanaka already used for Tier 2) -- not
picked implicitly by whatever this module happens to accept.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

_MAX_TARGET_VF = 0.7  # physical packing ceiling; RSA becomes infeasible well before this


@dataclass(frozen=True)
class Inclusion:
    """One MWCNT inclusion: centerline segment (center +/- length/2 along orientation), all in units of CNT diameter."""

    center: np.ndarray       # shape (3,)
    orientation: np.ndarray  # unit vector, shape (3,)
    length: float            # centerline length, in diameters
    radius: float = 0.5      # fixed by the diameter-normalisation


@dataclass(frozen=True)
class RVEGeometry:
    """A generated periodic RVE: the cubic domain plus its placed inclusions."""

    domain_size: float
    inclusions: tuple[Inclusion, ...]
    target_vf: float
    achieved_vf: float


def effective_volume_fraction(nominal_vf: float, purity: float = 1.0) -> float:
    """De-rate nominal filler volume fraction for MWCNT purity.

    `purity` is the mass fraction of the nominal filler that is genuine
    reinforcing MWCNT (the remainder -- amorphous carbon, catalyst residue --
    contributes negligible stiffness). The de-rating is applied as
    volume-proportional to mass-purity, which is exact only if the
    impurity-phase density is close to the MWCNT density; this is the
    standard assumption absent phase-resolved density data.
    """
    if not (0.0 < purity <= 1.0):
        raise ValueError("purity must lie in (0, 1].")
    if not (0.0 <= nominal_vf < 1.0):
        raise ValueError("nominal_vf must lie in [0, 1).")
    return nominal_vf * purity


def closest_points_segment_segment(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> float:
    """Minimum distance between finite 3D segments [p1,q1] and [p2,q2].

    Robust closest-point-between-segments algorithm (Ericson, "Real-Time
    Collision Detection"), with explicit handling of degenerate
    (zero-length) segments to avoid division by zero.
    """
    eps = 1e-12
    p1, q1, p2, q2 = (np.asarray(x, dtype=float) for x in (p1, q1, p2, q2))
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r

    if a <= eps and e <= eps:
        s, t = 0.0, 0.0
    elif a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return float(np.linalg.norm(c1 - c2))


def min_image_capsule_distance(
    c1: np.ndarray, u1: np.ndarray, l1: float,
    c2: np.ndarray, u2: np.ndarray, l2: float,
    domain_size: float,
) -> float:
    """Segment-segment distance under periodic (minimum-image) wrap.

    Checks the candidate segment against all 27 periodic images (including
    the direct, unshifted case) of the other segment and returns the
    minimum -- so an overlap that only occurs by wrapping across a domain
    face is still detected.
    """
    c1, u1, c2, u2 = (np.asarray(x, dtype=float) for x in (c1, u1, c2, u2))
    p1, q1 = c1 - u1 * (l1 / 2.0), c1 + u1 * (l1 / 2.0)
    best = np.inf
    for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3):
        shift = np.array([dx, dy, dz], dtype=float) * domain_size
        c2_shifted = c2 + shift
        p2 = c2_shifted - u2 * (l2 / 2.0)
        q2 = c2_shifted + u2 * (l2 / 2.0)
        d = closest_points_segment_segment(p1, q1, p2, q2)
        if d < best:
            best = d
    return best


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Uniform-on-the-sphere random unit vector (isotropic 3D orientation)."""
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def generate_rve_geometry(
    target_vf: float,
    aspect_ratio: float,
    domain_size: float,
    rng: np.random.Generator,
    max_attempts_per_inclusion: int = 2000,
) -> RVEGeometry:
    """Place non-overlapping capsule inclusions in a periodic cubic RVE via RSA.

    Raises RuntimeError if an inclusion cannot be placed within the attempt
    budget -- this means the requested (target_vf, aspect_ratio, domain_size)
    combination is infeasible (excluded volume too large for the domain),
    not that the RVE was silently under-filled.
    """
    if target_vf < 0.0:
        raise ValueError("target_vf must be non-negative.")
    if target_vf >= _MAX_TARGET_VF:
        raise ValueError(f"target_vf must be physically packable (< {_MAX_TARGET_VF}); got {target_vf}.")
    if aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be positive.")
    if domain_size <= 0.0:
        raise ValueError("domain_size must be positive.")

    radius = 0.5
    length = aspect_ratio  # diameter normalised to 1.0
    cyl_volume = np.pi * radius**2 * length
    domain_volume = domain_size**3
    n_target = round(target_vf * domain_volume / cyl_volume)

    inclusions: list[Inclusion] = []
    for _ in range(n_target):
        placed = False
        for _attempt in range(max_attempts_per_inclusion):
            center = rng.uniform(0.0, domain_size, size=3)
            orientation = random_unit_vector(rng)
            if all(
                min_image_capsule_distance(
                    center, orientation, length,
                    other.center, other.orientation, other.length,
                    domain_size,
                ) >= 2 * radius
                for other in inclusions
            ):
                inclusions.append(Inclusion(center=center, orientation=orientation, length=length, radius=radius))
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"Could not place inclusion {len(inclusions) + 1}/{n_target} after "
                f"{max_attempts_per_inclusion} attempts (target_vf={target_vf}, "
                f"aspect_ratio={aspect_ratio}, domain_size={domain_size}). Packing is "
                "likely infeasible at this volume fraction / aspect ratio / domain "
                "size combination -- try a larger domain, a lower target_vf, or a "
                "reduced effective aspect ratio."
            )

    achieved_vf = len(inclusions) * cyl_volume / domain_volume
    return RVEGeometry(
        domain_size=domain_size,
        inclusions=tuple(inclusions),
        target_vf=target_vf,
        achieved_vf=achieved_vf,
    )
