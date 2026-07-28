"""Continuous periodic through-fiber geometry for Tier-3 FE-RVE homogenization.

Models aligned MWCNTs as continuous fibers that thread the full RVE along a
fixed axis and wrap under periodicity -- i.e. effectively infinite-aspect-ratio
fibers. This sidesteps the RSA jamming wall that finite explicit rods hit at
realistic nanotube slenderness (excluded volume ~ aspect_ratio^2; see
rve_geometry.py's docstring on why real MWCNT slenderness jams almost
immediately for the finite-rod treatment).

Because every fiber spans the full domain length along the axis, placement is a
2D problem: pack non-overlapping fiber cross-sections (disks of radius r) in the
transverse plane via periodic 2D Random Sequential Addition, then extrude each
disk along the axis through the whole domain. The volume fraction is then
exactly the transverse-plane area fraction, and 2D disk RSA only jams well above
(~0.55-0.6 for random packing) the area fractions this study targets, so
placement is robust where 3D finite-rod RSA was fragile.

This module is straight aligned fibers only. Sinusoidal waviness (a transverse
offset field along the axis) is layered on in a later step; the continuous
treatment is what makes an honest waviness-knockdown study possible at real
slenderness, holding aspect ratio effectively infinite while varying tortuosity.

Lengths are non-dimensionalised in units of fiber diameter d (radius = 0.5),
consistent with rve_geometry.py. Pure-numpy: meshing/tagging lives in
rve_mesh.py, exactly as the finite-rod geometry keeps that separation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 2D random (RSA) disk packing jams well below the hexagonal-close-pack area
# fraction (~0.9069); this ceiling rejects requests that are physically
# unpackable up front rather than letting RSA spin to its attempt budget.
_MAX_AREA_FRACTION = 0.7


@dataclass(frozen=True)
class ContinuousFiberGeometry:
    """A generated periodic RVE of aligned continuous through-fibers.

    `fiber_centers` are 2D coordinates in the transverse plane (the two axes
    other than `axis`); each fiber runs the full domain length along `axis` and
    wraps periodically, so no axial coordinate is stored.
    """

    domain_size: float
    fiber_centers: np.ndarray  # shape (n, 2), transverse-plane coordinates
    axis: int                  # axial direction (0/1/2); centers live in the other two axes
    radius: float
    target_vf: float
    achieved_vf: float


def min_image_disk_distance(c1: np.ndarray, c2: np.ndarray, domain_size: float) -> float:
    """Minimum-image distance between two transverse-plane points on a periodic square.

    Wrapping is applied per axis (a separation larger than half the domain is
    shorter the other way around), so a disk near one face is correctly seen as
    close to a disk near the opposite face.
    """
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    d = np.abs(c1 - c2)
    d = np.minimum(d, domain_size - d)
    return float(np.hypot(d[0], d[1]))


def generate_continuous_fiber_geometry(
    target_vf: float,
    domain_size: float,
    rng: np.random.Generator,
    radius: float = 0.5,
    axis: int = 2,
    max_attempts_per_fiber: int = 5000,
) -> ContinuousFiberGeometry:
    """Place non-overlapping continuous fiber cross-sections via periodic 2D RSA.

    Raises RuntimeError if a fiber cannot be placed within the attempt budget --
    meaning the requested (target_vf, domain_size, radius) is too dense to pack,
    surfaced explicitly rather than silently under-filling the RVE (the same
    fail-loud convention as rve_geometry.py).
    """
    if target_vf < 0.0:
        raise ValueError("target_vf must be non-negative.")
    if target_vf >= _MAX_AREA_FRACTION:
        raise ValueError(
            f"target_vf must be physically packable (< {_MAX_AREA_FRACTION}); got {target_vf}."
        )
    if domain_size <= 0.0:
        raise ValueError("domain_size must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}.")

    transverse_area = domain_size**2
    fiber_area = np.pi * radius**2
    n_target = round(target_vf * transverse_area / fiber_area)

    centers: list[np.ndarray] = []
    for _ in range(n_target):
        placed = False
        for _attempt in range(max_attempts_per_fiber):
            candidate = rng.uniform(0.0, domain_size, size=2)
            if all(
                min_image_disk_distance(candidate, other, domain_size) >= 2 * radius
                for other in centers
            ):
                centers.append(candidate)
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"Could not place fiber {len(centers) + 1}/{n_target} after "
                f"{max_attempts_per_fiber} attempts (target_vf={target_vf}, "
                f"domain_size={domain_size}, radius={radius}). Transverse packing is "
                "likely infeasible at this area fraction -- try a larger domain or a "
                "lower target_vf."
            )

    fiber_centers = np.array(centers, dtype=float).reshape(-1, 2)
    achieved_vf = len(centers) * fiber_area / transverse_area
    return ContinuousFiberGeometry(
        domain_size=domain_size,
        fiber_centers=fiber_centers,
        axis=axis,
        radius=radius,
        target_vf=target_vf,
        achieved_vf=achieved_vf,
    )
