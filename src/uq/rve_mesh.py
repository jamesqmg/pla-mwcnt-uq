"""Periodic background mesh + phase tagging for Tier-3 FE-RVE homogenization.

Builds a structured tensor-product tetrahedral mesh over the cubic RVE
domain and tags each element as matrix or inclusion by testing its
centroid against the periodic inclusion geometry from `rve_geometry.py`.
This is a voxel/pixel-FE approach: the phase boundary is approximated by
element membership, not conformed to exactly by the mesh. That's standard
practice for computational homogenization when a conforming CAD mesher
(e.g. gmsh around explicit capsule surfaces) isn't in the toolchain --
scikit-fem itself doesn't provide one. The tradeoff is a "staircase"
geometric error at the phase interface that shrinks as the mesh refines;
it is not a conforming boundary and should not be presented as one.

Also identifies the periodic node correspondence needed for periodic
boundary conditions: nodes on each pair of opposite domain faces are
matched exactly, which the structured tensor mesh guarantees (opposing
faces share identical coordinate grids). Corner and edge nodes are
returned separately, not folded into the face-pair lists -- a corner is
shared by all 3 face-pairs and an edge by 2, and resolving that overlap
into a consistent constraint hierarchy is the concern of the constraint
assembly in the solve step, not this module.

Performance note: `classify_element_phase` was originally an explicit
O(n_elements * n_inclusions) loop; measured at ~3.5 minutes for a single
solve at ~10K elements / 27 inclusions (see
Thesis2/fe-rve-asset-inventory.md build log), which made an MC ensemble
of solves impractical. It now uses a KD-tree over inclusion centers
(including their 26 periodic images) for broad-phase pruning: each
element centroid only runs the exact segment-distance test against
inclusions the tree says are actually within reach, not all of them. This
changes performance, not correctness -- the narrow-phase test is the same
`closest_points_segment_segment` primitive from step 1, applied to
already-shifted segment endpoints instead of re-deriving all 27 periodic
images per pair inside the hot loop.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import skfem
from scipy.spatial import cKDTree

from src.uq.rve_continuous_geometry import ContinuousFiberGeometry
from src.uq.rve_geometry import RVEGeometry, closest_points_segment_segment
from src.uq.rve_wavy_geometry import WavyFiberGeometry, fiber_polyline


class Phase(IntEnum):
    """Per-element material phase tag."""

    MATRIX = 0
    INCLUSION = 1
    INTERPHASE = 2


# Module-level aliases kept for backward compatibility with existing call
# sites (rve_solve.py, generate_fe_rve_demo_viz.py, generate_fig1/2 scripts)
# -- IntEnum members compare equal to and behave as plain ints, so no
# call-site changes are required.
MATRIX_PHASE = Phase.MATRIX
INCLUSION_PHASE = Phase.INCLUSION
INTERPHASE_PHASE = Phase.INTERPHASE


@dataclass(frozen=True)
class PeriodicNodePairs:
    """Node correspondence across the three pairs of opposite cube faces.

    `corner_nodes` and `edge_nodes` are listed separately because they
    appear in multiple face-pair arrays (a corner is on all 3 face pairs,
    an edge on 2) -- the solve step's constraint hierarchy needs to know
    that, not have it silently double-counted as independent face pairs.
    """

    x_pairs: np.ndarray
    y_pairs: np.ndarray
    z_pairs: np.ndarray
    corner_nodes: np.ndarray
    edge_nodes: np.ndarray


def build_background_mesh(domain_size: float, n_elements_per_side: int) -> skfem.MeshTet:
    """Structured tensor-product tetrahedral mesh over [0, domain_size]^3."""
    if domain_size <= 0.0:
        raise ValueError("domain_size must be positive.")
    if n_elements_per_side < 1:
        raise ValueError("n_elements_per_side must be at least 1.")
    grid = np.linspace(0.0, domain_size, n_elements_per_side + 1)
    return skfem.MeshTet.init_tensor(grid, grid, grid)


def classify_element_phase(mesh: skfem.MeshTet, geometry: RVEGeometry) -> np.ndarray:
    """Per-element phase tag (MATRIX_PHASE or INCLUSION_PHASE) by centroid membership.

    An element is INCLUSION_PHASE if its centroid lies within any
    inclusion's radius, checked under the periodic minimum-image
    convention -- an element near one domain face can be inside an
    inclusion physically placed near the opposite face and wrapped
    through, and a non-periodic check would miss that.

    Uses a KD-tree over inclusion centers (with their 26 periodic images)
    for broad-phase pruning: each element only runs the exact
    segment-distance test against inclusions actually within reach, not
    every inclusion in the RVE. See the module docstring's performance note.
    """
    centroids = mesh.p[:, mesh.t].mean(axis=1)  # shape (3, n_elements)
    n_elements = centroids.shape[1]
    tags = np.full(n_elements, MATRIX_PHASE, dtype=int)
    inclusions = geometry.inclusions
    if not inclusions:
        return tags

    domain_size = geometry.domain_size
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float) * domain_size

    image_points = np.concatenate([inc.center + shifts for inc in inclusions], axis=0)
    image_inclusion_idx = np.repeat(np.arange(len(inclusions)), len(shifts))
    max_reach = max(inc.radius + inc.length / 2.0 for inc in inclusions)

    tree = cKDTree(image_points)
    for e in range(n_elements):
        point = centroids[:, e]
        candidates = tree.query_ball_point(point, r=max_reach + 1e-9)
        for ci in candidates:
            inc = inclusions[image_inclusion_idx[ci]]
            shifted_center = image_points[ci]
            half = inc.orientation * (inc.length / 2.0)
            d = closest_points_segment_segment(point, point, shifted_center - half, shifted_center + half)
            if d <= inc.radius:
                tags[e] = INCLUSION_PHASE
                break
    return tags


def periodic_face_node_pairs(mesh: skfem.MeshTet, domain_size: float, tol: float = 1e-9) -> PeriodicNodePairs:
    """Match nodes on each pair of opposite cube faces for periodic BCs.

    Requires a structured tensor mesh (as produced by
    build_background_mesh) so opposite faces share identical (y,z) /
    (x,z) / (x,y) coordinate grids -- raises RuntimeError if a node on
    one face has no coordinate-matching counterpart on the opposite face.

    Code-review note (2026-07-09): this has zero production callers.
    rve_solve.py's `_periodic_node_map` solves the same underlying problem
    (matching a slave node to its periodic master) but takes a different,
    simpler path: it zeros ALL high coordinates of a node at once and looks
    up the resulting position directly, giving one master + one cumulative
    shift per node regardless of how many axes are wrapped. This function's
    per-axis face-pair decomposition can't do that in one step for an edge
    or corner node (wrapped on 2 or 3 axes) -- a single x_pairs or y_pairs
    entry only unwraps one axis, so reconstructing the fully-unwrapped
    master would mean chaining through multiple pair lookups, exactly the
    "constraint hierarchy" complexity this module's own docstring defers to
    the solve step. The solve step chose not to build that chain and instead
    reimplemented the equivalent logic directly. This function is not wrong
    or redundant computation -- it answers a genuine, narrower question
    (per-axis face correspondence) that no current caller needs. Kept
    (tested, correct) rather than deleted, per the "flag, don't silently
    resolve" convention this codebase uses elsewhere.
    """
    p = mesh.p

    def face_nodes(axis: int, value: float) -> np.ndarray:
        return np.nonzero(np.isclose(p[axis], value, atol=tol))[0]

    def match(axis: int) -> np.ndarray:
        lo = face_nodes(axis, 0.0)
        hi = face_nodes(axis, domain_size)
        other_axes = [a for a in range(3) if a != axis]
        hi_keys = {tuple(np.round(p[other_axes, i], 9)): i for i in hi}
        pairs = []
        for lo_i in lo:
            key = tuple(np.round(p[other_axes, lo_i], 9))
            hi_i = hi_keys.get(key)
            if hi_i is None:
                raise RuntimeError(
                    f"No matching node found on the opposite face for node {lo_i} "
                    f"(axis {axis}) -- mesh is not a structured tensor grid over "
                    f"[0, {domain_size}]^3."
                )
            pairs.append((lo_i, hi_i))
        return np.array(pairs, dtype=int)

    x_pairs, y_pairs, z_pairs = match(0), match(1), match(2)

    on_boundary = [
        np.isclose(p[a], 0.0, atol=tol) | np.isclose(p[a], domain_size, atol=tol)
        for a in range(3)
    ]
    n_boundary_axes = on_boundary[0].astype(int) + on_boundary[1].astype(int) + on_boundary[2].astype(int)
    corner_nodes = np.nonzero(n_boundary_axes == 3)[0]
    edge_nodes = np.nonzero(n_boundary_axes == 2)[0]

    return PeriodicNodePairs(
        x_pairs=x_pairs, y_pairs=y_pairs, z_pairs=z_pairs,
        corner_nodes=corner_nodes, edge_nodes=edge_nodes,
    )


def classify_continuous_fiber_phase(
    mesh: skfem.MeshTet, geometry: ContinuousFiberGeometry, interphase_thickness: float = 0.0
) -> np.ndarray:
    """Per-element phase tag for aligned continuous through-fibers.

    An element is INCLUSION_PHASE if its transverse-plane centroid lies within a
    fiber radius of any fiber center, under the periodic minimum-image
    convention in the two transverse axes (fibers wrap across transverse faces).
    With `interphase_thickness` > 0, elements in the coaxial shell
    r < d <= r + t are tagged INTERPHASE_PHASE; the default 0.0 reproduces the
    two-phase behavior exactly.
    The axial coordinate is irrelevant -- a straight fiber spans the whole
    domain along `geometry.axis`, so membership is purely a 2D transverse test.

    Like `classify_element_phase`, this is a non-conforming voxel tagger: the
    phase boundary is approximated by element membership and carries a
    staircase error that shrinks with mesh refinement. A cKDTree over the 9
    periodic images of the transverse fiber centers does the broad-phase lookup;
    the nearest image within a radius tags the element.
    """
    centroids = mesh.p[:, mesh.t].mean(axis=1)  # (3, n_elements)
    n_elements = centroids.shape[1]
    tags = np.full(n_elements, MATRIX_PHASE, dtype=int)

    centers = geometry.fiber_centers
    if len(centers) == 0:
        return tags

    transverse_axes = [a for a in range(3) if a != geometry.axis]
    points = centroids[transverse_axes, :].T  # (n_elements, 2)

    domain_size = geometry.domain_size
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=2)), dtype=float) * domain_size
    image_centers = (centers[:, None, :] + shifts[None, :, :]).reshape(-1, 2)

    tree = cKDTree(image_centers)
    dist, _ = tree.query(points, k=1)
    if interphase_thickness > 0.0:
        shell = (dist > geometry.radius) & (dist <= geometry.radius + interphase_thickness)
        tags[shell] = INTERPHASE_PHASE
    tags[dist <= geometry.radius] = INCLUSION_PHASE
    return tags


def classify_wavy_fiber_phase(
    mesh: skfem.MeshTet, geometry: WavyFiberGeometry, interphase_thickness: float = 0.0
) -> np.ndarray:
    """Per-element phase tag for wavy continuous through-fibers.

    An element is INCLUSION_PHASE if its 3D centroid lies within a fiber radius
    of any fiber's sampled centerline polyline, under the periodic transverse-
    image convention. With `interphase_thickness` > 0, elements in the coaxial
    shell r < d <= r + t are tagged INTERPHASE_PHASE; the default 0.0 reproduces
    the two-phase behavior exactly. Unlike the straight-fiber tagger this cannot reduce to a
    2D transverse test -- a wavy centerline moves through the axial coordinate --
    so it is a genuine point-to-segment test, broad-phased by a cKDTree over the
    segment midpoints (including the 9 transverse periodic images). Same
    non-conforming voxel approximation and staircase error as the other taggers.
    """
    centroids = mesh.p[:, mesh.t].mean(axis=1)  # (3, n_elements)
    n_elements = centroids.shape[1]
    tags = np.full(n_elements, MATRIX_PHASE, dtype=int)

    fibers = geometry.fibers
    if not fibers:
        return tags

    transverse_axes = [a for a in range(3) if a != geometry.axis]
    domain_size = geometry.domain_size

    seg_a: list[np.ndarray] = []
    seg_b: list[np.ndarray] = []
    for fiber in fibers:
        poly = fiber_polyline(fiber, domain_size, geometry.n_samples)  # (n+1, 3)
        for su, sv in itertools.product((-1, 0, 1), repeat=2):
            shifted = poly.copy()
            shifted[:, transverse_axes[0]] += su * domain_size
            shifted[:, transverse_axes[1]] += sv * domain_size
            seg_a.append(shifted[:-1])
            seg_b.append(shifted[1:])

    A = np.concatenate(seg_a, axis=0)  # (n_segments, 3)
    B = np.concatenate(seg_b, axis=0)
    midpoints = 0.5 * (A + B)
    half_len_max = 0.5 * float(np.linalg.norm(B - A, axis=1).max())
    reach = geometry.radius + half_len_max + interphase_thickness + 1e-9
    outer = geometry.radius + interphase_thickness

    tree = cKDTree(midpoints)
    points = centroids.T  # (n_elements, 3)
    for e in range(n_elements):
        point = points[e]
        best = np.inf
        for ci in tree.query_ball_point(point, reach):
            d = closest_points_segment_segment(point, point, A[ci], B[ci])
            if d < best:
                best = d
                if best <= geometry.radius:
                    break  # inclusion is the strongest tag; no closer classification possible
        if best <= geometry.radius:
            tags[e] = INCLUSION_PHASE
        elif interphase_thickness > 0.0 and best <= outer:
            tags[e] = INTERPHASE_PHASE
    return tags


def _barycentric_weights(order: int) -> np.ndarray:
    """Barycentric sample weights for a tet: all (i,j,k,m)/order with i+j+k+m=order."""
    pts = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            for k in range(order + 1 - i - j):
                pts.append((i, j, k, order - i - j - k))
    return np.array(pts, dtype=float) / order


def classify_wavy_fiber_fractions(
    mesh: skfem.MeshTet, geometry: WavyFiberGeometry, order: int = 3
) -> np.ndarray:
    """Per-element fiber volume fraction phi in [0, 1] (composite-voxel material assignment).

    Removes most of the binary-voxel staircase error that over-stiffens thin wavy
    fibers at feasible resolution: elements well inside a fiber are 1 and well
    outside are 0 (decided cheaply from the vectorized centroid-to-fiber distance),
    while only elements straddling a fiber surface are sub-sampled on an
    order-`order` barycentric grid of the tet to get a partial fraction. Reuses the
    same segment set + cKDTree as classify_wavy_fiber_phase.
    """
    centroids = mesh.p[:, mesh.t].mean(axis=1)  # (3, n_elements)
    n_elements = centroids.shape[1]
    phi = np.zeros(n_elements)

    fibers = geometry.fibers
    if not fibers:
        return phi

    transverse_axes = [a for a in range(3) if a != geometry.axis]
    domain_size = geometry.domain_size
    radius = geometry.radius

    seg_a: list[np.ndarray] = []
    seg_b: list[np.ndarray] = []
    for fiber in fibers:
        poly = fiber_polyline(fiber, domain_size, geometry.n_samples)
        for su, sv in itertools.product((-1, 0, 1), repeat=2):
            shifted = poly.copy()
            shifted[:, transverse_axes[0]] += su * domain_size
            shifted[:, transverse_axes[1]] += sv * domain_size
            seg_a.append(shifted[:-1])
            seg_b.append(shifted[1:])
    A = np.concatenate(seg_a, axis=0)
    B = np.concatenate(seg_b, axis=0)
    midpoints = 0.5 * (A + B)
    half_len_max = 0.5 * float(np.linalg.norm(B - A, axis=1).max())
    tree = cKDTree(midpoints)

    def _min_dist(point: np.ndarray) -> float:
        best = np.inf
        for ci in tree.query_ball_point(point, radius + half_len_max):
            d = closest_points_segment_segment(point, point, A[ci], B[ci])
            if d < best:
                best = d
        return best

    # Vectorized banding: nearest-midpoint distance bounds the true segment
    # distance to within half_len_max, so elements far enough in/out are 1/0
    # without any exact test; only the straddling band is sub-sampled.
    xs = np.unique(np.round(mesh.p[0], 9))
    cell = float(xs[1] - xs[0]) if xs.size > 1 else domain_size
    band = cell * np.sqrt(3.0)  # a tet's extent from its centroid
    d_mid, _ = tree.query(centroids.T, k=1)
    definitely_in = d_mid + half_len_max < radius - band
    definitely_out = d_mid - half_len_max > radius + band
    phi[definitely_in] = 1.0

    vertices = mesh.p[:, mesh.t]  # (3, 4, n_elements)
    weights = _barycentric_weights(order)  # (n_samples, 4)
    straddling = np.nonzero(~definitely_in & ~definitely_out)[0]
    for e in straddling:
        sample_pts = np.einsum("sv,dv->ds", weights, vertices[:, :, e])  # (3, n_samples)
        inside = sum(
            1 for s in range(sample_pts.shape[1]) if _min_dist(sample_pts[:, s]) <= radius
        )
        phi[e] = inside / sample_pts.shape[1]
    return phi
