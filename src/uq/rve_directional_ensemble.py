"""Monte Carlo ensemble of directional (anisotropic) FE-RVE homogenization.

Production counterpart to rve_ensemble.py -- which is the SUPERSEDED demo path
(finite explicit capsules at a fake effective aspect ratio, reduced to a single
scalar effective modulus via one uniaxial solve). This module runs the
continuous straight-through-fiber pipeline (rve_continuous_geometry -> rve_mesh
-> rve_stiffness) with the batched iterative solver, and returns the full
transversely-isotropic engineering moduli -- axial E1 and transverse E2 -- for
each realization.

Each realization draws a fresh random transverse fiber packing at the same
target volume fraction (independent, reproducible RNG streams spawned from
base_seed via SeedSequence). The spread of (E1, E2) across realizations is the
Monte Carlo uncertainty band used by Fig 3 (directional moduli vs Vf) and Fig 4
(per-Vf distribution). The demo pipeline is left untouched; this is built
alongside it.

Failures are not caught or retried: if a target Vf cannot be packed, the
RuntimeError from generate_continuous_fiber_geometry propagates -- silently
retrying would bias the ensemble toward easy-to-pack draws (same fail-loud
convention as rve_ensemble.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.uq.rve_continuous_geometry import generate_continuous_fiber_geometry
from src.uq.rve_mesh import build_background_mesh, classify_continuous_fiber_phase
from src.uq.rve_solve import MaterialProps
from src.uq.rve_stiffness import engineering_constants, full_stiffness_tensor


@dataclass(frozen=True)
class DirectionalEnsembleConfig:
    """Parameters fixed across every realization of one directional MC ensemble.

    Bundled into one object (not loose positional args) for the same
    silent-transposition safety reason as RVEEnsembleConfig -- several fields
    share a type. `solver` is forwarded to the stiffness extraction: "iterative"
    (default, OOM-safe CG+AMG) for production, "direct" for small cross-checks.
    """

    domain_size: float
    n_elements_per_side: int
    matrix_props: MaterialProps
    inclusion_props: MaterialProps
    axis: int = 2
    radius: float = 0.5
    interphase_thickness: float = 0.0
    interphase_props: MaterialProps | None = None
    solver: str = "iterative"


@dataclass(frozen=True)
class DirectionalEnsembleResult:
    """Per-realization directional moduli from one MC ensemble at a fixed Vf."""

    e_axial: np.ndarray       # (n_realizations,) E1 along the fiber axis
    e_transverse: np.ndarray  # (n_realizations,) E2 averaged over the two transverse axes
    achieved_vf: np.ndarray   # (n_realizations,) realized area/volume fraction


def run_directional_mc_ensemble(
    target_vf: float,
    config: DirectionalEnsembleConfig,
    n_realizations: int,
    base_seed: int,
) -> DirectionalEnsembleResult:
    """Directional moduli (E1, E2) for each of n_realizations random packings.

    Reproducible: the same base_seed always yields the same ensemble; different
    base_seeds give independent draws.
    """
    if n_realizations < 1:
        raise ValueError("n_realizations must be at least 1.")

    child_seeds = np.random.SeedSequence(base_seed).spawn(n_realizations)
    e_axial = np.empty(n_realizations)
    e_transverse = np.empty(n_realizations)
    achieved_vf = np.empty(n_realizations)
    for i, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed)
        geometry = generate_continuous_fiber_geometry(
            target_vf, config.domain_size, rng, radius=config.radius, axis=config.axis,
        )
        mesh = build_background_mesh(config.domain_size, config.n_elements_per_side)
        element_phase = classify_continuous_fiber_phase(
            mesh, geometry, interphase_thickness=config.interphase_thickness,
        )
        C = full_stiffness_tensor(
            mesh, element_phase, config.domain_size,
            config.matrix_props, config.inclusion_props,
            interphase_props=config.interphase_props, solver=config.solver,
        )
        e1, e2 = engineering_constants(C, axial=config.axis)
        e_axial[i] = e1
        e_transverse[i] = e2
        achieved_vf[i] = geometry.achieved_vf
    return DirectionalEnsembleResult(
        e_axial=e_axial, e_transverse=e_transverse, achieved_vf=achieved_vf,
    )
