"""Monte Carlo ensemble wrapper for Tier-3 FE-RVE homogenization.

Wraps the geometry -> mesh -> solve pipeline (rve_geometry.py,
rve_mesh.py, rve_solve.py) into repeated independent realizations at a
fixed target volume fraction. Each realization draws a fresh random RVE
(different inclusion placement/orientation) and solves for the effective
modulus under the same imposed macro strain; the resulting distribution
across realizations is the actual Monte Carlo uncertainty wrap -- this is
what Fig 4 (MC ensemble distribution) and the modulus-vs-Vf band in Fig 3
are built from, not a single point-solve.

Failures are not caught or retried: if `generate_rve_geometry` can't pack
a given realization (RSA infeasible at this target_vf / aspect_ratio /
domain_size), that propagates as a RuntimeError. Silently retrying with a
new seed would bias the ensemble toward "easy to pack" draws and hide a
real signal (that this combination is near its packing limit) -- see
rve_geometry.py's own docstring on this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.uq.rve_geometry import generate_rve_geometry
from src.uq.rve_mesh import build_background_mesh, classify_element_phase
from src.uq.rve_solve import MaterialProps, solve_homogenization


@dataclass(frozen=True)
class RVEEnsembleConfig:
    """Parameters shared across every realization in one MC ensemble
    (everything that stays fixed across a wt% sweep except target_vf).

    Bundled into one object rather than passed as separate positional
    arguments to `run_mc_ensemble` -- several of those (aspect_ratio,
    domain_size, both MaterialProps tuples) share a type, which made the
    previous 9-positional-argument signature a real silent-transposition
    risk (code review, 2026-07-09).
    """

    aspect_ratio: float
    domain_size: float
    n_elements_per_side: int
    matrix_props: MaterialProps
    inclusion_props: MaterialProps
    macro_strain: np.ndarray


def run_mc_ensemble(
    target_vf: float,
    config: RVEEnsembleConfig,
    n_realizations: int,
    base_seed: int,
) -> np.ndarray:
    """Effective modulus (macro_stress[0,0] / macro_strain[0,0]) for each realization.

    Realizations use independent, reproducible RNG streams spawned from
    `base_seed` via `numpy.random.SeedSequence` -- the same base_seed
    always reproduces the same ensemble, and different base_seeds give
    genuinely independent draws.
    """
    if n_realizations < 1:
        raise ValueError("n_realizations must be at least 1.")
    macro_strain = np.asarray(config.macro_strain, dtype=float)
    e11 = macro_strain[0, 0]
    if e11 == 0.0:
        raise ValueError("macro_strain[0, 0] must be nonzero to compute an effective modulus ratio.")

    child_seeds = np.random.SeedSequence(base_seed).spawn(n_realizations)
    effective_moduli = np.empty(n_realizations)
    for i, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed)
        geometry = generate_rve_geometry(target_vf, config.aspect_ratio, config.domain_size, rng)
        mesh = build_background_mesh(config.domain_size, config.n_elements_per_side)
        element_phase = classify_element_phase(mesh, geometry)
        result = solve_homogenization(
            mesh, element_phase, config.domain_size,
            config.matrix_props, config.inclusion_props, macro_strain,
        )
        effective_moduli[i] = result.macro_stress[0, 0] / e11
    return effective_moduli
