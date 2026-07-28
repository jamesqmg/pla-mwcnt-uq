"""Full anisotropic stiffness homogenization for Tier-3 FE-RVE.

The continuous-through-fiber redesign makes the RVE response anisotropic
(transversely isotropic for aligned fibers): a single scalar effective modulus
no longer describes it. This module extracts the complete 6x6 stiffness matrix C
by running the six independent unit macro-strain load cases through
`rve_solve.solve_homogenization` -- which already returns the full macro-stress
for any imposed symmetric macro-strain, so no change to the solver core is
needed -- and assembling C column by column.

Convention: Voigt engineering notation with strain order
[e_xx, e_yy, e_zz, gamma_yz, gamma_xz, gamma_xy] (engineering shear
gamma_ij = 2 e_ij) and stress order [s_xx, s_yy, s_zz, s_yz, s_xz, s_xy], so
that sigma_voigt = C @ strain_voigt. Column k of C is read off by imposing a
unit Voigt strain in component k and recording the resulting macro-stress. The
shear load cases therefore impose a tensor strain of 0.5 on the relevant
off-diagonal pair (gamma = 1 <-> e_ij = 0.5). This is validated to reproduce the
textbook isotropic stiffness (C11 = lam + 2mu, C12 = lam, C44 = mu) exactly for
a single-phase domain -- see test_rve_stiffness.py.

Performance: this runs six full solves per RVE. Each solve currently reassembles
and refactorizes the stiffness matrix, which depends only on the mesh, phase
tags, and material props -- not on the macro-strain. A production optimization
(factorize once, solve six right-hand sides) is deferred; at the convergence /
production scale it is worth doing, but it changes speed, not the result.
"""
from __future__ import annotations

import numpy as np

from src.uq.rve_solve import MaterialProps, solve_macro_strains


def _voigt_unit_strains() -> list[np.ndarray]:
    """The six symmetric unit macro-strain tensors, one per Voigt component.

    Normal cases set a single diagonal component to 1; shear cases set the
    symmetric off-diagonal pair to 0.5 so the engineering shear gamma equals 1.
    """
    cases: list[np.ndarray] = []
    for a in range(3):  # e_xx, e_yy, e_zz
        e = np.zeros((3, 3))
        e[a, a] = 1.0
        cases.append(e)
    for i, j in ((1, 2), (0, 2), (0, 1)):  # gamma_yz, gamma_xz, gamma_xy
        e = np.zeros((3, 3))
        e[i, j] = e[j, i] = 0.5
        cases.append(e)
    return cases


def _stress_to_voigt(sigma: np.ndarray) -> np.ndarray:
    """Pack a symmetric 3x3 stress tensor into Voigt order [xx, yy, zz, yz, xz, xy]."""
    return np.array([
        sigma[0, 0], sigma[1, 1], sigma[2, 2],
        sigma[1, 2], sigma[0, 2], sigma[0, 1],
    ])


def full_stiffness_tensor(
    mesh,
    element_phase: np.ndarray,
    domain_size: float,
    matrix_props: MaterialProps,
    inclusion_props: MaterialProps,
    interphase_props: MaterialProps | None = None,
    lame_fields: tuple[np.ndarray, np.ndarray] | None = None,
    solver: str = "direct",
) -> np.ndarray:
    """Homogenized 6x6 Voigt stiffness matrix C (see module docstring for convention).

    `lame_fields` (e.g. from composite_lame_fields) is passed through so a
    composite-voxel material field is assembled once and reused. `solver`
    ("direct" | "iterative") is forwarded. The six load cases share a single
    assembly and preconditioner setup via solve_macro_strains.
    """
    stresses = solve_macro_strains(
        mesh, domain_size, matrix_props, inclusion_props, _voigt_unit_strains(),
        element_phase=element_phase, interphase_props=interphase_props,
        lame_fields=lame_fields, solver=solver,
    )
    C = np.zeros((6, 6))
    for k, macro_stress in enumerate(stresses):
        C[:, k] = _stress_to_voigt(macro_stress)
    return C


def engineering_constants(C: np.ndarray, axial: int = 2) -> tuple[float, float]:
    """Axial and transverse Young's moduli from the stiffness matrix.

    Inverts C to the compliance S and reads Young's moduli off its normal-strain
    diagonal (E_i = 1 / S_ii). `axial` is the fiber-axis index (0/1/2); the
    transverse modulus is averaged over the two transverse directions, which are
    equal for an ideally transversely-isotropic response and close for a
    finite realization.
    """
    S = np.linalg.inv(C)
    transverse = [a for a in range(3) if a != axial]
    e_axial = 1.0 / S[axial, axial]
    e_transverse = 0.5 * (1.0 / S[transverse[0], transverse[0]] + 1.0 / S[transverse[1], transverse[1]])
    return float(e_axial), float(e_transverse)
