"""Periodic-BC linear elasticity solve for Tier-3 FE-RVE homogenization.

Solves u(x) = macro_strain @ x + u*(x) for the periodic fluctuation field
u*, given a two-phase (matrix/inclusion) elasticity field on the tagged
mesh from rve_mesh.py, then returns the volume-averaged macroscopic
stress for that imposed macro strain.

Periodic BC via DOF elimination, not Lagrange multipliers or a penalty
method: every boundary node with at least one coordinate at domain_size
("slave") is expressed as an affine function of its "master" node -- the
same position with those coordinates wrapped to 0:

    u(slave) = u(master) + macro_strain @ shift,
    shift = slave_position - master_position

`shift` is exactly domain_size on each wrapped axis and zero elsewhere, so
this single rule handles faces, edges, and corners uniformly with no
special-casing -- an edge or corner is just a node with 2 or 3 wrapped
axes instead of 1. One reference node (the origin) is pinned to remove
rigid-body translation, which periodicity alone doesn't fix (it only
constrains relative displacement between paired nodes).

Correctness anchor: for a single-phase domain this recovers the imposed
affine field EXACTLY (zero fluctuation to floating-point precision) and
the textbook isotropic stress-strain relation sigma = lambda*tr(e)*I +
2*mu*e -- verified in src/tests/test_rve_solve.py and logged in
Thesis2/fe-rve-asset-inventory.md. There's no heterogeneity in that case
to require any deviation from the imposed macroscopic strain, so it's a
real correctness check, not just a smoke test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skfem
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg, spsolve
from skfem.models.elasticity import lame_parameters, linear_elasticity, sym_grad

from src.uq.rve_mesh import INCLUSION_PHASE, INTERPHASE_PHASE

MaterialProps = tuple[float, float]  # (E, nu)


@dataclass(frozen=True)
class HomogenizationResult:
    """Result of one periodic-BC elasticity solve at a given macro strain."""

    macro_strain: np.ndarray
    macro_stress: np.ndarray
    displacement: np.ndarray
    max_fluctuation: float


def _validate_material_props(props: MaterialProps, name: str) -> None:
    E, nu = props
    if E <= 0.0:
        raise ValueError(f"{name} Young's modulus must be positive, got {E}.")
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"{name} Poisson ratio must lie in (-1, 0.5), got {nu}.")


def phase_lame_fields(
    element_phase: np.ndarray,
    matrix_props: MaterialProps,
    inclusion_props: MaterialProps,
    interphase_props: MaterialProps | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-element (Lambda, Mu) arrays, shape (n_elements, 1), from phase tags.

    Supports the optional interphase third phase: elements tagged
    INTERPHASE_PHASE take `interphase_props`. Passing interphase_props=None keeps
    the two-phase behavior; it is an error for the tags to contain interphase
    elements in that case.
    """
    _validate_material_props(matrix_props, "matrix")
    _validate_material_props(inclusion_props, "inclusion")
    element_phase = np.asarray(element_phase)
    lam_m, mu_m = lame_parameters(*matrix_props)
    lam_i, mu_i = lame_parameters(*inclusion_props)

    lam_field = np.full(element_phase.shape, lam_m, dtype=float)
    mu_field = np.full(element_phase.shape, mu_m, dtype=float)
    is_inclusion = element_phase == INCLUSION_PHASE
    lam_field[is_inclusion] = lam_i
    mu_field[is_inclusion] = mu_i

    is_interphase = element_phase == INTERPHASE_PHASE
    if is_interphase.any():
        if interphase_props is None:
            raise ValueError(
                "element_phase contains INTERPHASE tags but interphase_props is None."
            )
        _validate_material_props(interphase_props, "interphase")
        lam_p, mu_p = lame_parameters(*interphase_props)
        lam_field[is_interphase] = lam_p
        mu_field[is_interphase] = mu_p

    return lam_field.reshape(-1, 1), mu_field.reshape(-1, 1)


def composite_lame_fields(
    fiber_fraction: np.ndarray,
    matrix_props: MaterialProps,
    inclusion_props: MaterialProps,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-element (Lambda, Mu) from a fiber volume fraction phi (composite-voxel).

    Mixes the constituent Lame parameters linearly by phi (a Voigt/volume-average
    per element), so boundary elements carry a partial modulus instead of the
    all-or-nothing binary tag -- the standard fix for voxel-FE staircase error.
    Two-phase (matrix/fiber); the interphase composite is layered on later.
    """
    _validate_material_props(matrix_props, "matrix")
    _validate_material_props(inclusion_props, "inclusion")
    phi = np.clip(np.asarray(fiber_fraction, dtype=float), 0.0, 1.0)
    lam_m, mu_m = lame_parameters(*matrix_props)
    lam_i, mu_i = lame_parameters(*inclusion_props)
    lam_field = ((1.0 - phi) * lam_m + phi * lam_i).reshape(-1, 1)
    mu_field = ((1.0 - phi) * mu_m + phi * mu_i).reshape(-1, 1)
    return lam_field, mu_field


def _periodic_node_map(
    mesh: skfem.MeshTet, domain_size: float, tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Master node index + shift vector for every node (see module docstring)."""
    p = mesh.p
    n_nodes = p.shape[1]
    is_high = np.isclose(p, domain_size, atol=tol)
    node_is_slave = is_high.any(axis=0)
    pos_to_idx = {tuple(np.round(p[:, i], 9)): i for i in range(n_nodes)}
    master_of = np.arange(n_nodes)
    shift_of = np.zeros((3, n_nodes))
    for i in np.nonzero(node_is_slave)[0]:
        master_pos = p[:, i].copy()
        master_pos[is_high[:, i]] = 0.0
        key = tuple(np.round(master_pos, 9))
        if key not in pos_to_idx:
            raise RuntimeError(
                f"No matching node found on the low-side face for node {i} -- "
                f"mesh is not a structured tensor grid over [0, {domain_size}]^3."
            )
        master_of[i] = pos_to_idx[key]
        shift_of[:, i] = p[:, i] - master_pos
    try:
        origin_idx = pos_to_idx[(0.0, 0.0, 0.0)]
    except KeyError:
        raise RuntimeError("Mesh has no node at the origin (0,0,0); cannot pin rigid-body translation.") from None
    return master_of, shift_of, node_is_slave, origin_idx


def assemble_periodic_constraints(
    mesh: skfem.MeshTet,
    basis: skfem.Basis,
    domain_size: float,
    macro_strain: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    """Build the DOF-elimination expansion: full_dofs = T @ reduced_dofs + b."""
    macro_strain = np.asarray(macro_strain, dtype=float)
    if macro_strain.shape != (3, 3):
        raise ValueError(f"macro_strain must be a 3x3 array, got shape {macro_strain.shape}.")
    if not np.allclose(macro_strain, macro_strain.T, atol=1e-10):
        raise ValueError("macro_strain must be symmetric.")

    master_of, shift_of, node_is_slave, origin_idx = _periodic_node_map(mesh, domain_size)
    node_b = macro_strain @ shift_of  # (3, n_nodes)

    nodal_dofs = basis.nodal_dofs
    n_nodes = mesh.p.shape[1]
    n_dofs = basis.N

    master_node_list = np.nonzero(~node_is_slave)[0]
    master_node_list = master_node_list[master_node_list != origin_idx]
    reduced_index_of = {
        (node, c): k
        for k, (node, c) in enumerate((n, c) for n in master_node_list for c in range(3))
    }
    n_reduced = 3 * len(master_node_list)

    rows, cols, vals = [], [], []
    b_full = np.zeros(n_dofs)
    for node in range(n_nodes):
        if node == origin_idx:
            continue  # pinned: u=0, no T entry, no shift
        m = master_of[node]
        for c in range(3):
            dof = nodal_dofs[c, node]
            if not node_is_slave[node]:
                rows.append(dof)
                cols.append(reduced_index_of[(node, c)])
                vals.append(1.0)
            else:
                if m != origin_idx:
                    rows.append(dof)
                    cols.append(reduced_index_of[(m, c)])
                    vals.append(1.0)
                b_full[dof] += node_b[c, node]

    T = csr_matrix((vals, (rows, cols)), shape=(n_dofs, n_reduced))
    return T, b_full


_CG_RTOL = 1e-10
_CG_MAXITER = 5000


def _amg_or_ilu_preconditioner(Kr: csr_matrix):
    """SPD preconditioner for CG: smoothed-aggregation AMG (pyamg) if installed,
    else an incomplete-LU fallback.

    pyamg is imported lazily so this module never hard-depends on it -- absent
    pyamg the default direct path is completely unaffected, and the iterative
    path degrades to ILU rather than failing to import.
    """
    try:
        import pyamg
    except ImportError:
        from scipy.sparse.linalg import LinearOperator, spilu
        ilu = spilu(Kr.tocsc())
        return LinearOperator(Kr.shape, ilu.solve)
    ml = pyamg.smoothed_aggregation_solver(Kr.tocsr())
    return ml.aspreconditioner(cycle="V")


def _solve_reduced(Kr: csr_matrix, Fr: np.ndarray, solver: str, M=None, callback=None) -> np.ndarray:
    """Solve the reduced SPD system Kr @ u = Fr by the requested method.

    "direct" is the original SuperLU path (unchanged default). "iterative"/"cg"
    runs preconditioned conjugate gradient, which keeps memory ~O(N) at
    production mesh sizes where direct LU fill-in OOMs. A prebuilt preconditioner
    `M` is reused when given (so the six load cases of one RVE share one AMG
    setup); `callback` is forwarded to `cg` for iteration counting.
    """
    if solver == "direct":
        return spsolve(Kr, Fr)
    if solver in ("iterative", "cg"):
        if M is None:
            M = _amg_or_ilu_preconditioner(Kr)
        u, info = cg(Kr, Fr, rtol=_CG_RTOL, atol=0.0, maxiter=_CG_MAXITER, M=M, callback=callback)
        if info != 0:
            raise RuntimeError(
                f"CG failed to converge to rtol={_CG_RTOL} within {_CG_MAXITER} "
                f"iterations (info={info}); the preconditioner may be inadequate "
                "for this contrast/mesh."
            )
        return u
    raise ValueError(f"Unknown solver {solver!r}; expected 'direct' or 'iterative'.")


def _macro_stress(
    basis: skfem.Basis,
    u_full: np.ndarray,
    lam_field: np.ndarray,
    mu_field: np.ndarray,
    n_elements: int,
) -> np.ndarray:
    """Volume-averaged macroscopic 3x3 stress from a full displacement field."""
    uh = basis.interpolate(u_full)
    strain = sym_grad(uh)  # (3, 3, n_elements, n_quadpoints)
    lam_b = lam_field.reshape(1, 1, n_elements, 1)
    mu_b = mu_field.reshape(1, 1, n_elements, 1)
    trace = strain[0, 0] + strain[1, 1] + strain[2, 2]
    stress = 2.0 * mu_b * strain
    for d in range(3):
        stress[d, d] = stress[d, d] + lam_b[0, 0] * trace
    dx = basis.dx
    total_volume = dx.sum()
    return np.array([
        [(stress[i, j] * dx).sum() / total_volume for j in range(3)]
        for i in range(3)
    ])


def solve_homogenization(
    mesh: skfem.MeshTet,
    element_phase: np.ndarray,
    domain_size: float,
    matrix_props: MaterialProps,
    inclusion_props: MaterialProps,
    macro_strain: np.ndarray,
    interphase_props: MaterialProps | None = None,
    lame_fields: tuple[np.ndarray, np.ndarray] | None = None,
    solver: str = "direct",
) -> HomogenizationResult:
    """Solve one periodic-BC load case and return the homogenized response.

    `lame_fields` (a precomputed (Lambda, Mu) pair, e.g. from composite_lame_fields)
    bypasses the tag-based material assignment when given, so element_phase may be
    None in that case. `solver` selects the linear solve: "direct" (default,
    SuperLU) or "iterative" (preconditioned CG, OOM-safe at production scale).
    """
    e = skfem.ElementVector(skfem.ElementTetP1())
    basis = skfem.Basis(mesh, e)

    if lame_fields is not None:
        lam_field, mu_field = lame_fields
    else:
        lam_field, mu_field = phase_lame_fields(element_phase, matrix_props, inclusion_props, interphase_props)
    K = linear_elasticity(lam_field, mu_field).assemble(basis).tocsr()

    T, b_full = assemble_periodic_constraints(mesh, basis, domain_size, macro_strain)
    Kr = (T.T @ K @ T).tocsr()
    Fr = -(T.T @ (K @ b_full))
    u_reduced = _solve_reduced(Kr, Fr, solver)
    u_full = T @ u_reduced + b_full

    macro_strain = np.asarray(macro_strain, dtype=float)
    nodal_dofs = basis.nodal_dofs
    p = mesh.p
    u_grid = np.column_stack([u_full[nodal_dofs[c, :]] for c in range(3)])
    affine = (macro_strain @ p).T
    max_fluctuation = float(np.abs(u_grid - affine).max())

    macro_stress = _macro_stress(basis, u_full, lam_field, mu_field, mesh.t.shape[1])

    return HomogenizationResult(
        macro_strain=macro_strain,
        macro_stress=macro_stress,
        displacement=u_full,
        max_fluctuation=max_fluctuation,
    )


def _periodic_rhs_shift(
    mesh: skfem.MeshTet, basis: skfem.Basis, domain_size: float, macro_strain: np.ndarray
) -> np.ndarray:
    """Just the inhomogeneous b_full term for a macro strain (T is strain-independent).

    Matches the b_full built inside assemble_periodic_constraints, but skips the T
    assembly so several macro strains can reuse one T. Slave DOFs carry the imposed
    affine jump macro_strain @ shift; all other DOFs are zero.
    """
    _, shift_of, node_is_slave, origin_idx = _periodic_node_map(mesh, domain_size)
    node_b = np.asarray(macro_strain, dtype=float) @ shift_of  # (3, n_nodes)
    nodal_dofs = basis.nodal_dofs
    b_full = np.zeros(basis.N)
    slaves = np.nonzero(node_is_slave)[0]
    slaves = slaves[slaves != origin_idx]
    for c in range(3):
        b_full[nodal_dofs[c, slaves]] = node_b[c, slaves]
    return b_full


def solve_macro_strains(
    mesh: skfem.MeshTet,
    domain_size: float,
    matrix_props: MaterialProps,
    inclusion_props: MaterialProps,
    macro_strains,
    element_phase: np.ndarray | None = None,
    interphase_props: MaterialProps | None = None,
    lame_fields: tuple[np.ndarray, np.ndarray] | None = None,
    solver: str = "direct",
    iter_counts: list | None = None,
) -> list[np.ndarray]:
    """Homogenized macro-stress for several macro strains that share one assembly.

    Assembles the stiffness K, the periodic operator T, the reduced matrix Kr and
    (for the iterative solver) the AMG/ILU preconditioner ONCE, then solves each
    macro strain as a fresh right-hand side. Result-identical to calling
    solve_homogenization once per strain, but avoids the redundant assembly and
    preconditioner setup that dominate the six-load-case stiffness extraction. If
    `iter_counts` is a list, each strain's CG iteration count is appended to it.
    """
    e = skfem.ElementVector(skfem.ElementTetP1())
    basis = skfem.Basis(mesh, e)
    if lame_fields is not None:
        lam_field, mu_field = lame_fields
    else:
        lam_field, mu_field = phase_lame_fields(element_phase, matrix_props, inclusion_props, interphase_props)
    K = linear_elasticity(lam_field, mu_field).assemble(basis).tocsr()

    T, _ = assemble_periodic_constraints(mesh, basis, domain_size, macro_strains[0])
    Kr = (T.T @ K @ T).tocsr()
    M = _amg_or_ilu_preconditioner(Kr) if solver in ("iterative", "cg") else None

    n_elements = mesh.t.shape[1]
    stresses = []
    for macro_strain in macro_strains:
        b_full = _periodic_rhs_shift(mesh, basis, domain_size, macro_strain)
        Fr = -(T.T @ (K @ b_full))
        counter = {"n": 0}
        callback = (lambda _xk: counter.__setitem__("n", counter["n"] + 1)) if iter_counts is not None else None
        u_reduced = _solve_reduced(Kr, Fr, solver, M=M, callback=callback)
        if iter_counts is not None:
            iter_counts.append(counter["n"])
        u_full = T @ u_reduced + b_full
        stresses.append(_macro_stress(basis, u_full, lam_field, mu_field, n_elements))
    return stresses
