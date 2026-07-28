"""Directional Young's modulus from a homogenized stiffness tensor (Fig-6 raster orientation).

E(n) = 1 / (S_ijkl n_i n_j n_k n_l), where S is the compliance tensor (inverse of the
6x6 Voigt stiffness C from rve_stiffness.py) and n is a unit load direction. This gives
the raster-orientation effect directly from the aligned-fiber C -- rotating the load
relative to the fibers -- without building fibers along multiple directions in the RVE.

Convention matches rve_stiffness.py: engineering Voigt (strain order
[e_xx, e_yy, e_zz, gamma_yz, gamma_xz, gamma_xy], with gamma = 2e). Converting the
engineering-Voigt compliance to the 4th-order tensor carries a factor 1/2 on each shear
index (and 1 on each normal index). That factor rule is exactly what makes an isotropic
C give a direction-independent modulus -- the property test_rve_orientation.py checks,
which guards against the classic Voigt-factor-of-2 error.
"""
from __future__ import annotations

import numpy as np

# (i, j) -> engineering Voigt index
_VOIGT_INDEX = {
    (0, 0): 0, (1, 1): 1, (2, 2): 2,
    (1, 2): 3, (2, 1): 3,
    (0, 2): 4, (2, 0): 4,
    (0, 1): 5, (1, 0): 5,
}


def _compliance_tensor(C: np.ndarray) -> np.ndarray:
    """Full 3x3x3x3 compliance tensor from a 6x6 engineering-Voigt stiffness matrix."""
    S = np.linalg.inv(C)
    S_tensor = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            idx_ij = _VOIGT_INDEX[(i, j)]
            f_ij = 1.0 if idx_ij < 3 else 0.5
            for k in range(3):
                for m in range(3):
                    idx_km = _VOIGT_INDEX[(k, m)]
                    f_km = 1.0 if idx_km < 3 else 0.5
                    S_tensor[i, j, k, m] = S[idx_ij, idx_km] * f_ij * f_km
    return S_tensor


def youngs_modulus_in_direction(C: np.ndarray, direction) -> float:
    """Young's modulus in a load direction: 1/E(n) = S_ijkl n_i n_j n_k n_l."""
    n = np.asarray(direction, dtype=float)
    n = n / np.linalg.norm(n)
    S = _compliance_tensor(C)
    compliance = np.einsum("ijkm,i,j,k,m->", S, n, n, n, n)
    return float(1.0 / compliance)


def youngs_modulus_vs_angle(
    C: np.ndarray, angles, axial_axis: int = 2, transverse_axis: int = 0
) -> np.ndarray:
    """E(theta) as the load rotates from the fiber axis (theta=0) toward a transverse axis.

    n(theta) = cos(theta) * e_axial + sin(theta) * e_transverse.
    """
    angles = np.asarray(angles, dtype=float)
    e_axial = np.zeros(3)
    e_axial[axial_axis] = 1.0
    e_trans = np.zeros(3)
    e_trans[transverse_axis] = 1.0

    out = np.empty(angles.shape)
    for idx, theta in np.ndenumerate(angles):
        n = np.cos(theta) * e_axial + np.sin(theta) * e_trans
        out[idx] = youngs_modulus_in_direction(C, n)
    return out
