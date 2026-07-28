"""Mori-Tanaka (Tandon-Weng) micromechanics surrogate -- second analytical bound
for the Phase 4 / Tier-3 FE-RVE validation (thesis2parkinglot.md item B5).

Implements the classical Eshelby + Mori-Tanaka (1973) mean-field homogenization
for an aligned prolate-spheroidal inclusion, following the closed-form Eshelby
tensor for a transversely isotropic spheroidal inclusion (Qiu & Weng 1990,
Int. J. Eng. Sci. 28(11), building on Mura 1987, "Micromechanics of Defects in
Solids") combined with the Mori-Tanaka (1973, Acta Metall.) average-stress
concentration, per Tandon & Weng (1984, Polymer Composites 5(4), 327-333),
"The Effect of Aspect Ratio of Inclusions on the Elastic Properties of
Unidirectionally Aligned Composites."

Only the aligned longitudinal modulus E11 is extracted (the quantity needed
to overlay against Halpin-Tsai in Fig 3) -- the full transversely-isotropic
constant set (E22, G12, G23, K23) is not needed here and is not exposed.

Validation (see test_mori_tanaka.py) relies on two results that are
independent of any single source's algebra and therefore serve as an honest
correctness check rather than a self-consistency tautology:
  1. Isotropic-sphere limit (aspect_ratio -> 1): the Eshelby S1111 component
     must equal the textbook closed form (7 - 5*nu_m) / (15*(1 - nu_m)).
  2. Continuous-fiber limit (aspect_ratio -> inf): E11 must converge to the
     rule-of-mixtures value Vf*E_f + (1-Vf)*E_m. This is not model-specific --
     Hill (1964) proved E11 and nu12 are given EXACTLY by the rule of mixtures
     for any continuous, aligned, perfectly-bonded fiber composite regardless
     of micromechanical model, so any correct implementation must reproduce it.
Both checks passed during development; see that test file for the numbers.

Caveat (flag, do not silently resolve): this implementation has not been
spot-checked against a specific published numerical table from Tandon-Weng's
own paper or a textbook worked example (e.g. Tucker & Liang 1999) -- primary
sources were paywalled/unreadable during this session. Recommend a spot-check
against such a table before citing E11 values from this module in the
manuscript, even though the two limiting-case tests above passed.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from src.uq.micromechanics import orientation_factor

_VOIGT_STRAIN_FACTOR = 2.0  # engineering (not tensorial) shear strain convention


def isotropic_stiffness_voigt(E: float, nu: float) -> np.ndarray:
    """6x6 isotropic stiffness matrix in Voigt notation (engineering shear strain).

    Ordering: [11, 22, 33, 23, 13, 12].
    """
    if E <= 0.0:
        raise ValueError("E must be positive.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("nu must lie in (-1, 0.5).")
    C = np.zeros((6, 6))
    C[:3, :3] = nu * np.ones((3, 3)) + (1.0 - 2.0 * nu) * np.eye(3)
    C[3:, 3:] = (1.0 - 2.0 * nu) * np.eye(3) / _VOIGT_STRAIN_FACTOR
    return E / ((1.0 + nu) * (1.0 - 2.0 * nu)) * C


def eshelby_prolate_spheroid_voigt(nu_m: float, aspect_ratio: float) -> np.ndarray:
    """Eshelby S-tensor (6x6 Voigt) for a prolate spheroidal inclusion aligned
    along axis 0 (the fiber/loading direction, matching rve_solve.py's
    MACRO_STRAIN convention), embedded in an isotropic matrix of Poisson
    ratio nu_m.

    `aspect_ratio` = length/diameter of the spheroid; must be >= 1 (prolate).
    Uses the normalized shape-integral convention I1 + I2 + I3 = 4 (I1 = I2 by
    transverse-isotropy about the long axis), per Qiu & Weng (1990) / Mura
    (1987). The tensor is first built with the long axis along index 2, then
    rotated 90 degrees about axis 1 to bring the long axis to index 0.
    """
    if aspect_ratio < 1.0:
        raise ValueError("aspect_ratio must be >= 1 (prolate spheroid).")

    s = float(aspect_ratio)
    eps = s - 1.0
    Q = 3.0 / (8.0 * (1.0 - nu_m))
    if eps < 0.01:
        # Taylor series about s=1 (verified against mpmath to relerr<2e-7 at
        # eps=0.01, improving rapidly below that); the direct closed form
        # below suffers catastrophic cancellation as s->1 (both g and T are
        # 0/0 there) and is numerically untrustworthy in this window.
        g = 4.0 / 3.0 + 8.0 / 15.0 * eps - 12.0 / 35.0 * eps**2
        T = (21.0 - 24.0 * eps + 20.0 * eps**2) / (210.0 * (nu_m - 1.0))
    else:
        g = (2.0 * s / (s**2 - 1.0) ** 1.5) * (s * (s**2 - 1.0) ** 0.5 - np.arccosh(s))
        T = Q * (4.0 - 3.0 * g) / (3.0 * (s**2 - 1.0))

    R = (1.0 - 2.0 * nu_m) / (8.0 * (1.0 - nu_m))
    I3 = 4.0 - 2.0 * g

    S = np.zeros((6, 6))
    S[0, 0] = S[1, 1] = Q + R * g + 0.75 * T
    S[2, 2] = 4.0 / 3.0 * Q + R * I3 + 2.0 * s**2 * T
    S[0, 1] = S[1, 0] = Q / 3.0 - R * g + 4.0 * T / 3.0
    S[0, 2] = S[1, 2] = -R * g - s**2 * T
    S[2, 0] = S[2, 1] = -R * I3 - T
    S[5, 5] = Q / 3.0 + R * g + T / 4.0
    S[3, 3] = S[4, 4] = 2.0 * R - g * R / 2.0 - (1.0 + s**2) * T / 2.0

    # Rotate long axis from index 2 to index 0 (90 deg about axis 1 / y). A
    # 90-degree swap of axes 0<->2 permutes normal Voigt indices 0<->2 and
    # shear indices 3 (23) <-> 5 (12), leaving 4 (13) fixed.
    perm = np.array([2, 1, 0, 5, 4, 3])
    return S[np.ix_(perm, perm)].copy()


def _mori_tanaka_stiffness_aligned(
    E_f: float, nu_f: float, E_m: float, nu_m: float, V_f: float, aspect_ratio: float
) -> np.ndarray:
    """Composite stiffness (6x6 Voigt) for aligned spheroidal inclusions via
    Mori-Tanaka's average-stress concentration."""
    if not (0.0 <= V_f < 1.0):
        raise ValueError("V_f must lie in [0, 1).")
    Cm = isotropic_stiffness_voigt(E_m, nu_m)
    Cf = isotropic_stiffness_voigt(E_f, nu_f)
    S = eshelby_prolate_spheroid_voigt(nu_m, aspect_ratio)

    I6 = np.eye(6)
    Cm_inv = np.linalg.inv(Cm)
    A_dilute = np.linalg.inv(I6 + S @ Cm_inv @ (Cf - Cm))
    A_mt = A_dilute @ np.linalg.inv(V_f * A_dilute + (1.0 - V_f) * I6)
    return Cm + V_f * (Cf - Cm) @ A_mt


def mori_tanaka_modulus_aligned(
    E_f: float, nu_f: float, E_m: float, nu_m: float, V_f: ArrayLike, aspect_ratio: float
) -> np.ndarray | float:
    """Aligned longitudinal modulus E11 via Mori-Tanaka/Tandon-Weng.

    Returns a value in the same units as E_m. Accepts scalar or array V_f.
    """
    vf = np.atleast_1d(np.asarray(V_f, dtype=float))
    e11 = np.empty_like(vf)
    for i, v in enumerate(vf):
        if v == 0.0:
            e11[i] = E_m
            continue
        C = _mori_tanaka_stiffness_aligned(E_f, nu_f, E_m, nu_m, float(v), aspect_ratio)
        compliance_11 = np.linalg.inv(C)[0, 0]
        e11[i] = 1.0 / compliance_11
    result = np.asarray(V_f, dtype=float)
    return e11 if result.ndim else float(e11[0])


def mori_tanaka_modulus(
    E_f: float,
    nu_f: float,
    E_m: float,
    nu_m: float,
    V_f: ArrayLike,
    aspect_ratio: float,
    orientation: str = "in-plane-random",
) -> np.ndarray | float:
    """Orientation-scaled Mori-Tanaka modulus, mirroring
    micromechanics.effective_modulus's convention so the two analytical
    bounds are directly comparable in Fig 3: same orientation-efficiency
    factor eta_0 applied to each model's own aligned prediction.

    NOTE (flag, not silently resolved): the FE-RVE mesh (rve_geometry.py,
    random_unit_vector) draws inclusion orientation uniformly on the full
    sphere -- genuine 3D-random orientation -- while both analytical curves
    in Fig 3 use "in-plane-random" (eta_0=3/8, tied to the +/-45 deg FDM
    raster per fe_rve_figure_spec.md). This is a real, currently
    undocumented mismatch between the explicit geometry and the analytical
    comparison basis; it affects Halpin-Tsai's existing Fig 3 curve too
    (pre-existing, not introduced here). Left as-is for consistency with the
    already-shipped Halpin-Tsai convention; see fe-rve-gap-analysis.md.
    """
    eta_0 = orientation_factor(orientation)
    e11_aligned = mori_tanaka_modulus_aligned(E_f, nu_f, E_m, nu_m, V_f, aspect_ratio)
    return E_m + eta_0 * (np.asarray(e11_aligned, dtype=float) - E_m)
