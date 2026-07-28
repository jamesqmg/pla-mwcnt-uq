"""Halpin-Tsai micromechanics surrogate for the Phase 4 physics-based validation.

Implements the deterministic modulus-uplift predictor described in §3.7 of the
methodology:

    E_c / E_m = (1 + xi * eta * V_f) / (1 - eta * V_f)            (Eq. 3.1)
    eta       = ((E_f/E_m) - 1) / ((E_f/E_m) + xi),  xi = 2(l/d)  (Eq. 3.2)

An orientation-efficiency factor eta_0 (1 aligned, 3/8 in-plane-random,
1/5 three-dimensionally random) scales the predicted longitudinal uplift to the
FDM raster condition. These three values are Krenchel's (1964) orientation
efficiency factors -- confirmed 2026-07-09 by reading hassanzadeh_2019 in full:
that paper's own orientation factor (fR=1/3 planar, 1/6 3D, via Cox 1952) uses
different numbers and is NOT the source of eta_0 here, despite covering the same
base Halpin-Tsai equations below (Eq 3.1/3.2 match its Eqs 1-2 term for term).
See References/references.bib:krenchel_1964. The reinforcement slope implied by
these equations — the predicted relative property increase per unit MWCNT weight
fraction — supplies the prior mean for beta_CNT in §3.5.

Honest scope (F1): Halpin-Tsai rigorously predicts elastic *modulus*, not
failure *strength*. These functions therefore inform only the sign and order of
magnitude of the reinforcement trend; absolute-strength prediction would require
interfacial parameters not recoverable from the literature dataset. The surrogate
is driven solely by material and geometric inputs and never sees the measured
tensile strengths, keeping the Phase 4 cross-check genuinely independent.

All modulus functions are unit-agnostic: E_f and E_m must share the same units
(the dataset stores CNT modulus in TPa and matrix modulus in GPa, so convert
before calling). V_f conversion follows the component-density rule with the F4
defaults below.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# F4 density defaults (g/cm^3): PLA from dataset, MWCNT from literature.
DENSITY_PLA: float = 1.24
DENSITY_MWCNT: float = 2.1

# Orientation-efficiency factors eta_0 (§3.7). Krenchel (1964) -- see module
# docstring; NOT from hassanzadeh_2019, which uses different orientation factors.
ORIENTATION_FACTORS: dict[str, float] = {
    "aligned": 1.0,
    "in-plane-random": 3.0 / 8.0,
    "3d-random": 1.0 / 5.0,
}


def weight_to_volume_fraction(
    wt_pct: ArrayLike,
    rho_filler: float = DENSITY_MWCNT,
    rho_matrix: float = DENSITY_PLA,
) -> np.ndarray | float:
    """Convert MWCNT weight percent to volume fraction via component densities.

    V_f = (w/rho_f) / (w/rho_f + (1-w)/rho_m),  with w = wt_pct / 100.

    Accepts a scalar or array. Raises ValueError if any weight is outside [0, 100].
    """
    if rho_filler <= 0.0 or rho_matrix <= 0.0:
        raise ValueError("densities must be positive.")
    w = np.asarray(wt_pct, dtype=float) / 100.0
    if np.any(np.isnan(w)):
        raise ValueError("wt_pct contains NaN (e.g. an unparsed 'NR' dataset cell).")
    if np.any(w < 0.0) or np.any(w > 1.0):
        raise ValueError("wt_pct must lie in [0, 100].")
    vol_filler = w / rho_filler
    vol_matrix = (1.0 - w) / rho_matrix
    vf = vol_filler / (vol_filler + vol_matrix)
    return vf if vf.ndim else float(vf)


def aspect_ratio(length_um: float, diameter_nm: float) -> float:
    """CNT aspect ratio l/d, converting length (um) and diameter (nm) to a ratio."""
    if diameter_nm <= 0.0:
        raise ValueError("diameter_nm must be positive.")
    if length_um <= 0.0:
        raise ValueError("length_um must be positive.")
    return (length_um * 1_000.0) / diameter_nm


def shape_factor(ar: float) -> float:
    """Halpin-Tsai shape factor xi = 2(l/d) for longitudinal reinforcement (Eq. 3.2)."""
    return 2.0 * ar


def reinforcing_efficiency(E_f: float, E_m: float, xi: float) -> float:
    """Reinforcing efficiency eta = ((E_f/E_m) - 1) / ((E_f/E_m) + xi)  (Eq. 3.2)."""
    if E_m <= 0.0:
        raise ValueError("E_m must be positive.")
    ratio = E_f / E_m
    return (ratio - 1.0) / (ratio + xi)


def halpin_tsai_modulus(E_f: float, E_m: float, V_f: ArrayLike, ar: float) -> np.ndarray | float:
    """Longitudinal composite modulus E_c from the Halpin-Tsai relation (Eq. 3.1).

    Returns a value in the same units as E_m. As ar -> inf this approaches the
    rule-of-mixtures (Voigt) upper bound (1 - V_f) E_m + V_f E_f.
    """
    xi = shape_factor(ar)
    eta = reinforcing_efficiency(E_f, E_m, xi)
    vf = np.asarray(V_f, dtype=float)
    denom = 1.0 - eta * vf
    if np.any(denom <= 0.0):
        raise ValueError(
            "eta*V_f >= 1: Halpin-Tsai denominator non-positive; check V_f is physical (< ~0.7)."
        )
    ec = (1.0 + xi * eta * vf) / denom * E_m
    return ec if ec.ndim else float(ec)


def orientation_factor(orientation: str) -> float:
    """Look up the orientation-efficiency factor eta_0 for a deposition condition."""
    try:
        return ORIENTATION_FACTORS[orientation]
    except KeyError:
        raise ValueError(
            f"Unknown orientation {orientation!r}; expected one of {sorted(ORIENTATION_FACTORS)}."
        ) from None


def effective_modulus(
    E_f: float, E_m: float, V_f: ArrayLike, ar: float, orientation: str = "in-plane-random"
) -> np.ndarray | float:
    """Orientation-scaled composite modulus: E_m + eta_0 * (E_c_aligned - E_m).

    The orientation factor scales only the predicted uplift over the matrix, so a
    zero volume fraction returns E_m regardless of orientation.
    """
    eta_0 = orientation_factor(orientation)
    ec_aligned = halpin_tsai_modulus(E_f, E_m, V_f, ar)
    return E_m + eta_0 * (ec_aligned - E_m)


def reinforcement_slope(
    E_f: float,
    E_m: float,
    ar: float,
    wt_pcts: ArrayLike,
    orientation: str = "in-plane-random",
    rho_filler: float = DENSITY_MWCNT,
    rho_matrix: float = DENSITY_PLA,
) -> float:
    """Predicted relative modulus uplift per unit MWCNT weight percent.

    Computes the orientation-scaled modulus across the supplied weight fractions,
    expresses it as a relative uplift (E_eff/E_m - 1), and returns the
    least-squares slope of that uplift against weight percent. This is the
    physics-implied reinforcement *trend* used as the prior mean for beta_CNT
    (sign + order of magnitude only; see module docstring on scope).
    """
    wt = np.asarray(wt_pcts, dtype=float)
    if wt.size < 2:
        raise ValueError("reinforcement_slope needs at least two weight fractions.")
    vf = weight_to_volume_fraction(wt, rho_filler=rho_filler, rho_matrix=rho_matrix)
    e_eff = effective_modulus(E_f, E_m, vf, ar, orientation)
    rel_uplift = np.asarray(e_eff, dtype=float) / E_m - 1.0
    slope, _ = np.polyfit(wt, rel_uplift, 1)
    return float(slope)
