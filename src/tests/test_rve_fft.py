"""Tests for FFT-based homogenization (Moulinec-Suquet) — the sanctioned reopen
path for wavy-absolute-E1 (fe-rve-asset-inventory.md sec 14 REVISIT-IF).

Correctness spine (independent of the fiber voxelizer):
  1. Single-phase medium -> exact isotropic C (C11=lam+2mu, C12=lam, C44=mu).
  2. Two-phase LAMINATE -> exact Backus (1962) closed form. FFT resolves pure
     grid-aligned layering with no staircase error, so this is a tight gate on
     the Green operator + solver core.
Physical cross-method gate (ties to the existing voxel-FE pipeline):
  3. Straight aligned continuous fibers -> axial E1 ~= Voigt rule-of-mixtures,
     matching the converged voxel-FE straight-fiber result (ROM +0.05% gate).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.uq.rve_fft import (
    full_stiffness_tensor_fft,
    homogenize_fft,
    lame_from_E_nu,
    voxelize_straight_fibers,
)
from src.uq.rve_stiffness import engineering_constants

MATRIX = (3.023, 0.35)     # (E, nu) — neat PLA
FIBER = (1000.0, 0.2)      # (E, nu) — MWCNT, 330x contrast


def _uniform_fields(shape, E, nu):
    lam, mu = lame_from_E_nu(E, nu)
    return np.full(shape, lam), np.full(shape, mu)


# --- Gate 1: single-phase exact isotropic stiffness -------------------------

def test_homogeneous_medium_recovers_isotropic_stiffness():
    n = 16
    lam, mu = lame_from_E_nu(*MATRIX)
    lam_f, mu_f = _uniform_fields((n, n, n), *MATRIX)
    C = full_stiffness_tensor_fft(lam_f, mu_f, domain_size=1.0, tol=1e-10)
    # C11 = lam + 2mu, C12 = lam, C44 = mu (Voigt engineering).
    assert C[0, 0] == pytest.approx(lam + 2 * mu, rel=1e-6)
    assert C[0, 1] == pytest.approx(lam, rel=1e-6)
    assert C[3, 3] == pytest.approx(mu, rel=1e-6)
    # Isotropy: the three normal-diagonal and three shear-diagonal blocks equal.
    assert C[1, 1] == pytest.approx(C[0, 0], rel=1e-6)
    assert C[2, 2] == pytest.approx(C[0, 0], rel=1e-6)
    assert C[4, 4] == pytest.approx(C[3, 3], rel=1e-6)
    assert C[5, 5] == pytest.approx(C[3, 3], rel=1e-6)


def test_homogeneous_medium_single_load_case():
    n = 8
    lam, mu = lame_from_E_nu(*MATRIX)
    lam_f, mu_f = _uniform_fields((n, n, n), *MATRIX)
    macro = np.zeros((3, 3))
    macro[2, 2] = 1.0  # uniaxial axial strain
    sigma, info = homogenize_fft(lam_f, mu_f, macro, domain_size=1.0, tol=1e-10)
    # Uniform medium: sigma = lam*tr(eps)*I + 2mu*eps, converges in 1 iteration.
    assert sigma[2, 2] == pytest.approx(lam + 2 * mu, rel=1e-8)
    assert sigma[0, 0] == pytest.approx(lam, rel=1e-8)
    assert sigma[0, 1] == pytest.approx(0.0, abs=1e-8)
    assert info["iterations"] <= 2


# --- Gate 2: two-phase laminate = exact Backus closed form ------------------

def _backus_laminate_C(props1, f1, props2, layer_axis):
    """Exact Backus (1962) effective stiffness for a two-phase isotropic laminate
    layered along `layer_axis`, returned as a 6x6 Voigt matrix. f1 is the volume
    fraction of phase 1; f2 = 1 - f1."""
    f2 = 1.0 - f1
    l1, m1 = lame_from_E_nu(*props1)
    l2, m2 = lame_from_E_nu(*props2)

    def avg(a1, a2):
        return f1 * a1 + f2 * a2

    def havg(a1, a2):  # harmonic (series) average
        return 1.0 / (f1 / a1 + f2 / a2)

    p1, p2 = l1 + 2 * m1, l2 + 2 * m2  # P-wave modulus per phase
    inv_p = havg(p1, p2)               # <1/(lam+2mu)>^-1
    r = avg(l1 / p1, l2 / p2)          # <lam/(lam+2mu)>

    C33 = inv_p
    C13 = r * inv_p
    C11 = avg(4 * m1 * (l1 + m1) / p1, 4 * m2 * (l2 + m2) / p2) + r * r * inv_p
    C12 = avg(2 * m1 * l1 / p1, 2 * m2 * l2 / p2) + r * r * inv_p
    C44 = havg(m1, m2)   # shear across the layering plane (harmonic)
    C66 = avg(m1, m2)    # shear in the layering plane (arithmetic)

    # Assemble transversely-isotropic C with symmetry axis = layer_axis.
    C = np.zeros((6, 6))
    ax = layer_axis
    tr = [i for i in range(3) if i != ax]
    # normal block
    C[ax, ax] = C33
    for t in tr:
        C[t, t] = C11
        C[t, ax] = C[ax, t] = C13
    C[tr[0], tr[1]] = C[tr[1], tr[0]] = C12
    # shear block: Voigt shear index k corresponds to the off-axis pair not-(i,j)
    # yz=3 (pair 1,2), xz=4 (pair 0,2), xy=5 (pair 0,1)
    pair_of = {3: {1, 2}, 4: {0, 2}, 5: {0, 1}}
    for k in (3, 4, 5):
        C[k, k] = C66 if ax not in pair_of[k] else C44
    return C


@pytest.mark.parametrize("layer_axis", [0, 1, 2])
def test_two_phase_laminate_matches_backus(layer_axis):
    n = 24
    f1 = 0.5  # phase 1 = fiber, occupies half the layers
    lam_m, mu_m = lame_from_E_nu(*MATRIX)
    lam_f, mu_f = lame_from_E_nu(*FIBER)
    lam = np.full((n, n, n), lam_m)
    mu = np.full((n, n, n), mu_m)
    idx = [slice(None)] * 3
    idx[layer_axis] = slice(0, n // 2)  # first half = fiber layers
    lam[tuple(idx)] = lam_f
    mu[tuple(idx)] = mu_f

    C = full_stiffness_tensor_fft(lam, mu, domain_size=1.0, tol=1e-9, max_iter=5000)
    C_exact = _backus_laminate_C(FIBER, f1, MATRIX, layer_axis)
    # Grid-aligned layering: FFT is exact up to solver tolerance.
    assert np.allclose(C, C_exact, rtol=2e-4, atol=2e-4 * C_exact.max())


# --- Gate 3: straight aligned fibers -> axial E1 ~= Voigt ROM ---------------

def test_straight_fiber_axial_modulus_matches_rule_of_mixtures():
    n = 48
    domain, radius, axis = 12.0, 0.5, 2
    # Place fibers on a grid to hit ~Vf=0.07; ROM axial E1 = Vf*Ef + (1-Vf)*Em.
    lam, mu, vf = voxelize_straight_fibers(
        n=n, domain_size=domain, radius=radius, axis=axis,
        matrix=MATRIX, fiber=FIBER, n_fibers_per_side=1,  # single central fiber
    )
    C = full_stiffness_tensor_fft(lam, mu, domain_size=domain, tol=1e-6,
                                  max_iter=4000, scheme="willot")
    E1, E2 = engineering_constants(C, axial=axis)
    rom = vf * FIBER[0] + (1 - vf) * MATRIX[0]
    # Continuous aligned fibers are iso-strain axially -> E1 ~= Voigt ROM.
    assert E1 == pytest.approx(rom, rel=0.03)
    assert E2 < E1  # transverse is compliant (matrix-dominated)
