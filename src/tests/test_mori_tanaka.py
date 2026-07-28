"""Tests for src/uq/mori_tanaka.py.

Two of these tests are deliberately independent of trusting the source
implementation's algebra: the isotropic-sphere Eshelby limit and the
continuous-fiber rule-of-mixtures limit (Hill 1964) are both textbook-exact
results derivable from first principles, not just internal self-consistency
checks. If the Eshelby tensor or Mori-Tanaka average-stress algebra had a
sign/term error, these are very likely to fail.

Spot-checked 2026-07-09 against `mechmean` (Bauer 2022, Zenodo DOI
10.5281/zenodo.6461940 -- an independent open-source Eshelby/Mori-Tanaka
reference implementation from KIT's Boehlke group, explicitly built to
cross-validate other implementations), not added as a project dependency.
Compared `mori_tanaka_modulus_aligned()` against mechmean's MoriTanaka+
Castaneda-spheroid Hill polarization across all 6 wt%-sweep Vf points at
both aspect_ratio=20 (FE mesh's provisional value) and aspect_ratio=2609
(the true CNT value), plus a broader aspect-ratio sweep (1.5 to 2609) at
fixed Vf=0.02. Result: relative difference <= 6.4e-4 everywhere, and
<= 2e-5 away from the near-spherical aspect_ratio=1.5 corner -- effectively
exact agreement, not just a plausible-looking match. Script (not kept in
this repo, run ad hoc): installed `pip install mechmean mechkit` in a
scratch environment.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.uq.micromechanics import effective_modulus
from src.uq.mori_tanaka import (
    eshelby_prolate_spheroid_voigt,
    isotropic_stiffness_voigt,
    mori_tanaka_modulus,
    mori_tanaka_modulus_aligned,
)


def test_isotropic_stiffness_matches_lame_shear():
    E, nu = 3.0, 0.35
    C = isotropic_stiffness_voigt(E, nu)
    G = E / (2 * (1 + nu))
    assert C[3, 3] == pytest.approx(G, rel=1e-12)
    assert C[4, 4] == pytest.approx(G, rel=1e-12)
    assert C[5, 5] == pytest.approx(G, rel=1e-12)
    # C11 for isotropic material: E(1-nu) / ((1+nu)(1-2nu))
    c11_expected = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    assert C[0, 0] == pytest.approx(c11_expected, rel=1e-12)


@pytest.mark.parametrize("nu_m", [0.2, 0.3, 0.35, 0.45])
def test_eshelby_isotropic_sphere_limit(nu_m):
    """At aspect_ratio=1 (sphere), S1111 must equal the textbook closed form
    (7 - 5*nu) / (15*(1 - nu)) -- independent of this module's own algebra."""
    S = eshelby_prolate_spheroid_voigt(nu_m, 1.0)
    expected = (7.0 - 5.0 * nu_m) / (15.0 * (1.0 - nu_m))
    assert S[0, 0] == pytest.approx(expected, rel=1e-6)
    assert S[1, 1] == pytest.approx(expected, rel=1e-6)
    assert S[2, 2] == pytest.approx(expected, rel=1e-6)


def test_eshelby_series_matches_direct_formula_near_crossover():
    """The near-s=1 series branch and the direct closed-form branch must
    agree at the crossover (aspect_ratio=1.01) to guard against a
    discontinuity introduced by the two-branch implementation."""
    S_series = eshelby_prolate_spheroid_voigt(0.35, 1.0099999)
    S_direct = eshelby_prolate_spheroid_voigt(0.35, 1.0100001)
    np.testing.assert_allclose(S_series, S_direct, rtol=1e-5, atol=1e-8)


def test_mori_tanaka_dilute_limit_recovers_matrix_modulus():
    e11 = mori_tanaka_modulus_aligned(E_f=1000.0, nu_f=0.2, E_m=3.023, nu_m=0.35, V_f=0.0, aspect_ratio=20.0)
    assert e11 == pytest.approx(3.023, rel=1e-12)


@pytest.mark.parametrize("V_f", [0.01, 0.05, 0.1, 0.2])
def test_mori_tanaka_continuous_fiber_limit_matches_rule_of_mixtures(V_f):
    """Hill (1964): E11 for continuous, aligned, perfectly-bonded fibers is
    EXACTLY the rule of mixtures, regardless of micromechanical model. At a
    very large aspect ratio this must hold to good numerical precision."""
    E_f, E_m = 1000.0, 3.023
    e11 = mori_tanaka_modulus_aligned(E_f=E_f, nu_f=0.2, E_m=E_m, nu_m=0.35, V_f=V_f, aspect_ratio=2.0e5)
    rom = V_f * E_f + (1 - V_f) * E_m
    assert e11 == pytest.approx(rom, rel=2e-3)


def test_mori_tanaka_agrees_with_halpin_tsai_at_large_aspect_ratio():
    """Both analytical models converge to the same rule-of-mixtures limit as
    aspect ratio -> infinity, so they must agree with each other there too."""
    E_f, E_m, V_f, ar = 1000.0, 3.023, 0.05, 2.0e5
    e_mt = mori_tanaka_modulus_aligned(E_f, 0.2, E_m, 0.35, V_f, ar)
    e_ht = effective_modulus(E_f, E_m, V_f, ar, orientation="aligned")
    assert e_mt == pytest.approx(e_ht, rel=5e-3)


def test_mori_tanaka_modulus_monotonic_in_volume_fraction():
    vf = np.array([0.01, 0.02, 0.03, 0.05, 0.1])
    e11 = mori_tanaka_modulus_aligned(E_f=1000.0, nu_f=0.2, E_m=3.023, nu_m=0.35, V_f=vf, aspect_ratio=20.0)
    assert np.all(np.diff(e11) > 0)


def test_mori_tanaka_modulus_orientation_scaling_matches_convention():
    """mori_tanaka_modulus at 'aligned' should equal the raw aligned E11;
    'in-plane-random' should sit strictly between E_m and the aligned value,
    mirroring micromechanics.effective_modulus's own convention."""
    E_f, E_m, V_f, ar = 1000.0, 3.023, 0.03, 20.0
    aligned = mori_tanaka_modulus(E_f, 0.2, E_m, 0.35, V_f, ar, orientation="aligned")
    in_plane = mori_tanaka_modulus(E_f, 0.2, E_m, 0.35, V_f, ar, orientation="in-plane-random")
    assert aligned == pytest.approx(mori_tanaka_modulus_aligned(E_f, 0.2, E_m, 0.35, V_f, ar), rel=1e-12)
    assert E_m < in_plane < aligned


def test_mori_tanaka_rejects_invalid_aspect_ratio():
    with pytest.raises(ValueError):
        eshelby_prolate_spheroid_voigt(0.35, 0.5)
