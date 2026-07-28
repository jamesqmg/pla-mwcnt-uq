"""
Tests for src/uq/micromechanics.py  — RED phase (all should fail before implementation)

Phase 4 / Track B1: Halpin-Tsai micromechanics surrogate.
Validates the deterministic modulus-uplift predictor that supplies the
reinforcement-slope prior mean (§3.5) and the independent cross-check (§3.7).

Run:  python -m pytest src/tests/test_micromechanics.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.micromechanics import (  # noqa: E402
    DENSITY_MWCNT,
    DENSITY_PLA,
    ORIENTATION_FACTORS,
    aspect_ratio,
    effective_modulus,
    halpin_tsai_modulus,
    orientation_factor,
    reinforcement_slope,
    reinforcing_efficiency,
    shape_factor,
    weight_to_volume_fraction,
)


class TestWeightToVolumeFraction:
    """wt% -> V_f conversion via component densities (Table 7 / F4 defaults)."""

    def test_zero_weight_gives_zero_volume(self):
        assert weight_to_volume_fraction(0.0) == 0.0

    def test_pure_filler_gives_unit_volume(self):
        assert weight_to_volume_fraction(100.0) == pytest.approx(1.0)

    def test_known_value_4pct(self):
        # w=0.04, rho_f=2.1, rho_m=1.24
        # V_f = (0.04/2.1) / (0.04/2.1 + 0.96/1.24)
        vf = weight_to_volume_fraction(4.0, rho_filler=2.1, rho_matrix=1.24)
        assert vf == pytest.approx(0.024012, abs=1e-5)

    def test_volume_fraction_below_weight_fraction_when_filler_denser(self):
        # filler denser than matrix -> V_f < w
        vf = weight_to_volume_fraction(5.0, rho_filler=2.1, rho_matrix=1.24)
        assert vf < 0.05

    def test_defaults_are_pla_and_mwcnt(self):
        explicit = weight_to_volume_fraction(3.0, rho_filler=DENSITY_MWCNT, rho_matrix=DENSITY_PLA)
        default = weight_to_volume_fraction(3.0)
        assert default == pytest.approx(explicit)

    def test_vectorised_input(self):
        out = weight_to_volume_fraction(np.array([0.0, 4.0, 100.0]))
        assert out[0] == pytest.approx(0.0)
        assert out[2] == pytest.approx(1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            weight_to_volume_fraction(-1.0)

    def test_weight_above_100_raises(self):
        with pytest.raises(ValueError):
            weight_to_volume_fraction(101.0)

    def test_nan_weight_raises(self):
        # "NR" dataset cells become NaN after parsing — must not propagate silently.
        with pytest.raises(ValueError):
            weight_to_volume_fraction(np.nan)

    def test_nonpositive_density_raises(self):
        with pytest.raises(ValueError):
            weight_to_volume_fraction(4.0, rho_filler=0.0)
        with pytest.raises(ValueError):
            weight_to_volume_fraction(4.0, rho_matrix=-1.24)


class TestAspectRatioAndShapeFactor:
    """l/d from CNT length (um) and diameter (nm); shape factor xi = 2(l/d)."""

    def test_aspect_ratio_unit_conversion(self):
        # 30 um / 11.5 nm = 30000 nm / 11.5 nm
        assert aspect_ratio(length_um=30.0, diameter_nm=11.5) == pytest.approx(30000.0 / 11.5)

    def test_shape_factor_is_twice_aspect_ratio(self):
        assert shape_factor(100.0) == pytest.approx(200.0)

    def test_zero_diameter_raises(self):
        with pytest.raises(ValueError):
            aspect_ratio(length_um=10.0, diameter_nm=0.0)

    def test_nonpositive_length_raises(self):
        with pytest.raises(ValueError):
            aspect_ratio(length_um=0.0, diameter_nm=11.5)


class TestReinforcingEfficiency:
    """eta = ((E_f/E_m) - 1) / ((E_f/E_m) + xi)   (Eq. 3.2)."""

    def test_efficiency_formula(self):
        E_f, E_m, xi = 1000.0, 3.0, 200.0
        ratio = E_f / E_m
        expected = (ratio - 1.0) / (ratio + xi)
        assert reinforcing_efficiency(E_f, E_m, xi) == pytest.approx(expected)

    def test_efficiency_between_zero_and_one_for_stiff_filler(self):
        eta = reinforcing_efficiency(1000.0, 3.0, 200.0)
        assert 0.0 < eta < 1.0

    def test_equal_moduli_gives_zero_efficiency(self):
        assert reinforcing_efficiency(3.0, 3.0, 200.0) == pytest.approx(0.0)

    def test_soft_filler_gives_negative_efficiency(self):
        # E_f < E_m -> eta < 0 -> composite modulus below the matrix.
        eta = reinforcing_efficiency(1.0, 3.0, 200.0)
        assert eta < 0.0
        assert halpin_tsai_modulus(1.0, 3.0, 0.02, 200.0) < 3.0

    def test_nonpositive_matrix_modulus_raises(self):
        with pytest.raises(ValueError):
            reinforcing_efficiency(1000.0, 0.0, 200.0)


class TestHalpinTsaiModulus:
    """E_c/E_m = (1 + xi*eta*V_f) / (1 - eta*V_f)   (Eq. 3.1)."""

    def test_zero_volume_returns_matrix_modulus(self):
        assert halpin_tsai_modulus(E_f=1000.0, E_m=3.0, V_f=0.0, ar=500.0) == pytest.approx(3.0)

    def test_monotonic_increasing_in_volume_fraction(self):
        vals = [halpin_tsai_modulus(1000.0, 3.0, vf, 500.0) for vf in (0.0, 0.01, 0.02, 0.05)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_monotonic_increasing_in_aspect_ratio(self):
        vals = [halpin_tsai_modulus(1000.0, 3.0, 0.02, ar) for ar in (10.0, 100.0, 1000.0)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_rom_voigt_limit_at_large_aspect_ratio(self):
        # As xi -> inf, HT longitudinal -> rule of mixtures (Voigt):
        # E_c -> (1 - V_f) E_m + V_f E_f
        E_f, E_m, V_f = 1000.0, 3.0, 0.02
        rom = (1.0 - V_f) * E_m + V_f * E_f
        ec = halpin_tsai_modulus(E_f, E_m, V_f, ar=1e7)
        assert ec == pytest.approx(rom, rel=1e-3)

    def test_result_exceeds_matrix_for_stiff_filler(self):
        assert halpin_tsai_modulus(1000.0, 3.0, 0.02, 500.0) > 3.0

    def test_nonphysical_volume_fraction_raises(self):
        # Guards caller error: a raw V_f > 1 (e.g. percent mistaken for a fraction)
        # drives eta*V_f >= 1 and a non-positive denominator.
        with pytest.raises(ValueError):
            halpin_tsai_modulus(1000.0, 3.0, V_f=1.5, ar=1.0)


class TestOrientationFactor:
    """eta_0: aligned=1, in-plane-random=3/8, 3D-random=1/5."""

    def test_aligned_is_unity(self):
        assert orientation_factor("aligned") == pytest.approx(1.0)

    def test_in_plane_random(self):
        assert orientation_factor("in-plane-random") == pytest.approx(3.0 / 8.0)

    def test_three_d_random(self):
        assert orientation_factor("3d-random") == pytest.approx(1.0 / 5.0)

    def test_table_keys_present(self):
        assert set(ORIENTATION_FACTORS) >= {"aligned", "in-plane-random", "3d-random"}

    def test_unknown_orientation_raises(self):
        with pytest.raises(ValueError):
            orientation_factor("diagonal")


class TestEffectiveModulus:
    """E_eff = E_m + eta_0 * (E_c_aligned - E_m): orientation scales the uplift."""

    def test_aligned_equals_halpin_tsai(self):
        E_f, E_m, V_f, ar = 1000.0, 3.0, 0.02, 500.0
        assert effective_modulus(E_f, E_m, V_f, ar, "aligned") == pytest.approx(
            halpin_tsai_modulus(E_f, E_m, V_f, ar)
        )

    def test_orientation_reduces_uplift(self):
        E_f, E_m, V_f, ar = 1000.0, 3.0, 0.02, 500.0
        aligned = effective_modulus(E_f, E_m, V_f, ar, "aligned")
        planar = effective_modulus(E_f, E_m, V_f, ar, "in-plane-random")
        spatial = effective_modulus(E_f, E_m, V_f, ar, "3d-random")
        assert E_m < spatial < planar < aligned

    def test_zero_volume_returns_matrix_regardless_of_orientation(self):
        for orient in ("aligned", "in-plane-random", "3d-random"):
            assert effective_modulus(1000.0, 3.0, 0.0, 500.0, orient) == pytest.approx(3.0)


class TestReinforcementSlope:
    """Predicted relative property uplift per unit wt% -> prior mean for beta_CNT."""

    def test_positive_for_stiff_filler(self):
        slope = reinforcement_slope(
            E_f=1000.0, E_m=3.0, ar=500.0,
            wt_pcts=np.array([0.0, 1.0, 2.0, 4.0]),
            orientation="in-plane-random",
        )
        assert slope > 0.0

    def test_orientation_scales_slope(self):
        wt = np.array([0.0, 1.0, 2.0, 4.0])
        aligned = reinforcement_slope(1000.0, 3.0, 500.0, wt, "aligned")
        planar = reinforcement_slope(1000.0, 3.0, 500.0, wt, "in-plane-random")
        assert aligned > planar > 0.0

    def test_requires_at_least_two_points(self):
        with pytest.raises(ValueError):
            reinforcement_slope(1000.0, 3.0, 500.0, np.array([2.0]), "aligned")

    def test_no_uplift_when_filler_matches_matrix(self):
        slope = reinforcement_slope(
            3.0, 3.0, 500.0, np.array([0.0, 1.0, 2.0, 4.0]), "aligned"
        )
        assert slope == pytest.approx(0.0, abs=1e-9)
