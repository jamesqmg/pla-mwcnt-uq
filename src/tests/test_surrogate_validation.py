"""
Tests for src/uq/surrogate_validation.py  — RED phase (fail before implementation).

Phase 4 / Track B4: the Tier-2 physics cross-check. The Halpin-Tsai surrogate is an
independent deterministic predictor (it never sees the measured tensile strengths). It
is compared against the FLAT-prior Bayesian model — not the informed-prior model — to
avoid circularity. The comparison is honest: it assesses the reinforcement TREND
(sign + order of magnitude), never pointwise absolute strength (modulus != strength).

Run:  python -m pytest src/tests/test_surrogate_validation.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.surrogate_validation import (  # noqa: E402
    assess_surrogate_consistency,
    directional_agreement,
    magnitude_ratio,
    relative_data_slope,
)


class TestRelativeDataSlope:
    """Normalise a raw strength slope (MPa/wt%) to a relative slope (fraction per wt%)."""

    def test_normalises_by_mean_strength(self):
        assert relative_data_slope(1.24, 62.0) == pytest.approx(0.02)

    def test_zero_slope_is_zero(self):
        assert relative_data_slope(0.0, 62.0) == 0.0

    def test_nonpositive_strength_raises(self):
        with pytest.raises(ValueError):
            relative_data_slope(1.24, 0.0)


class TestDirectionalAgreement:
    """Sign agreement between the physics slope and the data slope."""

    def test_both_positive_agree(self):
        assert directional_agreement(0.70, 1.13) is True

    def test_both_negative_agree(self):
        assert directional_agreement(-0.5, -0.2) is True

    def test_opposite_signs_disagree(self):
        assert directional_agreement(0.70, -0.2) is False

    def test_zero_data_slope_disagrees(self):
        assert directional_agreement(0.70, 0.0) is False


class TestMagnitudeRatio:
    """Ratio of the HT relative-modulus slope to the data relative-strength slope."""

    def test_ratio_value(self):
        assert magnitude_ratio(0.70, 0.0182) == pytest.approx(0.70 / 0.0182)

    def test_zero_data_slope_raises(self):
        with pytest.raises(ValueError):
            magnitude_ratio(0.70, 0.0)


class TestAssessSurrogateConsistency:
    """End-to-end honest verdict: directional consistency, magnitude gap, caveats."""

    def _run(self, data_slope=1.13):
        return assess_surrogate_consistency(
            ht_relative_modulus_slope=0.699,
            data_slope_raw_mpa=data_slope,
            data_slope_ci=(0.6, 1.7),
            mean_strength_mpa=62.0,
            orientation="in-plane-random",
            prior="flat",
        )

    def test_reports_directional_agreement_when_both_positive(self):
        out = self._run()
        assert out["trend_sign_agreement"] is True

    def test_flags_credible_interval_excludes_zero(self):
        out = self._run()
        assert out["data_slope_excludes_zero"] is True

    def test_magnitude_gap_reflects_modulus_not_strength(self):
        # HT modulus slope (~0.70/wt%) hugely exceeds the relative strength slope
        # (~0.018/wt%) — the ratio must be large, demonstrating modulus != strength.
        out = self._run()
        assert out["magnitude_ratio_modulus_to_strength"] > 10.0

    def test_records_caveats_and_independence(self):
        out = self._run()
        assert "modulus" in " ".join(out["caveats"]).lower()
        assert out["independent_of_tensile_data"] is True
        assert out["prior"] == "flat"

    def test_disagreement_when_data_slope_negative(self):
        out = self._run(data_slope=-0.5)
        assert out["trend_sign_agreement"] is False

    def test_ci_crossing_zero_flagged(self):
        out = assess_surrogate_consistency(
            ht_relative_modulus_slope=0.699,
            data_slope_raw_mpa=1.13,
            data_slope_ci=(-0.2, 2.4),
            mean_strength_mpa=62.0,
            orientation="in-plane-random",
            prior="flat",
        )
        assert out["data_slope_excludes_zero"] is False

    def test_rejects_informed_prior_for_circularity(self):
        # The cross-check must use the flat-prior model; informed prior would be circular.
        with pytest.raises(ValueError):
            assess_surrogate_consistency(
                ht_relative_modulus_slope=0.699,
                data_slope_raw_mpa=1.13,
                data_slope_ci=(0.6, 1.7),
                mean_strength_mpa=62.0,
                orientation="in-plane-random",
                prior="informed",
            )
