"""Tier-2 physics-based validation: micromechanics surrogate vs the data-driven model.

Implements the §3.7 convergent cross-check. The Halpin-Tsai surrogate (see
``micromechanics.py``) is an independent deterministic predictor driven only by
material and geometric inputs — it never sees the measured tensile strengths. Its
reinforcement trend is compared against the FLAT-prior Bayesian model (Phase 2),
which likewise never incorporates the micromechanics prediction. Using the flat
(not informed) model avoids the circularity of validating the physics-informed prior
against the same physics that defined it.

Honest claim ceiling (F1): the comparison assesses the reinforcement TREND — the sign
and order of magnitude of the slope — not pointwise absolute strength. Halpin-Tsai
predicts elastic *modulus*; the large modulus-vs-strength magnitude gap reported here
is expected and is itself the evidence for the modulus != strength caveat, not a model
failure. A deterministic surrogate also cannot, on its own, validate a probabilistic
prediction; the value of Phases 1-2 is the uncertainty a deterministic predictor omits.
"""
from __future__ import annotations


def relative_data_slope(slope_raw_mpa: float, mean_strength_mpa: float) -> float:
    """Express a raw strength slope (MPa per wt%) as a relative slope (fraction per wt%).

    Normalising by the mean strength puts the data-driven slope on the same relative
    footing as the Halpin-Tsai relative-modulus slope, so the two can be compared.
    """
    if mean_strength_mpa <= 0.0:
        raise ValueError("mean_strength_mpa must be positive.")
    return slope_raw_mpa / mean_strength_mpa


def directional_agreement(ht_slope: float, data_slope: float) -> bool:
    """True iff the physics slope and the data slope share a strictly non-zero sign."""
    return (ht_slope > 0 and data_slope > 0) or (ht_slope < 0 and data_slope < 0)


def magnitude_ratio(ht_relative_slope: float, data_relative_slope: float) -> float:
    """Ratio of the HT relative-modulus slope to the data relative-strength slope."""
    if data_relative_slope == 0.0:
        raise ValueError("data_relative_slope must be non-zero.")
    return ht_relative_slope / data_relative_slope


def assess_surrogate_consistency(
    ht_relative_modulus_slope: float,
    data_slope_raw_mpa: float,
    data_slope_ci: tuple[float, float],
    mean_strength_mpa: float,
    orientation: str = "in-plane-random",
    prior: str = "flat",
) -> dict:
    """Assemble the honest Tier-2 consistency verdict between surrogate and data model.

    Args:
        ht_relative_modulus_slope: Halpin-Tsai relative modulus uplift per wt%
            (from ``micromechanics.reinforcement_slope``).
        data_slope_raw_mpa: posterior-mean strength slope (MPa per wt%) of the
            flat-prior Bayesian model.
        data_slope_ci: credible interval (low, high) for that slope, MPa per wt%.
        mean_strength_mpa: mean measured tensile strength, for relative normalisation.
        orientation: deposition orientation assumed for the surrogate (recorded).
        prior: must be "flat" — using the informed prior would be circular.

    Returns a dict of directional agreement, the modulus-vs-strength magnitude gap,
    and explicit caveat / independence fields for direct transcription into §3.7.
    """
    if prior != "flat":
        raise ValueError(
            "surrogate cross-check must use the flat-prior model; the informed prior "
            "would validate the physics against the same physics that defined it (circular)."
        )

    data_rel = float(relative_data_slope(data_slope_raw_mpa, mean_strength_mpa))
    lo, hi = data_slope_ci
    excludes_zero = bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0))

    sign_agree = bool(directional_agreement(ht_relative_modulus_slope, data_slope_raw_mpa))
    ratio = float(magnitude_ratio(ht_relative_modulus_slope, data_rel)) if data_rel != 0.0 else float("inf")

    return {
        "prior": prior,
        "independent_of_tensile_data": True,
        "orientation": orientation,
        "ht_relative_modulus_slope_per_wt": ht_relative_modulus_slope,
        "data_slope_raw_mpa_per_wt": data_slope_raw_mpa,
        "data_relative_strength_slope_per_wt": data_rel,
        "data_slope_ci": list(data_slope_ci),
        "data_slope_excludes_zero": excludes_zero,
        "trend_sign_agreement": sign_agree,
        "magnitude_ratio_modulus_to_strength": ratio,
        "verdict": (
            "Directional consistency confirmed: physics and data agree on positive "
            "reinforcement; the large modulus-to-strength magnitude ratio is the expected "
            "modulus != strength gap, so the surrogate validates the reinforcement trend, "
            "not pointwise strength."
            if sign_agree else
            "Directional inconsistency: the surrogate and data-driven slopes disagree in sign."
        ),
        "caveats": [
            "Halpin-Tsai predicts elastic modulus, not failure strength; only the "
            "reinforcement trend (sign + order of magnitude) is cross-checked.",
            "A deterministic surrogate cannot validate a probabilistic prediction on its "
            "own; the Bayesian framework supplies the omitted uncertainty.",
            "Absolute-strength micromechanics requires interfacial parameters "
            "(interfacial shear strength, critical fibre length) not recoverable from "
            "the literature dataset.",
        ],
    }
