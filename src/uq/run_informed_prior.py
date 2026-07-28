"""Phase 2 expansion — Halpin-Tsai informed-prior sensitivity analysis (§3.5).

Runs the n=26 PLA+FDM+D638 Bayesian regression twice under identical sampler
settings and seed:

  * flat   — Normal(0, 1) on the CNT slope (the primary, validated reference);
  * informed — a physics-anchored prior centred at a +2 MPa/wt% reinforcement
    slope, with dispersion set so the prior 90% interval spans 0 -> 2x the mean
    (F2 default), keeping the n=26 likelihood dominant.

The informed prior is a *separate sensitivity analysis*: the flat-prior model
remains the primary reported result and the model carried into the Phase 4
micromechanics cross-check, avoiding circularity (it never sees the HT prediction).

Writes results/phase2_informed_prior.json. Never overwrites canonical results.

Run:  python -m src.uq.run_informed_prior
"""
from __future__ import annotations

import json
from pathlib import Path

from src.uq.bayesian import run_bayesian_pipeline, standardize, standardized_slope_prior
from src.uq.data_loader import filter_pla_fdm, load_dataset

ROOT = Path(__file__).parents[2]
STRICT_CSV = ROOT / "data" / "extraction" / "master_dataset_strict.csv"
OUT_JSON = ROOT / "results" / "phase2_informed_prior.json"

# Informed-prior specification (raw strength-slope units, MPa per wt%).
INFORMED_SLOPE_MEAN_MPA = 2.0          # user-confirmed: positive, single-digit (sign + OOM only)
F2_Z90 = 1.645                          # 90% interval spans 0 -> 2*mean  =>  sigma = mean / z90
SEED = 42
DRAWS, TUNE, CHAINS = 1000, 1000, 4


def _summary(result: dict) -> dict:
    """Extract the side-by-side comparison metrics from a pipeline result."""
    conv = result["convergence"]
    return {
        "slope_raw_mpa_per_wt": result["slope_raw_mean"],
        "slope_std_mean": result["slope_std_mean"],
        "slope_std_sd": result["slope_std_sd"],
        "picp_90_insample": result["picp_90"],
        "mpiw_90_mpa": result["mpiw_90"],
        "crps": result["crps"],
        "nlpd_psis_loo": result["nlpd"],
        "ece": result["ece"],
        "rhat_max": conv["rhat_max"],
        "ess_min": conv["ess_min"],
        "converged": conv["passed"],
    }


def main() -> dict:
    df = filter_pla_fdm(load_dataset(STRICT_CSV))
    x = df["cnt_wt_pct"].values.astype(float)
    y = df["tensile_mpa"].values.astype(float)

    _, _, params = standardize(x, y)
    sigma_raw = INFORMED_SLOPE_MEAN_MPA / F2_Z90
    slope_prior_std = standardized_slope_prior(INFORMED_SLOPE_MEAN_MPA, sigma_raw, params)

    flat = run_bayesian_pipeline(x, y, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED)
    informed = run_bayesian_pipeline(
        x, y, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED, slope_prior=slope_prior_std
    )

    payload = {
        "description": "Phase 2 Halpin-Tsai informed-prior sensitivity (flat-prior model remains primary).",
        "n": int(len(x)),
        "seed": SEED,
        "sampler": {"draws": DRAWS, "tune": TUNE, "chains": CHAINS},
        "informed_prior_spec": {
            "slope_mean_mpa_per_wt": INFORMED_SLOPE_MEAN_MPA,
            "slope_sd_mpa_per_wt": sigma_raw,
            "standardized_mu": slope_prior_std[0],
            "standardized_sigma": slope_prior_std[1],
            "dispersion_rule": "90% interval spans 0 -> 2x mean (F2)",
            "scope_caveat": "Informs sign + order of magnitude of the reinforcement slope only; "
                            "Halpin-Tsai predicts modulus, not strength (F1).",
        },
        "flat_prior": _summary(flat),
        "informed_prior": _summary(informed),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    out = main()
    f, i = out["flat_prior"], out["informed_prior"]
    print(f"Wrote {OUT_JSON}")
    print(f"{'metric':<22}{'flat':>14}{'informed':>14}")
    for k in ("slope_raw_mpa_per_wt", "slope_std_mean", "mpiw_90_mpa", "nlpd_psis_loo", "ece",
              "rhat_max", "ess_min"):
        print(f"{k:<22}{f[k]:>14.4f}{i[k]:>14.4f}")
    print(f"converged: flat={f['converged']}  informed={i['converged']}")
