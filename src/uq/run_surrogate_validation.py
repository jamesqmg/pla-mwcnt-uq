"""Phase 4 Tier-2 cross-check — Halpin-Tsai surrogate vs the flat-prior model (§3.7).

Pulls the flat-prior posterior slope from results/phase2_informed_prior.json, computes
the independent Halpin-Tsai relative-modulus reinforcement slope from the dataset
geometry, and writes the honest consistency verdict to results/surrogate_validation.json.

The surrogate uses only material/geometric inputs and never sees the tensile strengths;
it is compared against the flat-prior (not informed) model to keep the check independent.

Run:  python -m src.uq.run_surrogate_validation
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.uq.bayesian import standardize
from src.uq.data_loader import filter_pla_fdm, load_dataset, parse_numeric_column
from src.uq.micromechanics import aspect_ratio, reinforcement_slope
from src.uq.surrogate_validation import assess_surrogate_consistency

ROOT = Path(__file__).parents[2]
STRICT_CSV = ROOT / "data" / "extraction" / "master_dataset_strict.csv"
MASTER_CSV = ROOT / "data" / "extraction" / "master_dataset.csv"
PHASE2_JSON = ROOT / "results" / "phase2_informed_prior.json"
OUT_JSON = ROOT / "results" / "surrogate_validation.json"

Z90 = 1.645
ORIENTATION = "in-plane-random"  # FDM raster deposits fibres in-plane


def _ht_slope() -> float:
    """Halpin-Tsai relative-modulus reinforcement slope from the graded da Silva PLA series."""
    full = pd.read_csv(MASTER_CSV)
    sub = full[full["source_paper"] == "dasilva_2025_polymers"]
    ef = parse_numeric_column(sub["CNT Young's Modulus (TPa)"]).iloc[0] * 1000.0  # TPa -> GPa
    em = parse_numeric_column(sub["Matrix Young's Modulus (GPa)"]).iloc[0]
    length = parse_numeric_column(sub["CNT Length (um)"]).iloc[0]
    diam = parse_numeric_column(sub["CNT Diameter (nm)"]).iloc[0]
    wt = parse_numeric_column(sub["CNT Weight Fraction (%)"]).to_numpy(dtype=float)
    ar = aspect_ratio(length, diam)
    return reinforcement_slope(ef, em, ar, wt, ORIENTATION)


def main() -> dict:
    df = filter_pla_fdm(load_dataset(STRICT_CSV))
    x = df["cnt_wt_pct"].values.astype(float)
    y = df["tensile_mpa"].values.astype(float)
    _, _, params = standardize(x, y)
    scale = params["y_std"] / params["x_std"]

    flat = json.loads(PHASE2_JSON.read_text())["flat_prior"]
    raw_mean = float(flat["slope_std_mean"] * scale)
    raw_sd = float(flat["slope_std_sd"] * scale)
    ci = (raw_mean - Z90 * raw_sd, raw_mean + Z90 * raw_sd)

    ht_slope = float(_ht_slope())
    verdict = assess_surrogate_consistency(
        ht_relative_modulus_slope=ht_slope,
        data_slope_raw_mpa=raw_mean,
        data_slope_ci=ci,
        mean_strength_mpa=float(np.mean(y)),
        orientation=ORIENTATION,
        prior="flat",
    )
    verdict["n"] = int(len(x))
    verdict["mean_strength_mpa"] = float(np.mean(y))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(verdict, indent=2))
    return verdict


if __name__ == "__main__":
    out = main()
    print(f"Wrote {OUT_JSON}")
    print(f"HT relative-modulus slope : {out['ht_relative_modulus_slope_per_wt']:.4f} /wt%")
    print(f"data strength slope       : {out['data_slope_raw_mpa_per_wt']:.4f} MPa/wt% "
          f"(CI {out['data_slope_ci'][0]:.3f}..{out['data_slope_ci'][1]:.3f})")
    print(f"data relative slope       : {out['data_relative_strength_slope_per_wt']:.4f} /wt%")
    print(f"trend sign agreement      : {out['trend_sign_agreement']}")
    print(f"data CI excludes zero     : {out['data_slope_excludes_zero']}")
    print(f"modulus:strength ratio    : {out['magnitude_ratio_modulus_to_strength']:.1f}x")
    print(f"verdict: {out['verdict']}")
