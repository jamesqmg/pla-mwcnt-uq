"""
src/uq/run_bayesian_loo.py — honest out-of-sample Bayesian LOO on the strict n=26 set.

Produces results/phase2_bayesian_loo.json: LOO PICP/MPIW/CRPS/NLPD/ECE computed from
exact posterior-predictive quantiles, plus the reliability curve and per-point
predictions for the parity/reliability figures. Replaces the in-sample Bayesian
evaluation (PICP=1.000) and the z-ratio reliability approximation (reviews C4 / P#6).

Usage:
    /Users/jimbo/miniconda3/bin/python -m src.uq.run_bayesian_loo
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.uq.data_loader import load_dataset, filter_pla_fdm
from src.uq.bayesian import run_bayesian_loo

_log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
STRICT_CSV = ROOT / "data" / "extraction" / "master_dataset_strict.csv"
OUT = ROOT / "results" / "phase2_bayesian_loo.json"

SEED = 20260620


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = filter_pla_fdm(load_dataset(STRICT_CSV))
    x = df["cnt_wt_pct"].values.astype(float)
    y = df["tensile_mpa"].values.astype(float)
    _log.info("Strict set n=%d (PLA+FDM+D638). Running Bayesian LOO (draws=1000, chains=4)…", len(x))

    res = run_bayesian_loo(x, y, draws=1000, tune=1000, chains=4, degree=1, seed=SEED)

    OUT.write_text(json.dumps(res, indent=1))
    _log.info("Wrote %s", OUT)
    _log.info("LOO PICP@90=%.3f  MPIW@90=%.1f  NLPD=%.3f  ECE=%.3f  CRPS=%.2f",
              res["picp_90"], res["mpiw_90"], res["nlpd"], res["ece"], res["crps"])
    _log.info("Reliability curve: %s",
              ", ".join(f"{nom:.2f}->{emp:.3f}" for nom, emp in res["reliability_curve"]))


if __name__ == "__main__":
    main()
