"""
src/uq/run_subsampling_refit.py — reproducible subsampling refit runner.

Replaces the missing one-off generator for results/subsampling_*.json. Bumps the
bootstrap n-ladder to 500 repeats so the specimen-budget inversion (§4.4) stops
showing the n=5→28 cliff that was a low-repeat variance artifact (reviews P#1/#7).

Modes
-----
    bootstrap : 500-repeat bootstrap across the full n-ladder (cheap; skip Bayesian)
                → results/subsampling_bootstrap.json
    bayesian  : Bayesian NUTS at key n only (expensive)
                → results/subsampling_bayesian_key.json
    merge     : union bootstrap + bayesian-key by n; recompute inversion table
                → results/subsampling_combined.json, results/inversion_table.json

Usage
-----
    PY=/Users/jimbo/miniconda3/bin/python
    $PY -m src.uq.run_subsampling_refit bootstrap --repeats 500
    $PY -m src.uq.run_subsampling_refit bayesian  --repeats 20
    $PY -m src.uq.run_subsampling_refit merge

Run modes in that order; merge consumes the two upstream JSONs.
"""
from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.uq.subsampling import (
    prepare_parent_dataset,
    build_n_ladder,
    run_subsampling_experiment,
)
from src.uq.inversion import compute_inversion_table

_log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PARENT_CSV = ROOT / "data" / "extraction" / "master_dataset.csv"
RESULTS = ROOT / "results"

# n-ladder preserved from the original combined output (keeps n=26 case-study point)
LADDER_VALUES = [5, 8, 10, 15, 20, 26, 30, 35]  # max_n appended at runtime
BAYESIAN_KEY_N = [5, 10, 20, 26, 44]

BOOTSTRAP_JSON = RESULTS / "subsampling_bootstrap.json"
BAYESIAN_JSON = RESULTS / "subsampling_bayesian_key.json"
COMBINED_JSON = RESULTS / "subsampling_combined.json"
INVERSION_JSON = RESULTS / "inversion_table.json"

# Inversion targets — tightened grid around the coverage region of interest so the
# refit produces a smooth, defensible budget curve rather than a sparse 3-point table.
PICP_TARGETS = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
MPIW_TARGETS = [55.0, 50.0, 45.0, 40.0, 35.0, 30.0]

SEED = 20260620


def load_parent() -> tuple:
    """Load the unfiltered parent pool (n≈44) used as the n-ladder source."""
    df = pd.read_csv(PARENT_CSV)
    x, y = prepare_parent_dataset(df)
    return x, y


def merge_subsampling_results(
    bootstrap_records: list[dict],
    bayesian_records: list[dict],
) -> list[dict]:
    """Union bootstrap-only records with bayesian-key records, keyed by n.

    Bootstrap defines the ladder; bayesian metrics are folded into the matching n.
    Returns a new list sorted ascending by n; inputs are not mutated.
    """
    merged = {r["n"]: deepcopy(r) for r in bootstrap_records}
    boot_ns = set(merged)
    for b in bayesian_records:
        n = b["n"]
        if n not in boot_ns:
            raise ValueError(
                f"Bayesian record n={n} not present in bootstrap ladder {sorted(boot_ns)}."
            )
        # Fold in ONLY bayesian fields — never let a bayesian record's (possibly
        # stale) bootstrap fields overwrite the fresh bootstrap curve.
        bay_only = {k: v for k, v in b.items() if "bayesian" in k}
        merged[n].update(deepcopy(bay_only))
    return [merged[n] for n in sorted(merged)]


def run_bootstrap_mode(repeats: int, B: int) -> None:
    x, y = load_parent()
    ladder = build_n_ladder(len(x), values=LADDER_VALUES + [len(x)])
    _log.info("Bootstrap refit: ladder=%s repeats=%d B=%d", ladder, repeats, B)
    records = run_subsampling_experiment(
        x, y, n_ladder=ladder, B=B, n_repeats=repeats,
        seed=SEED, skip_bayesian=True, degree=1,
    )
    BOOTSTRAP_JSON.write_text(json.dumps(records, indent=1))
    _log.info("Wrote %s", BOOTSTRAP_JSON)


def run_bayesian_mode(repeats: int, draws: int, tune: int, chains: int) -> None:
    x, y = load_parent()
    ladder = [n for n in BAYESIAN_KEY_N if n <= len(x)]
    _log.info("Bayesian refit: key n=%s repeats=%d", ladder, repeats)
    records = run_subsampling_experiment(
        x, y, n_ladder=ladder, B=1, n_repeats=repeats, seed=SEED,
        skip_bayesian=False, draws=draws, tune=tune, chains=chains, degree=1,
    )
    # keep only bayesian fields + n (bootstrap here runs with B=1 as a throwaway)
    slim = []
    for r in records:
        keep = {"n": r["n"]}
        keep.update({k: v for k, v in r.items() if "bayesian" in k})
        slim.append(keep)
    BAYESIAN_JSON.write_text(json.dumps(slim, indent=1))
    _log.info("Wrote %s", BAYESIAN_JSON)


def run_merge_mode() -> None:
    boot = json.loads(BOOTSTRAP_JSON.read_text())
    bayes = json.loads(BAYESIAN_JSON.read_text()) if BAYESIAN_JSON.exists() else []
    combined = merge_subsampling_results(boot, bayes)
    COMBINED_JSON.write_text(json.dumps(combined, indent=1))
    _log.info("Wrote %s (%d n-values)", COMBINED_JSON, len(combined))

    table = compute_inversion_table(
        COMBINED_JSON, picp_targets=PICP_TARGETS, mpiw_targets=MPIW_TARGETS, alpha="90",
    )
    INVERSION_JSON.write_text(json.dumps(table, indent=1))
    _log.info("Wrote %s", INVERSION_JSON)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["bootstrap", "bayesian", "merge"])
    p.add_argument("--repeats", type=int, default=500)
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    args = p.parse_args(argv)

    if args.mode == "bootstrap":
        run_bootstrap_mode(repeats=args.repeats, B=args.B)
    elif args.mode == "bayesian":
        run_bayesian_mode(repeats=args.repeats, draws=args.draws,
                          tune=args.tune, chains=args.chains)
    else:
        run_merge_mode()


if __name__ == "__main__":
    main()
