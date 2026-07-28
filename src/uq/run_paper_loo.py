"""
src/uq/run_paper_loo.py — honest leave-one-PAPER-out (LOPO) evaluation on the strict n=26 set.

The strict PLA+FDM+D638 corpus is drawn from only 3 source papers, so leave-one-ROW-out
LOO keeps same-paper twins in the training fold and overstates cross-study generalization.
This driver re-runs both methods holding out one whole PAPER at a time (groups=source_paper),
the honest robustness stress test, and writes results/phase_paper_loo.json alongside the
canonical row-LOO files (which are left untouched).

Also reports the per-fold NLPD standard error so the bootstrap-vs-Bayesian NLPD gap can be
read as a statistical tie rather than a difference.

Run:  /Users/jimbo/miniconda3/bin/python -m src.uq.run_paper_loo
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

from src.uq.data_loader import load_dataset, filter_pla_fdm
from src.uq.bootstrap import run_bootstrap_pipeline
from src.uq.bayesian import run_bayesian_loo

_log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
STRICT_CSV = ROOT / "data" / "extraction" / "master_dataset_strict.csv"
OUT = ROOT / "results" / "phase_paper_loo.json"

# Match the canonical row-LOO drivers for like-for-like comparison.
BOOT_SEED = 42
BAYES_SEED = 20260620


def _nlpd_per_fold(res: dict) -> np.ndarray:
    yt = np.asarray(res["loo_y_true"])
    m = np.asarray(res["loo_pred_mean"])
    s = np.asarray(res["loo_pred_std"])
    return -stats.norm.logpdf(yt, loc=m, scale=s)


def _per_paper_coverage(res: dict, papers: np.ndarray) -> dict:
    yt = np.asarray(res["loo_y_true"])
    lo = np.asarray(res["loo_lower_90"])
    up = np.asarray(res["loo_upper_90"])
    covered = (yt >= lo) & (yt <= up)
    out = {}
    for p in np.unique(papers):
        m = papers == p
        out[str(p)] = {"n": int(m.sum()), "coverage_90": float(covered[m].mean())}
    return out


def _summ(res: dict, papers: np.ndarray) -> dict:
    pf = _nlpd_per_fold(res)
    return {
        "eval_scheme": res["eval_scheme"],
        "n_groups": res["n_groups"],
        "picp_90": res["picp_90"],
        "picp_95": res["picp_95"],
        "mpiw_90": res["mpiw_90"],
        "crps": res["crps"],
        "nlpd": res["nlpd"],
        "nlpd_se": float(pf.std(ddof=1) / np.sqrt(len(pf))),
        "ece": res["ece"],
        "per_paper_coverage_90": _per_paper_coverage(res, papers),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = filter_pla_fdm(load_dataset(STRICT_CSV))
    x = df["cnt_wt_pct"].values.astype(float)
    y = df["tensile_mpa"].values.astype(float)
    papers = df["source_paper"].values
    n_papers = len(np.unique(papers))
    _log.info("Strict set n=%d from %d papers. Running leave-one-PAPER-out…", len(x), n_papers)

    boot = run_bootstrap_pipeline(
        x, y, B=1000, alpha=0.10, seed=BOOT_SEED, eval_mode="loo", groups=papers
    )
    bay = run_bayesian_loo(
        x, y, draws=1000, tune=1000, chains=4, degree=1, seed=BAYES_SEED, groups=papers
    )

    # Paired NLPD comparison on the same held-out points (both aligned by global index).
    diff = _nlpd_per_fold(boot) - _nlpd_per_fold(bay)
    paired = {
        "bootstrap_minus_bayesian_mean": float(diff.mean()),
        "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "note": "|mean| < ~2*se => statistical tie",
    }

    boot_summary = _summ(boot, papers)
    bay_summary = _summ(bay, papers)
    out = {
        "dataset": "master_dataset_strict.csv (PLA+FDM+D638)",
        "n": len(x),
        "n_papers": n_papers,
        "eval_scheme": "leave-one-paper-out",
        "bootstrap_paper_loo": boot_summary,
        "bayesian_paper_loo": bay_summary,
        "nlpd_paired": paired,
    }
    OUT.write_text(json.dumps(out, indent=1))
    _log.info("Wrote %s", OUT)
    for name, s in (("Bootstrap", boot_summary), ("Bayesian", bay_summary)):
        _log.info("%s LOPO  PICP@90=%.3f  MPIW@90=%.1f  NLPD=%.3f±%.3f  ECE=%.3f",
                  name, s["picp_90"], s["mpiw_90"], s["nlpd"], s["nlpd_se"], s["ece"])
    _log.info("Paired NLPD (boot-bayes) = %.3f ± %.3f  → %s",
              paired["bootstrap_minus_bayesian_mean"], paired["se"], paired["note"])


if __name__ == "__main__":
    main()
