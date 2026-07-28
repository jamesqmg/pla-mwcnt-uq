"""Phase 2c — per-paper random-intercept hierarchical model (§3.5b expansion).

Re-runs the n=44 pool with the Option B slope pooling AND a per-paper random
intercept (6 papers with complete (wt%, strength) pairs). Reports tau_alpha, the
between-study baseline-strength variation that the intercept absorbs — the answer
to the "blunt binary grouping" and study-heterogeneity critiques.

Writes results/hierarchical_phase2c_paper_intercept.json. Never overwrites
canonical results (hierarchical_phase2b.json is left untouched).

Run:  python -m src.uq.run_paper_intercept
"""
from __future__ import annotations

import json
from pathlib import Path

from src.uq.hierarchical import load_hierarchical_data_with_papers, run_hierarchical_v2_pipeline

ROOT = Path(__file__).parents[2]
MASTER_CSV = ROOT / "data" / "extraction" / "master_dataset.csv"
OUT_JSON = ROOT / "results" / "hierarchical_phase2c_paper_intercept.json"

SEED = 42
DRAWS, TUNE, CHAINS = 1000, 1000, 4


def main() -> dict:
    x, y, group_idx, paper_idx, paper_names = load_hierarchical_data_with_papers(MASTER_CSV)
    res = run_hierarchical_v2_pipeline(
        x, y, group_idx, paper_idx, n_groups=2, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED
    )
    conv = res["convergence"]
    beta_g = res["beta_g_posterior_mean"]  # (n_groups, n_coef): [:,1] = group slopes

    payload = {
        "description": "Phase 2c hierarchical model: Option B slope pooling + per-paper random intercept.",
        "n": res["n"],
        "n_groups": res["n_groups"],
        "n_papers": res["n_papers"],
        "papers": paper_names,
        "note": "7th source (sankar_2025) contributed a single row excluded for missing data; "
                "6 papers carry complete (wt%, strength) pairs in the n=44 pool.",
        "seed": SEED,
        "sampler": {"draws": DRAWS, "tune": TUNE, "chains": CHAINS, "target_accept": 0.99},
        "tau_alpha_posterior_mean": res["tau_alpha_posterior_mean"],
        "tau_beta_posterior_mean": res["tau_beta_posterior_mean"],
        "beta_g_slope_fdm": float(beta_g[0, 1]),
        "beta_g_slope_non_fdm": float(beta_g[1, 1]),
        "alpha_paper_posterior_mean": {
            name: float(a) for name, a in zip(paper_names, res["alpha_paper_posterior_mean"])
        },
        "picp_90": res["picp_90"],
        "picp_95": res["picp_95"],
        "mpiw_90_mpa": res["mpiw_90"],
        "crps": res["crps"],
        "convergence": {
            "rhat_max": conv["rhat_max"],
            "ess_min": conv["ess_min"],
            "passed": conv["passed"],
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    out = main()
    print(f"Wrote {OUT_JSON}")
    print(f"converged={out['convergence']['passed']} "
          f"rhat={out['convergence']['rhat_max']:.4f} ess={out['convergence']['ess_min']:.0f}")
    print(f"tau_alpha={out['tau_alpha_posterior_mean']:.4f}  tau_beta={out['tau_beta_posterior_mean']:.4f}")
    print(f"slope FDM={out['beta_g_slope_fdm']:.3f}  non-FDM={out['beta_g_slope_non_fdm']:.3f}")
    print(f"PICP90={out['picp_90']:.3f}  MPIW90={out['mpiw_90_mpa']:.2f}  CRPS={out['crps']:.2f}")
    print("alpha_paper:", {k: round(v, 3) for k, v in out["alpha_paper_posterior_mean"].items()})
