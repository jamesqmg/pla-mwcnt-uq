"""
src/uq/run_estcon_revision.py — analyses for the ESTCON 2026 proceedings revision.

Answers the reviewer's comments on the ICES paper with numbers, not prose:

    degree : LOO-CV error of polynomial degree 1-3 on the strict n = 21 set (which
             backbone the paper may claim).
    loo    : per-fold LOO predictive samples for both methods on the corrected strict set
             (n = 21), with paired per-fold NLPD / CRPS differences, bootstrap
             intervals and Clopper-Pearson intervals on every PICP (comments 2, 3).
    hier   : hierarchical slopes with credible intervals on both slopes, their difference
             and ratio; prior sensitivity; the per-paper-intercept model and the
             route-by-paper crosstab that states the confound (comments 5, 6).
    budget : specimen budget on the FDM-only rows, scored on HELD-OUT rows (fit on n,
             test on the rest), ladder from n = 10, coverage at several nominal levels
             so a calibration surface can be drawn (comments 4, 7).

Outputs: results/estcon_revision_<mode>.json (+ .npz for posterior draws).

Usage:
    python -m src.uq.run_estcon_revision <mode>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.uq.bayesian import (
    _posterior_predictive_samples,
    build_model,
    run_bayesian_pipeline,
    run_nuts,
    standardize,
)
from src.uq.bootstrap import (
    _bootstrap_sample_matrix,
    compute_crps,
    fit_regression,
    run_bootstrap_pipeline,
)
from src.uq.data_loader import filter_pla_fdm, load_dataset
from src.uq.hierarchical import (
    GROUP_FDM,
    build_hierarchical_model,
    build_hierarchical_model_v2,
    check_convergence_all,
    load_hierarchical_data,
    load_hierarchical_data_with_papers,
    run_nuts_timed,
)

_log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
STRICT_CSV = ROOT / "data" / "extraction" / "master_dataset_strict.csv"
POOL_CSV = ROOT / "data" / "extraction" / "master_dataset.csv"
RESULTS = ROOT / "results"

BOOT_SEED = 42          # seed behind results/phase1_phase2_loo.json (bootstrap LOO)
BAYES_SEED = 20260620   # seed behind results/phase2_bayesian_loo.json
HIER_SEED = 42
LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


# ── small statistics helpers ─────────────────────────────────────────────────

def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial interval for k successes in n trials."""
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def paired_summary(d: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Mean of paired differences, its SE, a percentile bootstrap 95% CI and a t-test."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    t = stats.ttest_1samp(d, 0.0)
    return {
        "mean": float(d.mean()),
        "se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "ci95_boot": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "t_pvalue": float(t.pvalue),
    }


def interval_summary(draws: np.ndarray) -> dict:
    """Posterior mean, SD, 95% equal-tailed interval."""
    draws = np.asarray(draws, dtype=float)
    return {
        "mean": float(draws.mean()),
        "sd": float(draws.std(ddof=1)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def coverage_at_levels(samples: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row coverage indicator at each nominal level. samples: (n_rows, n_draws)."""
    out = {}
    for lv in LEVELS:
        lo = np.quantile(samples, (1 - lv) / 2, axis=1)
        hi = np.quantile(samples, 1 - (1 - lv) / 2, axis=1)
        out[str(lv)] = (y >= lo) & (y <= hi)
    return out


def load_strict() -> tuple[np.ndarray, np.ndarray]:
    df = filter_pla_fdm(load_dataset(STRICT_CSV))
    return df["cnt_wt_pct"].to_numpy(float), df["tensile_mpa"].to_numpy(float)


def write(name: str, payload: dict) -> None:
    path = RESULTS / f"estcon_revision_{name}.json"
    path.write_text(json.dumps(payload, indent=1))
    _log.info("Wrote %s", path)


# ── mode: degree ─────────────────────────────────────────────────────────────

def run_degree() -> None:
    x, y = load_strict()
    out = {}
    for deg in (1, 2, 3):
        err = np.array([
            y[i] - fit_regression(np.delete(x, i), np.delete(y, i), degree=deg)
            .predict(x[[i]].reshape(-1, 1))[0]
            for i in range(len(x))
        ])
        out[str(deg)] = {"loo_rmse": float(np.sqrt(np.mean(err ** 2))),
                         "loo_mae": float(np.mean(np.abs(err)))}
        _log.info("degree %d  LOO RMSE %.3f MPa", deg, out[str(deg)]["loo_rmse"])
    write("degree", {"n": len(x), "by_degree": out})


# ── mode: loo ────────────────────────────────────────────────────────────────

def bootstrap_loo_samples(x: np.ndarray, y: np.ndarray, B: int = 1000) -> np.ndarray:
    """Re-draw the per-fold bootstrap predictive samples exactly as run_bootstrap_pipeline
    does under eval_mode='loo' (same SeedSequence spawning), keeping the samples."""
    n = len(x)
    ss = np.random.SeedSequence(BOOT_SEED)
    ss.spawn(1)                       # consumed by the in-sample matrix in the pipeline
    child = ss.spawn(2 * n)
    b_i = max(B // 2, 100)
    rows = []
    for i in range(n):
        train = np.setdiff1d(np.arange(n), [i])
        rng = np.random.default_rng(child[i])
        rows.append(_bootstrap_sample_matrix(x[train], y[train], x[[i]], b_i, 1, rng, True)[:, 0])
    return np.vstack(rows)            # (n, b_i)


def bayesian_loo_samples(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-fold posterior-predictive samples, same seeding as run_bayesian_loo."""
    n = len(x)
    fold_seeds = np.random.SeedSequence(BAYES_SEED).spawn(n)
    rows = []
    for i in range(n):
        train = np.setdiff1d(np.arange(n), [i])
        x_std, y_std, params = standardize(x[train], y[train])
        x_held = (x[[i]] - params["x_mean"]) / params["x_std"]
        fseed = int(fold_seeds[i].generate_state(1)[0])
        idata = run_nuts(build_model(x_std, y_std, degree=1), draws=1000, tune=1000,
                         chains=4, seed=fseed)
        rows.append(_posterior_predictive_samples(idata, x_held, params, 1, seed=fseed)[0])
        _log.info("Bayesian LOO fold %d/%d", i + 1, n)
    n_post = min(len(r) for r in rows)
    return np.vstack([r[:n_post] for r in rows])


def per_fold_scores(samples: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    mean = samples.mean(axis=1)
    std = np.maximum(samples.std(axis=1, ddof=1), 1e-6)
    return {
        "nlpd": -stats.norm.logpdf(y, loc=mean, scale=std),
        "crps": np.array([compute_crps(y[[i]], samples[[i]]) for i in range(len(y))]),
        "mean": mean,
        "std": std,
    }


def method_metrics(samples: np.ndarray, y: np.ndarray) -> dict:
    """LOO metrics from per-fold predictive samples, same definitions as bootstrap.py."""
    sc = per_fold_scores(samples, y)
    cov = coverage_at_levels(samples, y)
    width = {lv: float(np.mean(np.quantile(samples, 1 - (1 - lv) / 2, axis=1)
                               - np.quantile(samples, (1 - lv) / 2, axis=1))) for lv in (0.9, 0.95)}
    picp = {k: float(v.mean()) for k, v in cov.items()}
    return {
        "picp": picp,
        "picp_cp95": {k: clopper_pearson(int(v.sum()), len(y)) for k, v in cov.items()},
        "picp_count": {k: int(v.sum()) for k, v in cov.items()},
        "mpiw_90": width[0.9], "mpiw_95": width[0.95],
        "crps": float(sc["crps"].mean()), "nlpd": float(sc["nlpd"].mean()),
        "ece": float(np.mean([abs(picp[str(lv)] - lv) for lv in LEVELS])),
    }


def run_loo() -> None:
    x, y = load_strict()
    n = len(y)
    _log.info("Strict set n=%d", n)
    s_boot = bootstrap_loo_samples(x, y)
    pipe = run_bootstrap_pipeline(x, y, B=1000, degree=1, seed=BOOT_SEED, eval_mode="loo")
    assert np.allclose(s_boot.mean(axis=1), pipe["loo_pred_mean"]), "bootstrap folds not reproduced"
    s_bay = bayesian_loo_samples(x, y)

    ins_boot = run_bootstrap_pipeline(x, y, B=1000, degree=1, seed=BOOT_SEED, eval_mode="bootstrap")
    ins_bay = run_bayesian_pipeline(x, y, draws=1000, tune=1000, chains=4, degree=1, seed=BAYES_SEED)

    sb, sy = per_fold_scores(s_boot, y), per_fold_scores(s_bay, y)
    cov_b, cov_y = coverage_at_levels(s_boot, y), coverage_at_levels(s_bay, y)
    discordant = {}
    for lv in LEVELS:
        b_only = int((cov_b[str(lv)] & ~cov_y[str(lv)]).sum())
        y_only = int((~cov_b[str(lv)] & cov_y[str(lv)]).sum())
        d = b_only + y_only
        discordant[str(lv)] = {"boot_only": b_only, "bayes_only": y_only,
                               "mcnemar_exact_p": float(stats.binomtest(b_only, d, 0.5).pvalue) if d else 1.0}

    payload = {
        "n": n, "studies": "Hardani 2024 (15) + Vidakis 2021 (6)", "degree": 1,
        "loo": {"bootstrap": method_metrics(s_boot, y), "bayesian": method_metrics(s_bay, y)},
        "in_sample": {
            "bootstrap": {"picp_90": ins_boot["picp_90"], "mpiw_90": ins_boot["mpiw_90"]},
            "bayesian": {"picp_90": ins_bay["picp_90"], "mpiw_90": ins_bay["mpiw_90"],
                         "rhat_max": ins_bay["convergence"]["rhat_max"],
                         "ess_min": ins_bay["convergence"]["ess_min"],
                         "prior_p5_mpa": ins_bay["prior_p5_mpa"], "prior_p95_mpa": ins_bay["prior_p95_mpa"],
                         "slope_raw_mean": ins_bay["slope_raw_mean"]},
        },
        "paired_boot_minus_bayes": {
            "nlpd": paired_summary(sb["nlpd"] - sy["nlpd"], seed=1),
            "crps": paired_summary(sb["crps"] - sy["crps"], seed=2),
        },
        "coverage_discordance": discordant,
        "per_fold": {
            "y": y.tolist(), "x": x.tolist(),
            "boot_mean": sb["mean"].tolist(), "boot_std": sb["std"].tolist(),
            "bayes_mean": sy["mean"].tolist(), "bayes_std": sy["std"].tolist(),
            "boot_nlpd": sb["nlpd"].tolist(), "bayes_nlpd": sy["nlpd"].tolist(),
            "boot_crps": sb["crps"].tolist(), "bayes_crps": sy["crps"].tolist(),
            "boot_lo90": np.quantile(s_boot, 0.05, axis=1).tolist(),
            "boot_hi90": np.quantile(s_boot, 0.95, axis=1).tolist(),
            "bayes_lo90": np.quantile(s_bay, 0.05, axis=1).tolist(),
            "bayes_hi90": np.quantile(s_bay, 0.95, axis=1).tolist(),
        },
    }
    write("loo", payload)
    np.savez_compressed(RESULTS / "estcon_revision_loo_samples.npz", boot=s_boot, bayes=s_bay, x=x, y=y)


# ── mode: hier ───────────────────────────────────────────────────────────────

def slope_draws(idata, params: dict) -> dict[str, np.ndarray]:
    """Slope draws per group, standardised and in MPa per wt%."""
    bg = idata.posterior["beta_g"].values.reshape(-1, 2, 2)  # (draws, group, coef)
    to_raw = params["y_std"] / params["x_std"]
    fdm, non = bg[:, GROUP_FDM, 1], bg[:, 1 - GROUP_FDM, 1]
    return {"fdm": fdm, "non_fdm": non, "fdm_raw": fdm * to_raw, "non_fdm_raw": non * to_raw}


def summarise_slopes(d: dict[str, np.ndarray]) -> dict:
    ratio = d["fdm"] / d["non_fdm"]
    return {
        "slope_fdm_std": interval_summary(d["fdm"]),
        "slope_non_fdm_std": interval_summary(d["non_fdm"]),
        "slope_fdm_mpa_per_wt": interval_summary(d["fdm_raw"]),
        "slope_non_fdm_mpa_per_wt": interval_summary(d["non_fdm_raw"]),
        "difference_std": interval_summary(d["fdm"] - d["non_fdm"]),
        "p_fdm_gt_non_fdm": float(np.mean(d["fdm"] > d["non_fdm"])),
        "p_non_fdm_slope_le_0": float(np.mean(d["non_fdm"] <= 0)),
        "ratio_of_means": float(d["fdm"].mean() / d["non_fdm"].mean()),
        "ratio_median": float(np.median(ratio)),
        "ratio_ci95": [float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))],
    }


def run_hier() -> None:
    x, y, g = load_hierarchical_data(POOL_CSV)
    x_std, y_std, params = standardize(x, y)
    out: dict = {"n": len(x), "fdm_n": int((g == GROUP_FDM).sum()), "non_fdm_n": int((g != GROUP_FDM).sum())}

    priors = {"base (beta_pop N(0,1), tau HalfN(0.5))": (1.0, 0.5),
              "tight tau HalfN(0.25)": (1.0, 0.25),
              "wide tau HalfN(1.0)": (1.0, 1.0),
              "wide beta_pop N(0,2), tau HalfN(1.0)": (2.0, 1.0)}
    out["prior_sensitivity"] = {}
    base_draws = None
    for label, (bs, ts) in priors.items():
        model = build_hierarchical_model(x_std, y_std, g, n_groups=2, degree=1,
                                         beta_pop_sigma=bs, tau_beta_sigma=ts)
        idata, _ = run_nuts_timed(model, seed=HIER_SEED)
        conv = check_convergence_all(idata)
        d = slope_draws(idata, params)
        out["prior_sensitivity"][label] = {**summarise_slopes(d),
                                           "rhat_max": conv["rhat_max"], "ess_min": conv["ess_min"]}
        _log.info("%s: FDM %.3f  non-FDM %.3f  ratio-of-means %.2f", label,
                  d["fdm"].mean(), d["non_fdm"].mean(), d["fdm"].mean() / d["non_fdm"].mean())
        if base_draws is None:
            base_draws = d
    out["base"] = out["prior_sensitivity"][next(iter(priors))]

    # Per-paper intercept model: does the route effect survive study baselines?
    x2, y2, g2, p2, names = load_hierarchical_data_with_papers(POOL_CSV)
    x2s, y2s, params2 = standardize(x2, y2)
    m2 = build_hierarchical_model_v2(x2s, y2s, g2, p2, n_groups=2, n_papers=len(names), degree=1)
    idata2, _ = run_nuts_timed(m2, seed=HIER_SEED, target_accept=0.99)
    out["paper_intercept"] = {**summarise_slopes(slope_draws(idata2, params2)),
                              "convergence": {k: v for k, v in check_convergence_all(idata2).items()
                                              if k in ("rhat_max", "ess_min", "passed")}}
    tab = pd.crosstab(pd.Series(names, dtype=str).iloc[p2].to_numpy(),
                      np.where(g2 == GROUP_FDM, "FDM", "non-FDM"))
    out["route_by_paper"] = {str(k): {str(c): int(v) for c, v in row.items()} for k, row in tab.iterrows()}
    write("hier", out)
    np.savez_compressed(RESULTS / "estcon_revision_hier_draws.npz",
                        fdm=base_draws["fdm"], non_fdm=base_draws["non_fdm"],
                        fdm_raw=base_draws["fdm_raw"], non_fdm_raw=base_draws["non_fdm_raw"])


# ── mode: budget ─────────────────────────────────────────────────────────────

def heldout_bootstrap(x_tr, y_tr, x_te, seed: int, B: int = 500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _bootstrap_sample_matrix(x_tr, y_tr, x_te, B, 1, rng, True).T  # (n_te, B)


def heldout_bayesian(x_tr, y_tr, x_te, seed: int) -> np.ndarray:
    x_std, y_std, params = standardize(x_tr, y_tr)
    x_te_std = (x_te - params["x_mean"]) / params["x_std"]
    idata = run_nuts(build_model(x_std, y_std, degree=1), draws=500, tune=500, chains=4, seed=seed)
    return _posterior_predictive_samples(idata, x_te_std, params, 1, seed=seed)


def run_budget(boot_repeats: int, bayes_repeats: int) -> None:
    x_all, y_all, g = load_hierarchical_data(POOL_CSV)
    x, y = x_all[g == GROUP_FDM], y_all[g == GROUP_FDM]
    N = len(x)
    boot_ladder = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    bayes_ladder = [10, 15, 20, 25, 30]
    ss = np.random.SeedSequence(20261107)

    def rung(n: int, repeats: int, method: str) -> dict:
        seeds = ss.spawn(repeats)
        hits = {str(lv): [] for lv in LEVELS}
        widths90, picp90_rep = [], []
        for r in range(repeats):
            s = int(seeds[r].generate_state(1)[0])
            idx = np.random.default_rng(s).permutation(N)
            tr, te = idx[:n], idx[n:]
            fn = heldout_bootstrap if method == "bootstrap" else heldout_bayesian
            samp = fn(x[tr], y[tr], x[te], s)
            cov = coverage_at_levels(samp, y[te])
            for lv in LEVELS:
                hits[str(lv)].append(cov[str(lv)])
            picp90_rep.append(float(cov["0.9"].mean()))
            widths90.append(float(np.mean(np.quantile(samp, 0.95, axis=1) - np.quantile(samp, 0.05, axis=1))))
        rec = {"n": n, "n_test_per_repeat": N - n, "repeats": repeats,
               "picp90_repeat_mean": float(np.mean(picp90_rep)),
               "picp90_repeat_sd": float(np.std(picp90_rep, ddof=1)),
               "mpiw90_mean": float(np.mean(widths90)), "mpiw90_sd": float(np.std(widths90, ddof=1)),
               "pooled": {}}
        for lv in LEVELS:
            h = np.concatenate(hits[lv_key := str(lv)])
            k, m = int(h.sum()), len(h)
            rec["pooled"][lv_key] = {"k": k, "m": m, "picp": k / m, "cp95": clopper_pearson(k, m)}
        _log.info("%s n=%d  held-out PICP90 %.3f  MPIW90 %.1f", method, n,
                  rec["pooled"]["0.9"]["picp"], rec["mpiw90_mean"])
        return rec

    out = {"pool": "FDM rows of master_dataset.csv", "N": N, "levels": list(LEVELS),
           "scoring": "fit on a random n-row subset, score prediction intervals on the remaining N-n rows",
           "bootstrap": [rung(n, boot_repeats, "bootstrap") for n in boot_ladder]}
    write("budget", out)  # checkpoint before the slow part
    out["bayesian"] = [rung(n, bayes_repeats, "bayesian") for n in bayes_ladder]
    write("budget", out)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("pymc", "pytensor", "arviz"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["degree", "loo", "hier", "budget"])
    p.add_argument("--boot-repeats", type=int, default=100)
    p.add_argument("--bayes-repeats", type=int, default=20)
    a = p.parse_args(argv)
    if a.mode == "degree":
        run_degree()
    elif a.mode == "loo":
        run_loo()
    elif a.mode == "hier":
        run_hier()
    else:
        run_budget(a.boot_repeats, a.bayes_repeats)


if __name__ == "__main__":
    main()
