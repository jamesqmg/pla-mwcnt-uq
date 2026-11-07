"""Hierarchical Bayesian regression with partial pooling across processing routes.

Per v6_design.md §4.1 (LOCKED 2026-05-12):

Dataset:   master_dataset.csv (n=45). NOT the n=26 PLA+FDM+D638 strict filter.
Grouping:  Option B (binary) — FDM (n=35) vs non-FDM (n=10), explicit string mapping.
Priors:    β_pop ~ N(0, 1)
           σ_pop ~ HalfN(1)        (population scale — informational, not used in likelihood)
           τ_β   ~ HalfN(0.5)      (group-level variation around β_pop)
           β_g   ~ N(β_pop, τ_β)   (partial-pooled group slopes)
           σ_obs ~ HalfN(1)
Sampling:  4 chains × 1000 draws, 1000 tune.
Gates:     R-hat < 1.01, ESS > 400. Abort if any chain > 30 min wall-clock.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Final

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.uq.bayesian import standardize
from src.uq.bootstrap import compute_crps, compute_mpiw, compute_picp
from src.uq.data_loader import parse_numeric_column

# ──────────────────────────────────────────────────────────────────────────────
# Locked Option B group mapping — verbatim strings from v6_design.md §4.1
# ──────────────────────────────────────────────────────────────────────────────

GROUP_FDM: Final[int]     = 0
GROUP_NON_FDM: Final[int] = 1

PROCESS_TO_GROUP: Final[dict[str, int]] = {
    # FDM (n=35 in master_dataset.csv)
    "FFF":                                                                GROUP_FDM,
    "3D Printing":                                                        GROUP_FDM,
    "Melt mixing + single-screw filament extrusion (for FDM/FFF use)":    GROUP_FDM,
    # non-FDM (n=10 in master_dataset.csv)
    "twin-screw extrusion + injection molding":                           GROUP_NON_FDM,
    "Filament (Melt Mixing + Single-Screw Extrusion — NOT FDM)":          GROUP_NON_FDM,
}


def assign_groups(processing_method: pd.Series) -> np.ndarray:
    """Map Processing Method strings to integer group indices via the locked Option B mapping.

    Raises KeyError if any value is outside the spec mapping — fail loud rather than silently
    mis-grouping a row.
    """
    s = processing_method.astype(str).str.strip()
    unknown = set(s) - set(PROCESS_TO_GROUP)
    if unknown:
        raise KeyError(f"unmapped Processing Method values: {sorted(unknown)}")
    return s.map(PROCESS_TO_GROUP).to_numpy(dtype=int)


def assign_papers(source_paper: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Map source_paper strings to contiguous integer codes for a random-intercept index.

    Unlike the locked Option B group mapping, paper levels are data-driven: codes follow
    first-appearance order. Returns ``(codes, names)`` where ``names[k]`` is the paper for
    code ``k``. Used by the per-paper random intercept (§3.5b expansion / Phase 2c).
    """
    s = source_paper.astype(str).str.strip()
    codes, uniques = pd.factorize(s, sort=False)
    return codes.astype(int), list(uniques)


def load_hierarchical_data(
    csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load master_dataset.csv (full n=45 pool) and emit (x, y, group_idx).

    Returns
    -------
    x : ndarray  — CNT weight fraction (%)
    y : ndarray  — tensile strength (MPa)
    group_idx : ndarray[int] — 0 for FDM, 1 for non-FDM (Option B locked)
    """
    df = pd.read_csv(csv_path)
    # Reuse the leading-float regex from data_loader so cells like "35.0 [GRAPHICAL]" parse.
    x = parse_numeric_column(df["CNT Weight Fraction (%)"]).to_numpy(dtype=float)
    y = parse_numeric_column(df["Tensile Strength Output (MPa)"]).to_numpy(dtype=float)
    group_idx = assign_groups(df["Processing Method"])

    mask = ~(np.isnan(x) | np.isnan(y))
    return x[mask], y[mask], group_idx[mask]


def load_hierarchical_data_with_papers(
    csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Same n=44 pool as load_hierarchical_data, plus a per-paper index for Phase 2c.

    Returns ``(x, y, group_idx, paper_idx, paper_names)``. Paper codes are assigned on the
    rows surviving the (wt%, strength) completeness mask, so a source contributing only
    incomplete rows does not occupy a level (e.g. sankar_2025 drops out, leaving 6 papers).
    """
    df = pd.read_csv(csv_path)
    x = parse_numeric_column(df["CNT Weight Fraction (%)"]).to_numpy(dtype=float)
    y = parse_numeric_column(df["Tensile Strength Output (MPa)"]).to_numpy(dtype=float)
    group_idx = assign_groups(df["Processing Method"])

    mask = ~(np.isnan(x) | np.isnan(y))
    paper_idx, paper_names = assign_papers(df.loc[mask, "source_paper"])
    return x[mask], y[mask], group_idx[mask], paper_idx, paper_names


# ──────────────────────────────────────────────────────────────────────────────
# Model construction
# ──────────────────────────────────────────────────────────────────────────────

def build_hierarchical_model(
    x: np.ndarray,
    y: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
    degree: int = 1,
    beta_pop_sigma: float = 1.0,
    tau_beta_sigma: float = 0.5,
) -> pm.Model:
    """Hierarchical regression with partial pooling on the slope coefficients.

    Each group g gets its own coefficient vector β_g of length (degree + 1).
    β_g is drawn from N(β_pop, τ_β), where β_pop is the population mean and
    τ_β controls how tightly groups cluster. Small τ_β → near-complete pooling;
    large τ_β → near-unpooled. ``beta_pop_sigma`` and ``tau_beta_sigma`` exist for the
    prior-sensitivity check; the defaults are the locked §4.1 priors.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    group_idx = np.asarray(group_idx, dtype=int)

    X = np.column_stack([x ** k for k in range(degree + 1)])  # (n, degree+1)
    n_coef = degree + 1

    with pm.Model() as model:
        beta_pop  = pm.Normal("beta_pop", mu=0.0, sigma=beta_pop_sigma, shape=n_coef)
        sigma_pop = pm.HalfNormal("sigma_pop", sigma=1.0)
        tau_beta  = pm.HalfNormal("tau_beta", sigma=tau_beta_sigma)

        # Non-centered parameterization to avoid the hierarchical funnel
        # (centered form was producing >100 divergences on n=44 real data).
        # Mathematically: β_g = β_pop + τ_β · z, with z ~ N(0, 1).
        z_g = pm.Normal("z_g", mu=0.0, sigma=1.0, shape=(n_groups, n_coef))
        beta_g = pm.Deterministic("beta_g", beta_pop + tau_beta * z_g)

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

        mu = pm.math.sum(beta_g[group_idx] * X, axis=1)
        pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y)

    # sigma_pop is part of the prior structure per §4.1 even though it does not appear
    # in the likelihood; retained so the spec table maps 1:1 to the variables in the model.
    _ = sigma_pop
    return model


def build_hierarchical_model_v2(
    x: np.ndarray,
    y: np.ndarray,
    group_idx: np.ndarray,
    paper_idx: np.ndarray,
    n_groups: int,
    n_papers: int,
    degree: int = 1,
    tau_alpha_sigma: float = 0.5,
) -> pm.Model:
    """Phase 2c: the Option B slope-pooling model plus a per-paper random intercept.

    Extends ``build_hierarchical_model`` additively with a zero-centred random intercept
    ``alpha_paper[paper] = tau_alpha * z_paper`` (non-centred), added to the linear predictor:
    ``mu = sum(beta_g[group] * X) + alpha_paper[paper]``. This absorbs between-study
    variance (different matrix strengths, process settings near-constant within a paper),
    answering the "blunt binary grouping" and study-heterogeneity critiques. ``tau_alpha``
    reports how much strength baseline varies across papers (standardised scale).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    group_idx = np.asarray(group_idx, dtype=int)
    paper_idx = np.asarray(paper_idx, dtype=int)

    X = np.column_stack([x ** k for k in range(degree + 1)])
    n_coef = degree + 1

    with pm.Model() as model:
        beta_pop = pm.Normal("beta_pop", mu=0.0, sigma=1.0, shape=n_coef)
        sigma_pop = pm.HalfNormal("sigma_pop", sigma=1.0)
        tau_beta = pm.HalfNormal("tau_beta", sigma=0.5)
        z_g = pm.Normal("z_g", mu=0.0, sigma=1.0, shape=(n_groups, n_coef))
        beta_g = pm.Deterministic("beta_g", beta_pop + tau_beta * z_g)

        # Per-paper random intercept (non-centred, zero-mean deviations).
        tau_alpha = pm.HalfNormal("tau_alpha", sigma=tau_alpha_sigma)
        z_paper = pm.Normal("z_paper", mu=0.0, sigma=1.0, shape=n_papers)
        alpha_paper = pm.Deterministic("alpha_paper", tau_alpha * z_paper)

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

        mu = pm.math.sum(beta_g[group_idx] * X, axis=1) + alpha_paper[paper_idx]
        pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y)

    _ = sigma_pop
    return model


def posterior_predict_hierarchical_v2(
    idata: az.InferenceData,
    x_pred_std: np.ndarray,
    group_idx_pred: np.ndarray,
    paper_idx_pred: np.ndarray,
    params: dict[str, float],
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior predictive intervals (MPa) including the per-paper intercept."""
    x_pred_std = np.asarray(x_pred_std, dtype=float)
    group_idx_pred = np.asarray(group_idx_pred, dtype=int)
    paper_idx_pred = np.asarray(paper_idx_pred, dtype=int)

    post = idata.posterior
    beta_g_arr = post["beta_g"].values
    alpha_arr = post["alpha_paper"].values
    sigma_arr = post["sigma_obs"].values

    beta_g_flat = beta_g_arr.reshape(-1, beta_g_arr.shape[-2], beta_g_arr.shape[-1])
    alpha_flat = alpha_arr.reshape(-1, alpha_arr.shape[-1])
    sigma_flat = sigma_arr.flatten()

    n_samples = beta_g_flat.shape[0]
    n_pred = len(x_pred_std)
    degree = beta_g_flat.shape[-1] - 1
    X_pred = np.column_stack([x_pred_std ** k for k in range(degree + 1)])

    rng = np.random.default_rng(0)
    pp = np.empty((n_samples, n_pred))
    for s in range(n_samples):
        coefs = beta_g_flat[s, group_idx_pred, :]
        mu = (X_pred * coefs).sum(axis=1) + alpha_flat[s, paper_idx_pred]
        pp[s] = rng.normal(mu, sigma_flat[s])

    pp_mpa = pp * params["y_std"] + params["y_mean"]
    lo_pct, up_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return pp_mpa.mean(axis=0), np.percentile(pp_mpa, lo_pct, axis=0), np.percentile(pp_mpa, up_pct, axis=0)


def run_hierarchical_v2_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    group_idx: np.ndarray,
    paper_idx: np.ndarray,
    n_groups: int = 2,
    n_papers: int | None = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    degree: int = 1,
    seed: int | None = None,
    target_accept: float = 0.99,
) -> dict:
    """End-to-end Phase 2c pipeline (slope pooling + per-paper intercept). Reports tau_alpha.

    Uses target_accept=0.99 by default: the added intercept level deepens the hierarchical
    funnel, so a smaller step size is needed to keep the sampler divergence-free.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    group_idx = np.asarray(group_idx, dtype=int)
    paper_idx = np.asarray(paper_idx, dtype=int)

    unique_codes = np.unique(paper_idx)
    if not np.array_equal(unique_codes, np.arange(len(unique_codes))):
        raise ValueError(
            f"paper_idx must be contiguous from 0; got unique values {unique_codes.tolist()}"
        )
    if n_papers is None:
        n_papers = int(unique_codes.shape[0])

    x_std, y_std, params = standardize(x, y)
    model = build_hierarchical_model_v2(
        x_std, y_std, group_idx, paper_idx, n_groups=n_groups, n_papers=n_papers, degree=degree
    )
    idata, runtimes = run_nuts_timed(
        model, draws=draws, tune=tune, chains=chains, seed=seed, target_accept=target_accept
    )
    convergence = check_convergence_all(idata)

    point_90, lo_90, up_90 = posterior_predict_hierarchical_v2(
        idata, x_std, group_idx, paper_idx, params, alpha=0.10
    )
    _, lo_95, up_95 = posterior_predict_hierarchical_v2(
        idata, x_std, group_idx, paper_idx, params, alpha=0.05
    )

    post = idata.posterior
    beta_g_flat = post["beta_g"].values.reshape(-1, n_groups, degree + 1)
    alpha_flat = post["alpha_paper"].values.reshape(-1, n_papers)
    sigma_flat = post["sigma_obs"].values.flatten()
    X_train = np.column_stack([x_std ** k for k in range(degree + 1)])
    n, B_eff = len(x), sigma_flat.shape[0]

    rng = np.random.default_rng(seed if seed is not None else 0)
    pp_std = np.empty((n, B_eff))
    for s in range(B_eff):
        coefs = beta_g_flat[s, group_idx, :]
        mu = (X_train * coefs).sum(axis=1) + alpha_flat[s, paper_idx]
        pp_std[:, s] = rng.normal(mu, sigma_flat[s])
    pp_mpa = pp_std * params["y_std"] + params["y_mean"]

    return {
        "picp_90": compute_picp(y, lo_90, up_90),
        "picp_95": compute_picp(y, lo_95, up_95),
        "mpiw_90": compute_mpiw(lo_90, up_90),
        "mpiw_95": compute_mpiw(lo_95, up_95),
        "crps": compute_crps(y, pp_mpa),
        "point_pred": point_90,
        "lower_90": lo_90,
        "upper_90": up_90,
        "lower_95": lo_95,
        "upper_95": up_95,
        "convergence": convergence,
        "chain_runtime_sec": runtimes,
        "n": n,
        "draws": draws,
        "chains": chains,
        "n_groups": n_groups,
        "n_papers": n_papers,
        "degree": degree,
        "beta_pop_posterior_mean": post["beta_pop"].values.mean(axis=(0, 1)),
        "beta_g_posterior_mean": post["beta_g"].values.mean(axis=(0, 1)),
        "tau_beta_posterior_mean": float(post["tau_beta"].values.mean()),
        "tau_alpha_posterior_mean": float(post["tau_alpha"].values.mean()),
        "alpha_paper_posterior_mean": alpha_flat.mean(axis=0),
        "idata": idata,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sampling with per-chain wall-clock timing (spec gate: 30 min/chain)
# ──────────────────────────────────────────────────────────────────────────────

def run_nuts_timed(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int | None = None,
    target_accept: float = 0.95,
) -> tuple[az.InferenceData, list[float]]:
    """Run NUTS with wall-clock recording per chain (single-process timing — total/chains).

    target_accept default 0.95: hierarchical models benefit from a higher target acceptance
    rate (smaller step size) to stay out of the funnel region; 0.8 (PyMC default) is too
    aggressive for this geometry even with non-centered parameterization.
    """
    with model:
        t0 = time.perf_counter()
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            random_seed=seed,
            progressbar=False,
            return_inferencedata=True,
            target_accept=target_accept,
        )
        elapsed = time.perf_counter() - t0
    runtime_per_chain = [elapsed / chains] * chains
    return idata, runtime_per_chain


def check_convergence_all(idata: az.InferenceData) -> dict:
    """R-hat < 1.01, ESS > 400 across every variable and every coordinate."""
    rhat_ds = az.rhat(idata)
    ess_ds  = az.ess(idata, method="bulk")

    rhat_per_var = {v: float(np.nanmax(rhat_ds[v].values)) for v in rhat_ds.data_vars}
    ess_per_var  = {v: float(np.nanmin(ess_ds[v].values))  for v in ess_ds.data_vars}

    rhat_max = max(rhat_per_var.values())
    ess_min  = min(ess_per_var.values())

    return {
        "rhat_max":     rhat_max,
        "ess_min":      ess_min,
        "passed":       rhat_max < 1.01 and ess_min > 400,
        "rhat_per_var": rhat_per_var,
        "ess_per_var":  ess_per_var,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Posterior predictive — per-row, using the row's group
# ──────────────────────────────────────────────────────────────────────────────

def posterior_predict_hierarchical(
    idata: az.InferenceData,
    x_pred_std: np.ndarray,
    group_idx_pred: np.ndarray,
    params: dict,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior predictive intervals on the original (MPa) scale.

    For each prediction row i, uses its group's slope β_g[group_idx_pred[i]] across all
    posterior samples, draws y ~ N(X β_g, σ_obs), and returns the percentile interval.
    """
    x_pred_std     = np.asarray(x_pred_std, dtype=float)
    group_idx_pred = np.asarray(group_idx_pred, dtype=int)

    post = idata.posterior
    # beta_g: (chain, draw, n_groups, n_coef) → flatten to (samples, n_groups, n_coef)
    beta_g_arr  = post["beta_g"].values
    sigma_arr   = post["sigma_obs"].values

    beta_g_flat = beta_g_arr.reshape(-1, beta_g_arr.shape[-2], beta_g_arr.shape[-1])
    sigma_flat  = sigma_arr.flatten()

    n_samples = beta_g_flat.shape[0]
    n_pred    = len(x_pred_std)
    n_coef    = beta_g_flat.shape[-1]
    degree    = n_coef - 1

    X_pred = np.column_stack([x_pred_std ** k for k in range(degree + 1)])

    rng = np.random.default_rng(0)
    pp = np.empty((n_samples, n_pred))
    for s in range(n_samples):
        # gather β_g[group] for each prediction row → shape (n_pred, n_coef)
        coefs = beta_g_flat[s, group_idx_pred, :]
        mu    = (X_pred * coefs).sum(axis=1)
        pp[s] = rng.normal(mu, sigma_flat[s])

    pp_mpa = pp * params["y_std"] + params["y_mean"]

    lo_pct = 100 * (alpha / 2)
    up_pct = 100 * (1 - alpha / 2)
    point  = pp_mpa.mean(axis=0)
    lower  = np.percentile(pp_mpa, lo_pct, axis=0)
    upper  = np.percentile(pp_mpa, up_pct, axis=0)

    return point, lower, upper


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_hierarchical_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int = 2,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    degree: int = 1,
    seed: int | None = None,
) -> dict:
    """End-to-end hierarchical Bayesian pipeline; mirrors run_bayesian_pipeline shape."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    group_idx = np.asarray(group_idx, dtype=int)

    x_std, y_std, params = standardize(x, y)

    model = build_hierarchical_model(x_std, y_std, group_idx, n_groups=n_groups, degree=degree)
    idata, runtimes = run_nuts_timed(model, draws=draws, tune=tune, chains=chains, seed=seed)

    convergence = check_convergence_all(idata)

    point_90, lo_90, up_90 = posterior_predict_hierarchical(
        idata, x_std, group_idx, params, alpha=0.10,
    )
    _, lo_95, up_95 = posterior_predict_hierarchical(
        idata, x_std, group_idx, params, alpha=0.05,
    )

    # PP matrix for CRPS
    post = idata.posterior
    beta_g_flat = post["beta_g"].values.reshape(-1, n_groups, degree + 1)
    sigma_flat  = post["sigma_obs"].values.flatten()
    X_train     = np.column_stack([x_std ** k for k in range(degree + 1)])
    n           = len(x)
    B_eff       = sigma_flat.shape[0]

    rng = np.random.default_rng(seed if seed is not None else 0)
    pp_std = np.empty((n, B_eff))
    for s in range(B_eff):
        coefs = beta_g_flat[s, group_idx, :]
        mu    = (X_train * coefs).sum(axis=1)
        pp_std[:, s] = rng.normal(mu, sigma_flat[s])
    pp_mpa = pp_std * params["y_std"] + params["y_mean"]

    beta_g_mean   = post["beta_g"].values.mean(axis=(0, 1))   # (n_groups, n_coef)
    beta_pop_mean = post["beta_pop"].values.mean(axis=(0, 1)) # (n_coef,)
    tau_beta_mean = float(post["tau_beta"].values.mean())

    return {
        "picp_90":                 compute_picp(y, lo_90, up_90),
        "picp_95":                 compute_picp(y, lo_95, up_95),
        "mpiw_90":                 compute_mpiw(lo_90, up_90),
        "mpiw_95":                 compute_mpiw(lo_95, up_95),
        "crps":                    compute_crps(y, pp_mpa),
        "point_pred":              point_90,
        "lower_90":                lo_90,
        "upper_90":                up_90,
        "lower_95":                lo_95,
        "upper_95":                up_95,
        "convergence":             convergence,
        "chain_runtime_sec":       runtimes,
        "n":                       n,
        "draws":                   draws,
        "chains":                  chains,
        "degree":                  degree,
        "n_groups":                n_groups,
        "beta_g_posterior_mean":   beta_g_mean,
        "beta_pop_posterior_mean": beta_pop_mean,
        "tau_beta_posterior_mean": tau_beta_mean,
        "idata":                   idata,
    }
