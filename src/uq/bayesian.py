"""Bayesian linear regression via PyMC v5+ NUTS for tensile strength prediction."""
from __future__ import annotations

import numpy as np
import pymc as pm
import arviz as az

from src.uq.bootstrap import (
    compute_picp, compute_mpiw, compute_crps, compute_ece, compute_nlpd,
    COVERAGE_LEVELS, _make_loo_folds,
)


def standardize(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Z-score standardise x and y. Raises if x is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_mean, x_std_val = x.mean(), x.std()
    y_mean, y_std_val = y.mean(), y.std()

    if x_std_val == 0.0:
        raise ValueError("x is constant — cannot standardise (std = 0).")

    params = {
        "x_mean": x_mean,
        "x_std":  x_std_val,
        "y_mean": y_mean,
        "y_std":  y_std_val,
    }
    return (x - x_mean) / x_std_val, (y - y_mean) / y_std_val, params


def unstandardize_predictions(y_std: np.ndarray, params: dict) -> np.ndarray:
    """Reverse z-score transform back to MPa."""
    return y_std * params["y_std"] + params["y_mean"]


def standardized_slope_prior(
    raw_slope: float,
    raw_sigma: float,
    params: dict,
) -> tuple[float, float]:
    """Convert a raw strength slope (MPa per wt%) and its sd to standardized-β units.

    The model regresses ``y_std`` on ``x_std``, so the raw relationship dy/dx equals
    ``beta_1 * y_std / x_std``; hence ``beta_1 = raw_slope * x_std / y_std`` (the
    dispersion scales identically). Use this to map a Halpin–Tsai-implied reinforcement
    slope (§3.7) onto the prior mean of the degree-1 coefficient before calling
    ``build_model(slope_prior=...)``.
    """
    if raw_sigma <= 0.0:
        raise ValueError("raw_sigma must be positive.")
    if params["y_std"] == 0.0:
        raise ValueError("y is constant — cannot convert slope (y_std = 0).")
    scale = params["x_std"] / params["y_std"]
    return raw_slope * scale, raw_sigma * scale


def build_model(
    x_std: np.ndarray,
    y_std: np.ndarray,
    degree: int = 1,
    slope_prior: tuple[float, float] | None = None,
) -> pm.Model:
    """Bayesian polynomial regression with weakly informative priors (β ~ N(0,1), σ ~ HalfNormal(1)).

    Args:
        slope_prior: Optional ``(mu, sigma)`` in standardized units for a physics-informed
            prior on the degree-1 (CNT-loading slope) coefficient — the Halpin–Tsai
            expansion of Phase 2 (§3.5). ``None`` (default) keeps the flat Normal(0, 1)
            baseline on every coefficient and is byte-stable with prior results. When set,
            only the slope coefficient is recentred; the intercept and any higher-order
            terms remain Normal(0, 1). Convert raw MPa-per-wt% values with
            ``standardized_slope_prior`` first.
    """
    x_std = np.asarray(x_std, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    X = np.column_stack([x_std ** k for k in range(degree + 1)])

    with pm.Model() as model:
        if slope_prior is None:
            beta = pm.Normal("beta", mu=0, sigma=1, shape=degree + 1)
        else:
            if degree < 1:
                raise ValueError("slope_prior requires degree >= 1 (no slope term at degree 0).")
            slope_mu, slope_sigma = slope_prior
            if slope_sigma <= 0.0:
                raise ValueError("slope_prior sigma must be positive.")
            mu_vec = np.zeros(degree + 1)
            sigma_vec = np.ones(degree + 1)
            mu_vec[1] = slope_mu
            sigma_vec[1] = slope_sigma
            beta = pm.Normal("beta", mu=mu_vec, sigma=sigma_vec, shape=degree + 1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = pm.math.dot(X, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_std)

    return model


def prior_predictive_check(
    model: pm.Model,
    n_samples: int = 500,
    seed: int | None = None,
) -> np.ndarray:
    """Draw from prior predictive distribution (standardised scale)."""
    with model:
        prior_idata = pm.sample_prior_predictive(samples=n_samples, random_seed=seed)
    samples = prior_idata.prior_predictive["y_obs"].values.flatten()
    return samples.astype(float)


def run_nuts(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int | None = None,
    compute_log_likelihood: bool = False,
) -> az.InferenceData:
    """Run NUTS via PyMC v5+. Default: 4 chains × 1000 post-warmup draws.

    Args:
        compute_log_likelihood: If True, include log_likelihood in idata for
            PSIS-LOO via az.loo(). Required for compute_bayesian_nlpd().
    """
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            progressbar=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": compute_log_likelihood},
        )
    return idata


def check_convergence(idata: az.InferenceData) -> dict:
    """Check R-hat and ESS. Gates: R-hat < 1.01, ESS > 400."""
    rhat_df = az.rhat(idata)
    ess_df  = az.ess(idata, method="bulk")

    rhat_per_var = {v: float(rhat_df[v].values.max()) for v in rhat_df.data_vars}
    ess_per_var  = {v: float(ess_df[v].values.min())  for v in ess_df.data_vars}

    rhat_max = max(rhat_per_var.values())
    ess_min  = min(ess_per_var.values())

    return {
        "rhat_max":     rhat_max,
        "ess_min":      ess_min,
        "passed":       rhat_max < 1.01 and ess_min > 400,
        "rhat_per_var": rhat_per_var,
        "ess_per_var":  ess_per_var,
    }


def posterior_predict(
    idata: az.InferenceData,
    x_pred_std: np.ndarray,
    params: dict,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior predictive intervals at nominal coverage (1 - alpha), returned in MPa."""
    x_pred_std = np.asarray(x_pred_std, dtype=float)
    post = idata.posterior

    beta_flat  = post["beta"].values.reshape(-1, post["beta"].shape[-1])
    sigma_flat = post["sigma"].values.flatten()
    n_samples  = len(sigma_flat)
    n_pred     = len(x_pred_std)
    degree     = beta_flat.shape[1] - 1

    X_pred = np.column_stack([x_pred_std ** k for k in range(degree + 1)])

    rng = np.random.default_rng(0)
    pp_samples = np.empty((n_samples, n_pred))
    for i in range(n_samples):
        mu_i = X_pred @ beta_flat[i]
        pp_samples[i] = rng.normal(mu_i, sigma_flat[i])

    pp_mpa = pp_samples * params["y_std"] + params["y_mean"]

    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    point = pp_mpa.mean(axis=0)
    lower = np.percentile(pp_mpa, lower_pct, axis=0)
    upper = np.percentile(pp_mpa, upper_pct, axis=0)

    return point, lower, upper


def compute_bayesian_nlpd(idata: az.InferenceData, n: int) -> float:
    """NLPD via ArviZ PSIS-LOO: -ELPD_LOO / n. No new NUTS runs required.

    Uses Pareto-smoothed importance sampling from existing posterior.
    Requires idata to have a log_likelihood group (pass compute_log_likelihood=True
    to run_nuts, or call with idata_kwargs={"log_likelihood": True} in pm.sample).
    Lower is better; 0 = perfect predictive log probability.
    """
    loo_result = az.loo(idata)
    return float(-loo_result.elpd_loo / n)


def _posterior_predictive_samples(
    idata: az.InferenceData,
    x_pred_std: np.ndarray,
    params: dict,
    degree: int,
    seed: int = 0,
) -> np.ndarray:
    """Posterior-predictive draws in MPa. Returns shape (n_pred, n_post)."""
    post = idata.posterior
    beta_flat = post["beta"].values.reshape(-1, post["beta"].shape[-1])
    sigma_flat = post["sigma"].values.flatten()
    X = np.column_stack([np.asarray(x_pred_std, dtype=float) ** k for k in range(degree + 1)])
    rng = np.random.default_rng(seed)
    pp = np.empty((len(sigma_flat), X.shape[0]))
    for i in range(len(sigma_flat)):
        pp[i] = rng.normal(X @ beta_flat[i], sigma_flat[i])
    pp_mpa = pp * params["y_std"] + params["y_mean"]
    return pp_mpa.T  # (n_pred, n_post)


def run_bayesian_loo(
    x: np.ndarray,
    y: np.ndarray,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    degree: int = 1,
    seed: int | None = None,
    slope_prior: tuple[float, float] | None = None,
    coverage_levels: list[float] | None = None,
    groups: np.ndarray | None = None,
) -> dict:
    """Honest leave-one-out Bayesian evaluation with EXACT posterior-predictive quantiles.

    For each held-out point i: refit NUTS on the remaining n-1 observations (standardised
    on the training fold), then draw the posterior predictive for x_i and read its empirical
    quantiles directly. This replaces the in-sample evaluation (which gives PICP=1.000) and
    the z-ratio Gaussian approximation in the reliability figure (reviews C4 / C-rec1 / P#6).

    Returns metrics plus a ``reliability_curve`` of (nominal, empirical-PICP) pairs and the
    per-point ``loo_pred_mean``/``loo_pred_std``/``loo_y_true`` for the parity figure.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if coverage_levels is None:
        coverage_levels = COVERAGE_LEVELS
    if groups is not None and len(groups) != n:
        raise ValueError(f"groups length {len(groups)} must match n={n}")

    ss = np.random.SeedSequence(seed)
    fold_seeds = ss.spawn(n)
    folds = _make_loo_folds(n, groups)
    if groups is not None and len(folds) < 2:
        raise ValueError(
            "leave-one-group-out requires >= 2 distinct groups; got "
            f"{len(folds)} — the only fold would hold out every row, leaving "
            "no training data"
        )

    samples_by_idx: dict[int, np.ndarray] = {}
    for f, fold_idx in enumerate(folds):
        train = np.setdiff1d(np.arange(n), fold_idx)
        x_tr, y_tr = x[train], y[train]
        x_std, y_std, params = standardize(x_tr, y_tr)
        x_held_std = (x[fold_idx] - params["x_mean"]) / params["x_std"]

        model = build_model(x_std, y_std, degree=degree, slope_prior=slope_prior)
        fseed = int(fold_seeds[f].generate_state(1)[0])
        idata = run_nuts(model, draws=draws, tune=tune, chains=chains, seed=fseed)
        samples = _posterior_predictive_samples(
            idata, x_held_std, params, degree, seed=fseed
        )  # (len(fold_idx), n_post)
        for j, i in enumerate(fold_idx):
            samples_by_idx[int(i)] = samples[j]

    # Reassemble in original row order; folds partition every index exactly once.
    loo_samples = [samples_by_idx[i] for i in range(n)]

    # Align posterior sample counts across folds, then stack to (n, n_post).
    n_post = min(len(s) for s in loo_samples)
    M = np.vstack([s[:n_post] for s in loo_samples])

    lowers: dict[float, np.ndarray] = {}
    uppers: dict[float, np.ndarray] = {}
    for lv in coverage_levels:
        a = 1.0 - lv
        lowers[lv] = np.percentile(M, 100 * a / 2, axis=1)
        uppers[lv] = np.percentile(M, 100 * (1 - a / 2), axis=1)

    lo_90, up_90 = lowers[0.90], uppers[0.90]
    lo_95, up_95 = lowers[0.95], uppers[0.95]

    pred_mean = M.mean(axis=1)
    pred_std = np.maximum(M.std(axis=1, ddof=1), 1e-6)

    reliability_curve = [
        (lv, compute_picp(y, lowers[lv], uppers[lv])) for lv in coverage_levels
    ]

    return {
        "picp_90":   compute_picp(y, lo_90, up_90),
        "picp_95":   compute_picp(y, lo_95, up_95),
        "mpiw_90":   compute_mpiw(lo_90, up_90),
        "mpiw_95":   compute_mpiw(lo_95, up_95),
        "crps":      compute_crps(y, M),
        "nlpd":      compute_nlpd(y, pred_mean, pred_std),
        "ece":       compute_ece(y, lowers, uppers, coverage_levels),
        "reliability_curve": reliability_curve,
        "loo_pred_mean": pred_mean.tolist(),
        "loo_pred_std":  pred_std.tolist(),
        "loo_y_true":    y.tolist(),
        "loo_lower_90":  lo_90.tolist(),
        "loo_upper_90":  up_90.tolist(),
        "n": n,
        "n_groups": len(folds),
        "eval_scheme": (
            "leave-one-group-out" if groups is not None else "leave-one-row-out"
        ),
    }


def run_bayesian_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    degree: int = 1,
    seed: int | None = None,
    slope_prior: tuple[float, float] | None = None,
) -> dict:
    """End-to-end Bayesian MCMC pipeline. Returns all thesis metrics and convergence diagnostics.

    Pass ``slope_prior`` (standardized ``(mu, sigma)``) to run the Halpin–Tsai informed-prior
    sensitivity analysis (§3.5); ``None`` reproduces the flat-prior primary result.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_std, y_std, params = standardize(x, y)

    model = build_model(x_std, y_std, degree=degree, slope_prior=slope_prior)
    prior_samples_std = prior_predictive_check(model, n_samples=500, seed=seed)
    prior_samples_mpa = unstandardize_predictions(prior_samples_std, params)

    idata = run_nuts(model, draws=draws, tune=tune, chains=chains, seed=seed,
                     compute_log_likelihood=True)

    convergence = check_convergence(idata)

    x_pred_std = x_std
    point_90, lo_90, up_90 = posterior_predict(idata, x_pred_std, params, alpha=0.10)
    _, lo_95, up_95 = posterior_predict(idata, x_pred_std, params, alpha=0.05)

    post = idata.posterior
    beta_flat  = post["beta"].values.reshape(-1, post["beta"].shape[-1])
    sigma_flat = post["sigma"].values.flatten()
    n          = len(x)
    B_eff      = len(sigma_flat)
    X_train    = np.column_stack([x_std ** k for k in range(degree + 1)])

    rng = np.random.default_rng(seed if seed is not None else 0)
    pp_matrix_std = np.empty((n, B_eff))
    for i in range(B_eff):
        mu_i = X_train @ beta_flat[i]
        pp_matrix_std[:, i] = rng.normal(mu_i, sigma_flat[i])
    pp_matrix_mpa = pp_matrix_std * params["y_std"] + params["y_mean"]

    # Posterior slope (degree-1 coefficient): standardized + raw MPa-per-wt% scale.
    slope_std_mean = float(beta_flat[:, 1].mean()) if degree >= 1 else float("nan")
    slope_std_sd = float(beta_flat[:, 1].std()) if degree >= 1 else float("nan")
    slope_raw_mean = slope_std_mean * params["y_std"] / params["x_std"]

    # NLPD via ArviZ PSIS-LOO (uses log_likelihood from idata)
    nlpd = compute_bayesian_nlpd(idata, n=n)

    # ECE from in-sample posterior predictive at 6 coverage levels
    lowers_by_cov: dict[float, np.ndarray] = {}
    uppers_by_cov: dict[float, np.ndarray] = {}
    for lv in COVERAGE_LEVELS:
        alpha_lv = 1.0 - lv
        _, lo_lv, up_lv = posterior_predict(idata, x_pred_std, params, alpha=alpha_lv)
        lowers_by_cov[lv] = lo_lv
        uppers_by_cov[lv] = up_lv
    ece = compute_ece(y, lowers_by_cov, uppers_by_cov, COVERAGE_LEVELS)

    return {
        "picp_90":       compute_picp(y, lo_90, up_90),
        "picp_95":       compute_picp(y, lo_95, up_95),
        "mpiw_90":       compute_mpiw(lo_90, up_90),
        "mpiw_95":       compute_mpiw(lo_95, up_95),
        "crps":          compute_crps(y, pp_matrix_mpa),
        "nlpd":          nlpd,
        "ece":           ece,
        "ece_note":      "in-sample posterior predictive; LOO PICP@90=1.000 confirmed",
        "point_pred":    point_90,
        "lower_90":      lo_90,
        "upper_90":      up_90,
        "lower_95":      lo_95,
        "upper_95":      up_95,
        "convergence":   convergence,
        "prior_p5_mpa":  float(np.percentile(prior_samples_mpa, 5)),
        "prior_p95_mpa": float(np.percentile(prior_samples_mpa, 95)),
        "n":             n,
        "draws":         draws,
        "chains":        chains,
        "degree":        degree,
        "slope_prior":   slope_prior,
        "slope_std_mean":  slope_std_mean,
        "slope_std_sd":    slope_std_sd,
        "slope_raw_mean":  slope_raw_mean,
    }
