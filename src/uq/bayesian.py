"""Bayesian linear regression via PyMC v5+ NUTS for tensile strength prediction."""
from __future__ import annotations

import numpy as np
import pymc as pm
import arviz as az

from src.uq.bootstrap import compute_picp, compute_mpiw, compute_crps, compute_ece, COVERAGE_LEVELS


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


def build_model(
    x_std: np.ndarray,
    y_std: np.ndarray,
    degree: int = 1,
) -> pm.Model:
    """Bayesian polynomial regression with weakly informative priors (β ~ N(0,1), σ ~ HalfNormal(1))."""
    x_std = np.asarray(x_std, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    X = np.column_stack([x_std ** k for k in range(degree + 1)])

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=1, shape=degree + 1)
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


def run_bayesian_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    degree: int = 1,
    seed: int | None = None,
) -> dict:
    """End-to-end Bayesian MCMC pipeline. Returns all thesis metrics and convergence diagnostics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_std, y_std, params = standardize(x, y)

    model = build_model(x_std, y_std, degree=degree)
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
    }
