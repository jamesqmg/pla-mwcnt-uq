"""Bootstrap resampling uncertainty quantification for tensile strength prediction."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

__all__ = [
    "fit_regression",
    "bootstrap_predict",
    "compute_picp",
    "compute_mpiw",
    "compute_crps",
    "compute_nlpd",
    "compute_ece",
    "run_bootstrap_pipeline",
    "COVERAGE_LEVELS",
]

# Coverage levels for ECE computation — [0.50 … 0.95]
COVERAGE_LEVELS: list[float] = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def fit_regression(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
) -> Pipeline:
    if degree < 1:
        raise ValueError(f"degree must be ≥ 1, got {degree}")
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg",  LinearRegression()),
    ])
    pipe.fit(x.reshape(-1, 1), y)
    return pipe


def _bootstrap_sample_matrix(
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray,
    B: int,
    degree: int,
    rng: np.random.Generator,
    add_noise: bool,
) -> np.ndarray:
    """Return (B, len(x_pred)) matrix of bootstrap predictive draws."""
    n = len(x)
    boot_preds = np.empty((B, len(x_pred)))
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        x_b, y_b = x[idx], y[idx]
        model_b = fit_regression(x_b, y_b, degree=degree)
        f_b = model_b.predict(x_pred.reshape(-1, 1))
        if add_noise:
            resids_b = y_b - model_b.predict(x_b.reshape(-1, 1))
            sigma_b = resids_b.std()
            f_b = f_b + rng.normal(0.0, sigma_b, size=len(x_pred))
        boot_preds[b] = f_b
    return boot_preds


def bootstrap_predict(
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray,
    B: int = 1000,
    alpha: float = 0.10,
    degree: int = 1,
    seed: int | None = None,
    add_noise: bool = True,  # True = prediction interval; False = CI for mean
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap prediction intervals (Efron & Tibshirani, 1993)."""
    rng = np.random.default_rng(seed)
    boot_preds = _bootstrap_sample_matrix(x, y, x_pred, B, degree, rng, add_noise)
    point = boot_preds.mean(axis=0)
    lower = np.quantile(boot_preds, alpha / 2,       axis=0)
    upper = np.quantile(boot_preds, 1.0 - alpha / 2, axis=0)
    return point, lower, upper


def compute_picp(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Prediction Interval Coverage Probability."""
    covered = (y_true >= lower) & (y_true <= upper)
    return float(covered.mean())


def compute_mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width in MPa."""
    return float((upper - lower).mean())


def compute_crps(y_true: np.ndarray, boot_samples: np.ndarray) -> float:
    """CRPS via energy-score form (Gneiting & Raftery, 2007), averaged over n."""
    n, B = boot_samples.shape
    if len(y_true) != n:
        raise ValueError(
            f"y_true length {len(y_true)} must match boot_samples rows {n}"
        )
    f_sorted = np.sort(boot_samples, axis=1)
    e_fy = np.abs(f_sorted - y_true[:, None]).mean(axis=1)

    # O(B log B) energy term
    k = np.arange(1, B + 1, dtype=float)
    coeff = 2.0 * k - B - 1.0
    e_ff = (f_sorted * coeff).sum(axis=1) / (B * B)
    e_ff = np.maximum(e_ff, 0.0)

    return float((e_fy - e_ff).mean())


def compute_nlpd(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
) -> float:
    """NLPD = -mean log N(y_true; pred_mean, pred_std). Lower is better.

    Approximates the bootstrap predictive distribution as a Gaussian using
    per-fold bootstrap sample mean and standard deviation. Used under LOO.
    """
    from scipy import stats as _stats
    pred_std = np.asarray(pred_std, dtype=float)
    if np.any(pred_std <= 0):
        raise ValueError("pred_std must be strictly positive")
    return float(-np.mean(_stats.norm.logpdf(y_true, loc=pred_mean, scale=pred_std)))


def compute_ece(
    y_true: np.ndarray,
    lowers_by_coverage: dict[float, np.ndarray],
    uppers_by_coverage: dict[float, np.ndarray],
    coverage_levels: list[float] | None = None,
) -> float:
    """ECE = mean |empirical_picp(level) - level| across coverage levels. Lower is better.

    Perfect calibration gives ECE = 0. Coverage levels are nominal fractions in (0, 1),
    e.g. 0.90 = 90% prediction interval.
    """
    if coverage_levels is None:
        coverage_levels = COVERAGE_LEVELS
    deviations = [
        abs(compute_picp(y_true, lowers_by_coverage[lv], uppers_by_coverage[lv]) - lv)
        for lv in coverage_levels
    ]
    return float(np.mean(deviations))


def run_bootstrap_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    B: int = 1000,
    alpha: float = 0.10,
    degree: int = 1,
    seed: int | None = None,
    eval_mode: str = "bootstrap",
    add_noise: bool = True,
) -> dict:
    """
    End-to-end bootstrap pipeline.

    eval_mode: "bootstrap" = in-sample; "loo" = leave-one-out cross-validation.
    """
    if eval_mode not in {"bootstrap", "loo"}:
        raise ValueError(f"eval_mode must be 'bootstrap' or 'loo', got {eval_mode!r}")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    ss = np.random.SeedSequence(seed)
    rng_main = np.random.default_rng(ss.spawn(1)[0])
    boot_matrix = _bootstrap_sample_matrix(x, y, x, B, degree, rng_main, add_noise)

    lo_90 = np.quantile(boot_matrix, 0.05,  axis=0)
    up_90 = np.quantile(boot_matrix, 0.95,  axis=0)
    lo_95 = np.quantile(boot_matrix, 0.025, axis=0)
    up_95 = np.quantile(boot_matrix, 0.975, axis=0)
    point_90 = boot_matrix.mean(axis=0)

    if eval_mode == "loo":
        child_seeds = ss.spawn(2 * n)
        loo_lo_90 = np.empty(n)
        loo_up_90 = np.empty(n)
        loo_lo_95 = np.empty(n)
        loo_up_95 = np.empty(n)
        # Per-fold storage for NLPD and ECE
        loo_y_true    = np.empty(n)
        loo_pred_mean = np.empty(n)
        loo_pred_std  = np.empty(n)
        loo_lowers: dict[float, np.ndarray] = {lv: np.empty(n) for lv in COVERAGE_LEVELS}
        loo_uppers: dict[float, np.ndarray] = {lv: np.empty(n) for lv in COVERAGE_LEVELS}

        for i in range(n):
            idx = np.delete(np.arange(n), i)
            rng_i90 = np.random.default_rng(child_seeds[i])
            rng_i95 = np.random.default_rng(child_seeds[n + i])
            b_i = max(B // 2, 100)
            mat90 = _bootstrap_sample_matrix(
                x[idx], y[idx], x[[i]], b_i, degree, rng_i90, add_noise
            )
            mat95 = _bootstrap_sample_matrix(
                x[idx], y[idx], x[[i]], b_i, degree, rng_i95, add_noise
            )
            loo_lo_90[i] = np.quantile(mat90, 0.05)
            loo_up_90[i] = np.quantile(mat90, 0.95)
            loo_lo_95[i] = np.quantile(mat95, 0.025)
            loo_up_95[i] = np.quantile(mat95, 0.975)
            # Per-fold moments from mat90 (sufficient for Gaussian NLPD approximation)
            samples_i = mat90.flatten()
            loo_y_true[i]    = y[i]
            loo_pred_mean[i] = samples_i.mean()
            loo_pred_std[i]  = max(samples_i.std(ddof=1), 1e-6)  # guard against zero std
            for lv in COVERAGE_LEVELS:
                tail = (1.0 - lv) / 2.0
                loo_lowers[lv][i] = np.quantile(samples_i, tail)
                loo_uppers[lv][i] = np.quantile(samples_i, 1.0 - tail)

        eval_lo_90, eval_up_90 = loo_lo_90, loo_up_90
        eval_lo_95, eval_up_95 = loo_lo_95, loo_up_95
    else:
        eval_lo_90, eval_up_90 = lo_90, up_90
        eval_lo_95, eval_up_95 = lo_95, up_95

    result: dict = {
        "picp_90":       compute_picp(y, eval_lo_90, eval_up_90),
        "picp_95":       compute_picp(y, eval_lo_95, eval_up_95),
        "mpiw_90":       compute_mpiw(eval_lo_90, eval_up_90),
        "mpiw_95":       compute_mpiw(eval_lo_95, eval_up_95),
        "crps":          compute_crps(y, boot_matrix.T),
        "point_pred":    point_90,
        "lower_90":      lo_90,
        "upper_90":      up_90,
        "lower_95":      lo_95,
        "upper_95":      up_95,
        "eval_lower_90": eval_lo_90,
        "eval_upper_90": eval_up_90,
        "eval_lower_95": eval_lo_95,
        "eval_upper_95": eval_up_95,
        "n":             n,
        "B":             B,
        "degree":        degree,
        "add_noise":     add_noise,
    }
    if eval_mode == "loo":
        result["nlpd"] = compute_nlpd(loo_y_true, loo_pred_mean, loo_pred_std)
        result["ece"]  = compute_ece(loo_y_true, loo_lowers, loo_uppers, COVERAGE_LEVELS)
        result["ece_by_coverage"] = {
            str(lv): abs(
                compute_picp(loo_y_true, loo_lowers[lv], loo_uppers[lv]) - lv
            )
            for lv in COVERAGE_LEVELS
        }
        # Per-fold predictions for parity + reliability figures
        result["loo_y_true"]    = loo_y_true.tolist()
        result["loo_pred_mean"] = loo_pred_mean.tolist()
        result["loo_pred_std"]  = loo_pred_std.tolist()
        result["loo_lower_90"]  = loo_lowers[0.9].tolist()
        result["loo_upper_90"]  = loo_uppers[0.9].tolist()
        result["picp_by_coverage"] = {
            str(lv): compute_picp(loo_y_true, loo_lowers[lv], loo_uppers[lv])
            for lv in COVERAGE_LEVELS
        }
    return result
