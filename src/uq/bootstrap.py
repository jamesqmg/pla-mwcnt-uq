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
    "run_bootstrap_pipeline",
]


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
        eval_lo_90, eval_up_90 = loo_lo_90, loo_up_90
        eval_lo_95, eval_up_95 = loo_lo_95, loo_up_95
    else:
        eval_lo_90, eval_up_90 = lo_90, up_90
        eval_lo_95, eval_up_95 = lo_95, up_95

    return {
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
