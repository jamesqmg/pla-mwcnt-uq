"""Subsampling experiment — divergence curves of PICP(n) and MPIW(n) across the n-ladder."""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from src.uq.bootstrap import run_bootstrap_pipeline
from src.uq.bayesian import run_bayesian_pipeline

__all__ = [
    "prepare_parent_dataset",
    "build_n_ladder",
    "subsample_dataset",
    "run_single_n",
    "run_subsampling_experiment",
    "divergence_threshold",
]

_log = logging.getLogger(__name__)
_NUMERIC_LEAD = re.compile(r"^-?\d+(?:\.\d+)?")


def _parse_num(val: object) -> float | None:
    if pd.isna(val):
        return None
    s = str(val).strip()
    m = _NUMERIC_LEAD.match(s)
    if m:
        return float(m.group())
    if s:
        _log.warning("_parse_num: could not parse %r — row will be dropped", s)
    return None


def prepare_parent_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract clean (x, y) arrays from a DataFrame. No material-system filter applied."""
    cnt_col = next(
        (c for c in df.columns if "cnt_wt_pct" in c.lower()
         or "weight fraction" in c.lower()),
        None,
    )
    ts_col = next(
        (c for c in df.columns if "tensile_mpa" in c.lower()
         or "tensile strength output" in c.lower()),
        None,
    )
    if cnt_col is None or ts_col is None:
        raise ValueError("Could not find CNT wt% or tensile MPa columns.")

    x_raw = df[cnt_col].map(_parse_num)
    y_raw = df[ts_col].map(_parse_num)
    mask = x_raw.notna() & y_raw.notna()
    x = x_raw[mask].values.astype(float)
    y = y_raw[mask].values.astype(float)

    physical = (y > 0) & (y < 1000)
    return x[physical], y[physical]


def build_n_ladder(
    max_n: int,
    values: list[int] | None = None,
) -> list[int]:
    """Build the n-ladder. Default steps: [5, 10, 15, 20, 25, 30, 35, 40, max_n]."""
    candidates = list(values) if values is not None else [5, 10, 15, 20, 25, 30, 35, 40, max_n]
    ladder = sorted(set(n for n in candidates if 5 <= n <= max_n))
    if len(ladder) < 3:
        raise ValueError(
            f"n-ladder must have ≥ 3 steps after filtering; got {ladder}. "
            f"Provide more values or increase max_n (currently {max_n})."
        )
    return ladder


def subsample_dataset(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a random subsample of size n without replacement."""
    N = len(x)
    if n > N:
        raise ValueError(f"n={n} exceeds parent size {N}.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n, replace=False)
    return x[idx], y[idx]


def run_single_n(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    B: int = 500,
    n_repeats: int = 20,
    seed: int | None = None,
    skip_bayesian: bool = False,
    draws: int = 500,
    tune: int = 500,
    chains: int = 4,
    degree: int = 1,
) -> dict:
    """
    Evaluate both UQ pipelines at a single n-value, averaged over n_repeats subsamples.

    Each repeat uses independent seeds for subsampling, bootstrap, and Bayesian.
    Bayesian failures are logged; result dict includes failure counts and effective repeats.
    """
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_repeats * 3)  # 3 independent seeds per repeat

    boot_metrics: dict[str, list] = {
        "picp_90": [], "picp_95": [], "mpiw_90": [], "mpiw_95": [], "crps": []
    }
    bay_metrics: dict[str, list] = {
        "picp_90": [], "picp_95": [], "mpiw_90": [], "mpiw_95": [], "crps": []
    }
    bay_failures = 0

    for r in range(n_repeats):
        sub_seed  = int(child_seeds[r * 3 + 0].generate_state(1)[0])
        boot_seed = int(child_seeds[r * 3 + 1].generate_state(1)[0])
        bay_seed  = int(child_seeds[r * 3 + 2].generate_state(1)[0])

        x_r, y_r = subsample_dataset(x, y, n=n, seed=sub_seed)

        b = run_bootstrap_pipeline(x_r, y_r, B=B, degree=degree,
                                   seed=boot_seed, add_noise=True)
        boot_metrics["picp_90"].append(b["picp_90"])
        boot_metrics["picp_95"].append(b["picp_95"])
        boot_metrics["mpiw_90"].append(b["mpiw_90"])
        boot_metrics["mpiw_95"].append(b["mpiw_95"])
        boot_metrics["crps"].append(b["crps"])

        if not skip_bayesian:
            try:
                bay = run_bayesian_pipeline(
                    x_r, y_r, draws=draws, tune=tune, chains=chains,
                    degree=degree, seed=bay_seed
                )
                bay_metrics["picp_90"].append(bay["picp_90"])
                bay_metrics["picp_95"].append(bay["picp_95"])
                bay_metrics["mpiw_90"].append(bay["mpiw_90"])
                bay_metrics["mpiw_95"].append(bay["mpiw_95"])
                bay_metrics["crps"].append(bay["crps"])
            except Exception as exc:
                bay_failures += 1
                _log.warning("Bayesian pipeline failed at n=%d repeat=%d: %s", n, r, exc)

    result: dict = {"n": n}

    for key, vals in boot_metrics.items():
        arr = np.array(vals, dtype=float)
        result[f"{key}_bootstrap"]     = float(arr.mean())
        result[f"{key}_bootstrap_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    if not skip_bayesian:
        result["bayesian_n_failures"]       = bay_failures
        result["bayesian_effective_repeats"] = n_repeats - bay_failures
        if bay_metrics["picp_90"]:
            for key, vals in bay_metrics.items():
                arr = np.array(vals, dtype=float)
                result[f"{key}_bayesian"]     = float(arr.mean())
                result[f"{key}_bayesian_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    return result


def run_subsampling_experiment(
    x: np.ndarray,
    y: np.ndarray,
    n_ladder: list[int],
    B: int = 500,
    n_repeats: int = 20,
    seed: int | None = None,
    skip_bayesian: bool = False,
    draws: int = 500,
    tune: int = 500,
    chains: int = 4,
    degree: int = 1,
) -> list[dict]:
    """Run the full subsampling experiment across the n-ladder. Returns list of dicts ordered by n."""
    ss = np.random.SeedSequence(seed)
    n_seeds = ss.spawn(len(n_ladder))
    results = []
    for i, n in enumerate(sorted(n_ladder)):
        n_seed = int(n_seeds[i].generate_state(1)[0])
        r = run_single_n(
            x, y, n=n, B=B, n_repeats=n_repeats, seed=n_seed,
            skip_bayesian=skip_bayesian, draws=draws, tune=tune,
            chains=chains, degree=degree,
        )
        results.append(r)
    return results


def divergence_threshold(
    results: list[dict],
    metric: str = "picp_90",
    delta: float = 0.10,
) -> dict:
    """
    Find the largest n where |PICP_bootstrap(n) - PICP_Bayesian(n)| > delta.

    Returns threshold_n (None if no divergence found), max_divergence, and per-n diffs.
    """
    bootstrap_key = f"{metric}_bootstrap"
    bayesian_key  = f"{metric}_bayesian"

    divergence_per_n: dict[int, float] = {}
    for r in results:
        if bootstrap_key in r and bayesian_key in r:
            diff = abs(r[bootstrap_key] - r[bayesian_key])
            divergence_per_n[r["n"]] = diff

    if not divergence_per_n:
        return {"threshold_n": None, "max_divergence": 0.0, "divergence_per_n": {}}

    max_divergence = max(divergence_per_n.values())

    # Scan descending: threshold = largest n where divergence still exceeds delta
    threshold_n = None
    for n in sorted(divergence_per_n, reverse=True):
        if divergence_per_n[n] > delta:
            threshold_n = n
            break

    return {
        "threshold_n":      threshold_n,
        "max_divergence":   float(max_divergence),
        "divergence_per_n": divergence_per_n,
    }
