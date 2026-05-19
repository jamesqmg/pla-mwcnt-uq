"""
src/uq/inversion.py — Divergence-curve inversion (specimen-budget tool)

Reads results/subsampling_combined.json and inverts the PICP(n) and MPIW(n)
curves to produce a specimen-budget table:

    "To achieve PICP ≥ target, you need at least n_min observations."
    "To achieve MPIW ≤ target (MPa), you need at least n_min observations."

This is the §4.4 inversion / value-of-information result.

Usage:
    from src.uq.inversion import compute_inversion_table
    table = compute_inversion_table("results/subsampling_combined.json")
    # → results/inversion_table.json  (write separately if needed)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np


# ── Default targets ────────────────────────────────────────────────────────────
# PICP targets (≥): want coverage above these thresholds
PICP_TARGETS: list[float] = [0.85, 0.875, 0.90]

# MPIW targets (≤, MPa): want interval width below these thresholds
MPIW_TARGETS: list[float] = [50.0, 40.0, 30.0]


def _find_stable_crossing(
    ns: list[int],
    vals: list[float],
    target: float,
    direction: Literal["above", "below"],
) -> float | None:
    """Find the minimum n where the curve stably meets the target.

    "Stably" means: the interpolated curve crosses the target at some n*,
    and all observed points at n ≥ n* also satisfy the criterion.  If the
    curve never meets the target, returns None.  If it already meets the
    target at the smallest n, returns that smallest n.

    Uses linear interpolation between adjacent observed (n, value) pairs.

    Args:
        ns:        Sorted list of sample sizes (ascending).
        vals:      Corresponding metric values (same length as ns).
        target:    The threshold to cross.
        direction: "above" — want val ≥ target (e.g. PICP).
                   "below" — want val ≤ target (e.g. MPIW).

    Returns:
        Minimum n (float, interpolated) or None.
    """
    ns_arr = np.array(ns, dtype=float)
    vals_arr = np.array(vals, dtype=float)

    def _satisfied(v: float) -> bool:
        return v >= target if direction == "above" else v <= target

    # Find the last run of satisfied points from the right.
    # We want the smallest n such that all observed n' ≥ n satisfy the criterion.
    last_fail_idx = -1
    for i in range(len(ns_arr)):
        if not _satisfied(vals_arr[i]):
            last_fail_idx = i

    if last_fail_idx == -1:
        # All points already satisfy — return smallest observed n
        return float(ns_arr[0])

    if last_fail_idx == len(ns_arr) - 1:
        # Even the largest n fails — target never stably achieved
        return None

    # Crossing is between last_fail_idx and last_fail_idx + 1
    # Linear interpolation to find the exact crossing point
    n0, v0 = ns_arr[last_fail_idx], vals_arr[last_fail_idx]
    n1, v1 = ns_arr[last_fail_idx + 1], vals_arr[last_fail_idx + 1]

    if abs(v1 - v0) < 1e-12:
        # Flat segment — if satisfied, return n0; else n1
        return float(n1) if _satisfied(v1) else None

    # Interpolate: n* = n0 + (target - v0) / (v1 - v0) * (n1 - n0)
    n_star = n0 + (target - v0) / (v1 - v0) * (n1 - n0)
    # Round up to nearest integer (need whole specimens)
    return float(int(np.ceil(n_star)))


def _build_metric_table(
    ns: list[int],
    vals: list[float],
    targets: list[float],
    direction: Literal["above", "below"],
) -> list[dict]:
    """Build a list of {target, n_min} dicts for one metric / one method."""
    return [
        {"target": t, "n_min": _find_stable_crossing(ns, vals, t, direction)}
        for t in targets
    ]


def compute_inversion_table(
    combined_json_path: str | Path,
    *,
    picp_targets: list[float] = PICP_TARGETS,
    mpiw_targets: list[float] = MPIW_TARGETS,
    alpha: str = "90",
) -> dict:
    """Invert divergence curves to produce a specimen-budget table.

    Args:
        combined_json_path: Path to results/subsampling_combined.json.
        picp_targets:       List of PICP thresholds (≥) to invert.
        mpiw_targets:       List of MPIW thresholds in MPa (≤) to invert.
        alpha:              Coverage level key suffix, e.g. "90" → picp_90_*.

    Returns:
        dict with keys "bootstrap" and "bayesian", each mapping to:
            {
              "picp_<alpha>": [{"target": float, "n_min": float | None}, ...],
              "mpiw_<alpha>": [{"target": float, "n_min": float | None}, ...],
            }
        n_min is None when the target is never stably achieved in the data.
    """
    data = json.loads(Path(combined_json_path).read_text())
    by_n: dict[int, dict] = {r["n"]: r for r in data}
    ns_all = sorted(by_n)

    picp_key = f"picp_{alpha}_bootstrap"
    mpiw_key = f"mpiw_{alpha}_bootstrap"
    picp_key_b = f"picp_{alpha}_bayesian"
    mpiw_key_b = f"mpiw_{alpha}_bayesian"

    # Bootstrap — present at all n in ns_all
    ns_boot = [n for n in ns_all if picp_key in by_n[n]]
    picp_boot = [by_n[n][picp_key] for n in ns_boot]
    mpiw_boot = [by_n[n][mpiw_key] for n in ns_boot]

    # Bayesian — only at key n values
    ns_bayes = [n for n in ns_all if picp_key_b in by_n[n]]
    picp_bayes = [by_n[n][picp_key_b] for n in ns_bayes]
    mpiw_bayes = [by_n[n][mpiw_key_b] for n in ns_bayes]

    metric_label = f"picp_{alpha}"
    mpiw_label = f"mpiw_{alpha}"

    result: dict = {
        "bootstrap": {
            metric_label: _build_metric_table(ns_boot, picp_boot, picp_targets, "above"),
            mpiw_label:   _build_metric_table(ns_boot, mpiw_boot, mpiw_targets, "below"),
        },
        "bayesian": {
            metric_label: _build_metric_table(ns_bayes, picp_bayes, picp_targets, "above"),
            mpiw_label:   _build_metric_table(ns_bayes, mpiw_bayes, mpiw_targets, "below"),
        },
    }
    return result
