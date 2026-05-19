"""
Figure 8 — Divergence Curves PICP(n) and MPIW(n)
Thesis §3.3 / §4 — shows how Bootstrap and Bayesian NUTS PICP and MPIW diverge
as n decreases from 44 to 5, with the divergence threshold n* ≈ 10 marked.

Data source: results/subsampling_combined.json

Run:  python figures/fig8_divergence_curves.py
Outputs: figures/fig8_divergence_curves.svg
         figures/fig8_divergence_curves.png
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from figures.style import (  # noqa: E402
    PALETTE as C, SIZES, LW, DPI_PNG, FONTSIZE,
    apply_style, label_panel, save_figure, LEGEND_KW,
)

# ── Parameterization ─────────────────────────────────────────────────────────
DATA_PATH          = ROOT / "results" / "subsampling_combined.json"

# Colors — from style bible
COLOR_BOOT         = C["bootstrap"]   # electric blue — Bootstrap
COLOR_BAYES        = C["bayesian"]    # vivid coral   — Bayesian NUTS
SHADE_ALPHA        = 0.15             # ±1 std band alpha

# Marker styles
MARKER_BOOT        = "o"
MARKER_BAYES       = "s"

# Nominal coverage reference line
NOMINAL_PICP       = 0.90

# Divergence threshold
THRESHOLD_N        = 10

# Axis ranges
PICP_YLIM          = (0.68, 1.06)
MPIW_YLIM          = (0, 85)     # MPa; clip ±1 std bands to this

# Figure
FIG_W, FIG_H       = SIZES["wide"]
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path: Path) -> tuple[list, dict]:
    """Return (sorted n values, records-by-n dict) from subsampling_combined.json."""
    records = json.loads(path.read_text())
    by_n = {r["n"]: r for r in records}
    ns = sorted(by_n)
    return ns, by_n


def main():
    apply_style()
    ns, by_n = load_data(DATA_PATH)

    # Extract series — Bootstrap: all n; Bayesian: key n only
    boot_picp  = [by_n[n]["picp_90_bootstrap"]     for n in ns]
    boot_pstd  = [by_n[n]["picp_90_bootstrap_std"] for n in ns]
    boot_mpiw  = [by_n[n]["mpiw_90_bootstrap"]     for n in ns]
    boot_mstd  = [by_n[n]["mpiw_90_bootstrap_std"] for n in ns]

    ns_bayes = [n for n in ns if "picp_90_bayesian" in by_n[n]]
    bayes_picp = [by_n[n]["picp_90_bayesian"]      for n in ns_bayes]
    bayes_mpiw = [by_n[n]["mpiw_90_bayesian"]      for n in ns_bayes]

    ns_arr       = np.array(ns)
    ns_bayes_arr = np.array(ns_bayes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    # ── Panel (a): PICP ──────────────────────────────────────────────────────
    ax1.axhline(NOMINAL_PICP, color=C["dark"], lw=LW["ref"], ls="--",
                alpha=0.6, label=f"Nominal {int(NOMINAL_PICP*100)}%")
    ax1.axvline(THRESHOLD_N, color=C["mid"], lw=LW["threshold"], ls=":",
                alpha=0.85, zorder=1)
    ax1.text(THRESHOLD_N + 0.5, PICP_YLIM[0] + 0.03,
             f"$n^* \\approx {THRESHOLD_N}$", fontsize=FONTSIZE["annotation"],
             color=C["mid"], va="bottom")

    # Bootstrap ±1 std band
    lo = np.clip([p - s for p, s in zip(boot_picp, boot_pstd)], PICP_YLIM[0], None)
    hi = np.clip([p + s for p, s in zip(boot_picp, boot_pstd)], None, PICP_YLIM[1])
    ax1.fill_between(ns_arr, lo, hi, alpha=SHADE_ALPHA, color=COLOR_BOOT)

    ax1.plot(ns_arr, boot_picp, f"{MARKER_BOOT}-",
             color=COLOR_BOOT, lw=LW["data"], ms=FONTSIZE["annotation"],
             label="Bootstrap")
    ax1.plot(ns_bayes_arr, bayes_picp, f"{MARKER_BAYES}--",
             color=COLOR_BAYES, lw=LW["data"], ms=FONTSIZE["annotation"],
             label="Bayesian NUTS")

    ax1.set_xlabel("Sample size $n$")
    ax1.set_ylabel("PICP at 90% nominal")
    ax1.set_xlim(ns_arr[0] - 1, ns_arr[-1] + 1)
    ax1.set_ylim(*PICP_YLIM)
    ax1.legend(loc="lower right", **LEGEND_KW)
    label_panel(ax1, "a")

    # ── Panel (b): MPIW ──────────────────────────────────────────────────────
    ax2.axvline(THRESHOLD_N, color=C["mid"], lw=LW["threshold"], ls=":",
                alpha=0.85, zorder=1)
    ax2.text(THRESHOLD_N + 0.5, MPIW_YLIM[1] * 0.92,
             f"$n^* \\approx {THRESHOLD_N}$", fontsize=FONTSIZE["annotation"],
             color=C["mid"], va="top")

    lo_m = np.clip([m - s for m, s in zip(boot_mpiw, boot_mstd)], 0, None)
    hi_m = np.clip([m + s for m, s in zip(boot_mpiw, boot_mstd)],
                   None, MPIW_YLIM[1])
    ax2.fill_between(ns_arr, lo_m, hi_m, alpha=SHADE_ALPHA, color=COLOR_BOOT)

    ax2.plot(ns_arr, boot_mpiw, f"{MARKER_BOOT}-",
             color=COLOR_BOOT, lw=LW["data"], ms=FONTSIZE["annotation"],
             label="Bootstrap")
    ax2.plot(ns_bayes_arr, bayes_mpiw, f"{MARKER_BAYES}--",
             color=COLOR_BAYES, lw=LW["data"], ms=FONTSIZE["annotation"],
             label="Bayesian NUTS")

    ax2.set_xlabel("Sample size $n$")
    ax2.set_ylabel("MPIW at 90% nominal (MPa)")
    ax2.set_xlim(ns_arr[0] - 1, ns_arr[-1] + 1)
    ax2.set_ylim(*MPIW_YLIM)
    ax2.legend(loc="upper right", **LEGEND_KW)
    label_panel(ax2, "b")

    save_figure(fig, Path(__file__).with_suffix(""))


if __name__ == "__main__":
    main()
