"""
Figure 9 — Divergence Curves PICP(n) and MPIW(n)
Thesis §3.6 — shows how Bootstrap and Bayesian NUTS PICP and MPIW track as n
decreases from 44 to 5. The methods stay close across the ladder (largest
between-method PICP gap is annotated); there is no sharp divergence threshold.

Data source: results/subsampling_combined.json

Run:  python figures/fig9_divergence_curves.py
Outputs: figures/fig9_divergence_curves.svg
         figures/fig9_divergence_curves.png
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

    # Annotate the largest between-method PICP gap (matches §3.6 text: ~0.04)
    _boot_at = dict(zip(ns_arr.tolist(), boot_picp))
    _gaps = [(abs(_boot_at[n] - y), n, _boot_at[n], y)
             for n, y in zip(ns_bayes_arr.tolist(), bayes_picp) if n in _boot_at]
    if _gaps:
        _g, _gn, _gb, _gy = max(_gaps)
        ax1.annotate("", xy=(_gn, _gb), xytext=(_gn, _gy),
                     arrowprops=dict(arrowstyle="<->", color=C["mid"], lw=1.3))
        ax1.text(_gn + 1.8, (_gb + _gy) / 2,
                 f"max gap $\\approx$ {_g:.2f}",
                 fontsize=FONTSIZE["annotation"], color=C["mid"], va="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor="none", alpha=0.8))

    # Data-scarcity ceiling: the calibrated (bootstrap) curve plateaus just below 0.90.
    ax1.annotate("plateau $\\approx$ 0.885 < 0.90\n(data-scarcity ceiling)",
                 xy=(30, 0.888), xytext=(19, 0.745),
                 fontsize=FONTSIZE["annotation"], color=C["mid"], ha="center",
                 arrowprops=dict(arrowstyle="->", color=C["mid"], lw=1.0))

    ax1.set_xlabel("Sample size $n$")
    ax1.set_ylabel("PICP at 90% nominal")
    ax1.set_xlim(ns_arr[0] - 1, ns_arr[-1] + 1)
    ax1.set_ylim(*PICP_YLIM)
    ax1.legend(loc="lower right", **LEGEND_KW)
    label_panel(ax1, "a")

    # ── Panel (b): MPIW ──────────────────────────────────────────────────────
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

    # Width budget: 50 MPa target (Table III). Bootstrap stays below it throughout;
    # Bayesian meets it only from n ~ 31.
    ax2.axhline(50, color=C["dark"], lw=LW["ref"], ls=":", alpha=0.75, zorder=1)
    ax2.text(ns_arr[-1], 52, "50 MPa budget", ha="right", va="bottom",
             fontsize=FONTSIZE["annotation"], color=C["dark"],
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor="none", alpha=0.85))
    ax2.annotate("Bayesian meets\nfrom $n \\approx$ 31", xy=(31, 50.3), xytext=(21, 66),
                 fontsize=FONTSIZE["annotation"], color=COLOR_BAYES, ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor="none", alpha=0.8),
                 arrowprops=dict(arrowstyle="->", color=COLOR_BAYES, lw=1.0))

    ax2.set_xlabel("Sample size $n$")
    ax2.set_ylabel("MPIW at 90% nominal (MPa)")
    ax2.set_xlim(ns_arr[0] - 1, ns_arr[-1] + 1)
    ax2.set_ylim(*MPIW_YLIM)
    ax2.legend(loc="upper right", **LEGEND_KW)
    label_panel(ax2, "b")

    save_figure(fig, Path(__file__).with_suffix(""))


if __name__ == "__main__":
    main()
