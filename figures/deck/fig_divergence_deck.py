"""
DECK figure — Divergence Curves PICP(n) / MPIW(n), dark/neon Allotrope profile.
Money slide (Finding 3): Bootstrap vs Bayesian NUTS PICP & MPIW vs n, n* ~ 10.

Content == manuscript Fig 9 (fig9_divergence_curves.py); only the visual profile
differs (dark bg + neon). Data source: results/subsampling_combined.json.

Run:  ~/miniconda3/bin/python figures/deck/fig_divergence_deck.py
Out:  Presentations/current/figures/divergence_deck.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent   # Thesis/
sys.path.insert(0, str(ROOT))

from figures.deck_style import (  # noqa: E402
    PALETTE as C, NEUTRAL, SIZES, FONTSIZE, LW, LEGEND_KW, MARKER_SIZE,
    apply_deck, deck_axes, label_panel, glow_line, band, save_deck,
)

DATA_PATH = ROOT / "results" / "subsampling_combined.json"
OUT_BASE  = ROOT / "Presentations" / "current" / "figures" / "divergence_deck"

NOMINAL_PICP = 0.90
THRESHOLD_N  = 10
PICP_YLIM    = (0.68, 1.06)
MPIW_YLIM    = (0, 85)
MARKER_BOOT, MARKER_BAYES = "o", "s"


def load_data(path: Path):
    records = json.loads(path.read_text())
    by_n = {r["n"]: r for r in records}
    return sorted(by_n), by_n


def main():
    apply_deck()
    ns, by_n = load_data(DATA_PATH)

    boot_picp = [by_n[n]["picp_90_bootstrap"]     for n in ns]
    boot_pstd = [by_n[n]["picp_90_bootstrap_std"] for n in ns]
    boot_mpiw = [by_n[n]["mpiw_90_bootstrap"]     for n in ns]
    boot_mstd = [by_n[n]["mpiw_90_bootstrap_std"] for n in ns]

    ns_bayes   = [n for n in ns if "picp_90_bayesian" in by_n[n]]
    bayes_picp = [by_n[n]["picp_90_bayesian"] for n in ns_bayes]
    bayes_mpiw = [by_n[n]["mpiw_90_bayesian"] for n in ns_bayes]

    ns_arr, ns_bayes_arr = np.array(ns), np.array(ns_bayes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["wide2"])

    # ── Panel (a): PICP ──────────────────────────────────────────────────────
    deck_axes(ax1)
    ax1.axhline(NOMINAL_PICP, color=NEUTRAL["ref"], lw=LW["ref"], ls="--",
                alpha=0.8, label=f"Nominal {int(NOMINAL_PICP*100)}%")
    ax1.axvline(THRESHOLD_N, color=C["orchid"], lw=LW["threshold"], ls=":",
                alpha=0.9, zorder=1)
    ax1.text(THRESHOLD_N + 0.6, PICP_YLIM[0] + 0.03, r"$n^* \approx 10$",
             fontsize=FONTSIZE["annotation"], color=C["orchid"], va="bottom")

    lo = np.clip([p - s for p, s in zip(boot_picp, boot_pstd)], PICP_YLIM[0], None)
    hi = np.clip([p + s for p, s in zip(boot_picp, boot_pstd)], None, PICP_YLIM[1])
    band(ax1, ns_arr, lo, hi, C["bootstrap"])

    glow_line(ax1, ns_arr, boot_picp, C["bootstrap"],
              marker=MARKER_BOOT, ms=MARKER_SIZE, label="Bootstrap")
    glow_line(ax1, ns_bayes_arr, bayes_picp, C["bayesian"], ls="--",
              marker=MARKER_BAYES, ms=MARKER_SIZE, label="Bayesian NUTS")

    ax1.set_xlabel(r"Sample size $n$")
    ax1.set_ylabel("PICP at 90% nominal")
    ax1.set_xlim(ns_arr[0] - 1, ns_arr[-1] + 1)
    ax1.set_ylim(*PICP_YLIM)
    ax1.legend(loc="lower right", **LEGEND_KW)
    label_panel(ax1, "a")

    # ── Panel (b): MPIW ──────────────────────────────────────────────────────
    deck_axes(ax2)
    ax2.axvline(THRESHOLD_N, color=C["orchid"], lw=LW["threshold"], ls=":",
                alpha=0.9, zorder=1)
    ax2.text(THRESHOLD_N + 0.6, MPIW_YLIM[1] * 0.92, r"$n^* \approx 10$",
             fontsize=FONTSIZE["annotation"], color=C["orchid"], va="top")

    lo_m = np.clip([m - s for m, s in zip(boot_mpiw, boot_mstd)], 0, None)
    hi_m = np.clip([m + s for m, s in zip(boot_mpiw, boot_mstd)], None, MPIW_YLIM[1])
    band(ax2, ns_arr, lo_m, hi_m, C["bootstrap"])

    glow_line(ax2, ns_arr, boot_mpiw, C["bootstrap"],
              marker=MARKER_BOOT, ms=MARKER_SIZE, label="Bootstrap")
    glow_line(ax2, ns_bayes_arr, bayes_mpiw, C["bayesian"], ls="--",
              marker=MARKER_BAYES, ms=MARKER_SIZE, label="Bayesian NUTS")

    ax2.set_xlabel(r"Sample size $n$")
    ax2.set_ylabel("MPIW at 90% nominal (MPa)")
    ax2.set_xlim(ns_arr[0] - 1, ns_arr[-1] + 1)
    ax2.set_ylim(*MPIW_YLIM)
    ax2.legend(loc="upper right", **LEGEND_KW)
    label_panel(ax2, "b")

    save_deck(fig, OUT_BASE)


if __name__ == "__main__":
    from figures.deck_style import MARKER_SIZE as MARKER_SIZE  # noqa: E402
    main()
