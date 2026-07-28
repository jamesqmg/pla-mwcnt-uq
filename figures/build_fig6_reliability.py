"""
build_fig6_reliability.py — Thesis Figure 6
Reliability (calibration) diagram: Bootstrap LOO vs Bayesian exact-LOO.

Both methods are evaluated strictly out-of-sample and read from the stored,
authoritative result files (no live recompute, so the figure cannot drift from
the numbers quoted in the manuscript):
  Bootstrap LOO   → results/phase1_phase2_loo.json  (ece_by_coverage, B = 1000)
  Bayesian        → results/phase2_bayesian_loo.json (reliability_curve, exact
                    LOO posterior-predictive quantiles, 26 refits on n-1)

Panel (a): empirical PICP vs nominal coverage α at the six reported levels, with
           the shaded miscalibration region against the perfect-calibration
           diagonal. ECE = mean |PICP − α| over the six levels.
Panel (b): coverage gap |PICP − α| per level — the ECE decomposition.

Output: figures/fig6_reliability.svg, figures/fig6_reliability.png (300 DPI)

Style anchor: Tang 2025 — bar charts, categorical color semantics.
              Catalanotti 2024 — open axes, frameless legend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIGURES = Path(__file__).parent
REPO    = FIGURES.parent
sys.path.insert(0, str(FIGURES))
from style import (
    apply_style, label_panel, save_figure,
    PALETTE, SIZES, LW, LEGEND_KW, FONTSIZE,
)

C = PALETTE

LEVELS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])


def _bootstrap_picp() -> np.ndarray:
    """Bootstrap LOO PICP per level, reconstructed from the stored coverage gaps.
    Bootstrap under-covers at every level, so PICP = α − gap (gap = |PICP − α|)."""
    with open(REPO / "results" / "phase1_phase2_loo.json") as fh:
        gap = json.load(fh)["phase1_bootstrap_loo"]["ece_by_coverage"]
    gaps = np.array([gap[f"{lv:g}"] for lv in LEVELS])
    return LEVELS - gaps


def _bayesian_picp() -> np.ndarray:
    """Bayesian exact-LOO PICP per level from the stored reliability curve."""
    with open(REPO / "results" / "phase2_bayesian_loo.json") as fh:
        curve = dict(json.load(fh)["reliability_curve"])
    return np.array([curve[lv] for lv in LEVELS])


def _ece(picp: np.ndarray) -> float:
    return float(np.mean(np.abs(picp - LEVELS)))


def build() -> None:
    apply_style()

    boot_picp  = _bootstrap_picp()
    bayes_picp = _bayesian_picp()
    boot_ece   = _ece(boot_picp)
    bayes_ece  = _ece(bayes_picp)
    print(f"Bootstrap ECE = {boot_ece:.3f} | Bayesian ECE = {bayes_ece:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=SIZES["wide"])

    # ── Panel (a): calibration curves at the six reported levels ─────────────
    ax = axes[0]
    diag = np.array([0.0, 1.0])
    ax.plot(diag, diag, color=C["dark"], lw=LW["ref"], ls="--",
            label="Perfect calibration", zorder=2)

    ax.fill_between(LEVELS, LEVELS, boot_picp,
                    color=C["bootstrap"], alpha=0.14, edgecolor="none", zorder=2)
    ax.plot(LEVELS, boot_picp, color=C["bootstrap"], lw=LW["data"],
            marker="o", markeredgecolor="white", markeredgewidth=LW["marker_edge"],
            label=f"Bootstrap LOO  (ECE = {boot_ece:.3f})", zorder=3)

    ax.fill_between(LEVELS, LEVELS, bayes_picp,
                    color=C["bayesian"], alpha=0.14, edgecolor="none", zorder=2)
    ax.plot(LEVELS, bayes_picp, color=C["bayesian"], lw=LW["data"],
            marker="s", markeredgecolor="white", markeredgewidth=LW["marker_edge"],
            label=f"Bayesian exact-LOO  (ECE = {bayes_ece:.3f})", zorder=3)

    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.35, 1.04)
    ax.set_xlabel(r"Nominal coverage $\alpha$")
    ax.set_ylabel("Empirical PICP")
    ax.legend(**{**LEGEND_KW, "fontsize": FONTSIZE["annotation"]}, loc="upper left")
    label_panel(ax, "a")

    # ── Panel (b): coverage gap bars (ECE decomposition) ─────────────────────
    ax2 = axes[1]
    gap_boot  = np.abs(boot_picp  - LEVELS)
    gap_bayes = np.abs(bayes_picp - LEVELS)

    w = 0.013
    ax2.bar(LEVELS - w, gap_boot,  width=w * 1.8,
            color=C["bootstrap"], alpha=0.88, label="Bootstrap (LOO)", zorder=3)
    ax2.bar(LEVELS + w, gap_bayes, width=w * 1.8,
            color=C["bayesian"],  alpha=0.88, label="Bayesian (exact LOO)", zorder=3)
    ax2.axhline(0, color=C["dark"], lw=LW["spine"])
    ax2.set_xlim(0.47, 1.0)
    ax2.set_xlabel(r"Nominal coverage $\alpha$")
    ax2.set_ylabel(r"$|\mathrm{PICP} - \alpha|$  (coverage gap)")
    ax2.legend(**LEGEND_KW)
    label_panel(ax2, "b")

    save_figure(fig, FIGURES / "fig6_reliability")
    print("Done → figures/fig6_reliability.svg + .png")


if __name__ == "__main__":
    build()
