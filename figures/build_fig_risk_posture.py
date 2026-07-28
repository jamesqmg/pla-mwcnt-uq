#!/usr/bin/env python3
"""figures/build_fig_risk_posture.py — risk-posture (calibration-width) plane.

One-glance summary of the n=26 primary comparison: bootstrap and Bayesian NUTS
sit on OPPOSITE sides of the perfect-calibration line. Bootstrap is sharper but
under-covers (overconfident); Bayesian is wider but over-covers (conservative).
The choice is risk posture, not aggregate skill. Numbers are the canonical LOO
values (Table I): bootstrap (MPIW 25.9, PICP 0.846); Bayesian (28.9, 1.000).

Exposes draw(ax) so it can be composed into the combined decision figure
(build_fig_decision.py). Standalone output: fig_risk_posture.png/.svg/.pdf
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import (  # noqa: E402
    PALETTE as C, FONTSIZE, LW, apply_style, save_figure,
)

# Canonical n=26 LOO numbers (Table I)
BOOT = dict(mpiw=25.9, picp=0.846)
BAYES = dict(mpiw=28.9, picp=1.000)
NOMINAL = 0.90


def draw(ax) -> None:
    xlim = (23.0, 31.0)
    ylim = (0.76, 1.04)

    # calibration zones (subtle tints) + the perfect-calibration line
    ax.axhspan(ylim[0], NOMINAL, color=C["bootstrap"], alpha=0.06, zorder=0)
    ax.axhspan(NOMINAL, ylim[1], color=C["bayesian"], alpha=0.06, zorder=0)
    ax.axhline(NOMINAL, color=C["dark"], lw=LW["ref"], ls="--", zorder=1)
    ax.text(xlim[1] - 0.15, NOMINAL + 0.004, "perfect calibration (90%)",
            ha="right", va="bottom", fontsize=FONTSIZE["annotation"], color=C["dark"])

    # zone labels
    ax.text(xlim[0] + 0.2, ylim[1] - 0.008, "conservative (over-covers)",
            ha="left", va="top", fontsize=FONTSIZE["annotation"], color=C["mid"])
    ax.text(xlim[0] + 0.2, ylim[0] + 0.008, "overconfident (under-covers)",
            ha="left", va="bottom", fontsize=FONTSIZE["annotation"], color=C["mid"])

    # method points on opposite sides of the calibration line
    ax.scatter([BOOT["mpiw"]], [BOOT["picp"]], s=160, color=C["bootstrap"],
               edgecolors="white", linewidth=1.0, zorder=4)
    ax.scatter([BAYES["mpiw"]], [BAYES["picp"]], s=160, color=C["bayesian"],
               marker="s", edgecolors="white", linewidth=1.0, zorder=4)
    ax.annotate("Bootstrap\n(25.9 MPa, 0.846)", xy=(BOOT["mpiw"], BOOT["picp"]),
                xytext=(BOOT["mpiw"], BOOT["picp"] - 0.028),
                ha="center", va="top", fontsize=FONTSIZE["annotation"],
                color=C["bootstrap"], fontweight="bold")
    ax.annotate("Bayesian NUTS\n(28.9 MPa, 1.000)", xy=(BAYES["mpiw"], BAYES["picp"]),
                xytext=(BAYES["mpiw"], BAYES["picp"] + 0.006),
                ha="center", va="bottom", fontsize=FONTSIZE["annotation"],
                color=C["bayesian"], fontweight="bold")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Mean interval width, MPIW at 90%  (MPa)  —  sharper $\\rightarrow$ wider")
    ax.set_ylabel("Coverage, PICP at 90%  (LOO)")


def main(out: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    draw(ax)
    save_figure(fig, out, formats=("png", "svg", "pdf"))


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "fig_risk_posture")
