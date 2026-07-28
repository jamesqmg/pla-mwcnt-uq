#!/usr/bin/env python3
"""figures/build_fig_method_selection.py — method-selection decision flow.

Camera-ready committee ask #2: a boxed/visual method-selection guideline.
Built in matplotlib via style.py so it shares the EXACT palette, font (Helvetica
Neue), flat scientific look, and the no-embedded-title convention of every other
paper figure. Honest post-retraction logic: multi-route -> hierarchical;
single-route -> bootstrap (cost) vs Bayesian (safety-critical). No n*=10 cliff.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import (  # noqa: E402
    PALETTE as C, FLOWCHART as F, FONTSIZE, LW, apply_style, save_figure, arrow_main,
)


def _box(ax, x, y, w, h, *, fc, ec, rounded, title, title_color, body=None, lw=1.4):
    box = "round,pad=0.015,rounding_size=0.10" if rounded else "square,pad=0.015"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=box,
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    if body is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=FONTSIZE["axis_label"], fontweight="bold",
                color=title_color, zorder=3, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center",
                fontsize=FONTSIZE["title"], fontweight="bold",
                color=title_color, zorder=3)
        ax.text(x + w / 2, y + h * 0.33, body, ha="center", va="center",
                fontsize=FONTSIZE["annotation"], color=C["dark"], zorder=3,
                linespacing=1.4)


def _edge(ax, p0, p1, label, lx, ly):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=arrow_main(C["mid"]), zorder=1)
    ax.text(lx, ly, label, fontsize=FONTSIZE["annotation"], color=C["mid"],
            ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=4)


def draw(ax) -> None:
    ax.set_xlim(0, 6.7)
    ax.set_ylim(0, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # decision nodes — white, dark edge, rounded
    _box(ax, 0.10, 2.18, 2.25, 0.80, fc="white", ec=C["dark"], rounded=True,
         title="Does the data span\nmultiple processing routes?", title_color=C["dark"])
    _box(ax, 0.10, 0.58, 2.25, 0.80, fc="white", ec=C["dark"], rounded=True,
         title="Single-route data:\nwhat is the priority?", title_color=C["dark"])

    # terminal nodes — canonical light tint fill + saturated method-color edge
    _box(ax, 3.00, 2.22, 3.60, 1.02, fc=F["terminal"], ec=C["hier"], rounded=False,
         title="Hierarchical Bayesian", title_color=C["hier"],
         body=r"Recovers per-route slopes (2.2$\times$ FDM vs non-FDM)"
              "\nand stabilises small groups bootstrap cannot fit")
    _box(ax, 3.00, 1.18, 3.60, 0.82, fc=F["main_fill"], ec=C["bootstrap"], rounded=False,
         title="Bootstrap", title_color=C["bootstrap"],
         body=r"Sharpest intervals, ~350$\times$ cheaper;"
              "\noverconfident — under-covers (PICP 0.846)")
    _box(ax, 3.00, 0.14, 3.60, 0.82, fc=F["bayes_fill"], ec=C["bayesian"], rounded=False,
         title="Bayesian NUTS", title_color=C["bayesian"],
         body="Conservative coverage guarantee (PICP 1.000)"
              "\nfor qualification margins; wider, costlier")

    # edges
    _edge(ax, (2.35, 2.58), (3.00, 2.73), "Yes", 2.64, 2.84)
    _edge(ax, (1.22, 2.18), (1.22, 1.38), "No", 1.44, 1.78)
    _edge(ax, (2.35, 1.05), (3.00, 1.56), r"cost · $n\geq$20", 2.74, 1.52)
    _edge(ax, (2.35, 0.92), (3.00, 0.56), "safety-critical", 2.52, 0.46)


def main(out: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.7, 3.5))
    draw(ax)
    save_figure(fig, out, formats=("pdf", "svg", "png"))


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "fig_method_selection")
