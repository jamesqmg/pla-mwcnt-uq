"""
Figure 2 — Methodological Positioning Map (9 anchor papers + this thesis)
Thesis §2.4 — two-dimensional conceptual space:
  X: methodological breadth (narrow / single material-system → broad / multi-polymer)
  Y: data-scarcity engagement (no quantitative sweep → explicit n-ladder)

Run:  python figures/build_fig2_positioning.py
Outputs: figures/fig2_positioning.svg
         figures/fig2_positioning.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from figures.style import (
    PALETTE as C, SIZES, LW, DPI_PNG,
    apply_style, save_figure,
    FONTSIZE, LEGEND_KW,
)

# ── Paper data (confirmed from Table 1 / §2.4 text) ─────────────────────────
# (x, y): x=breadth [0=narrow .. 1=broad], y=scarcity engagement [0=none .. 1=explicit]
PAPERS = [
    # Broad, no quantitative sweep (lower-right)
    dict(label="Tang et al.\n(2025)",           x=0.82, y=0.13, group="broad"),
    dict(label="Hiremath et al.\n(2025)",        x=0.74, y=0.08, group="broad"),
    dict(label="Champa-Bujaico\net al. (2022)",  x=0.64, y=0.20, group="broad"),
    # Narrow, no explicit sweep (lower-left) — works with small n but no n-ladder
    dict(label="Catalanotti\n(2022)",            x=0.22, y=0.30, group="narrow"),
    dict(label="Catalanotti\n(2024)",            x=0.18, y=0.38, group="narrow"),
    dict(label="Thomas et al.\n(2022)",          x=0.32, y=0.25, group="narrow"),
    dict(label="Doh et al.\n(2020)",             x=0.16, y=0.14, group="narrow"),
    dict(label="De Yro et al.\n(2023)",          x=0.10, y=0.08, group="narrow"),
    # Intermediate — explicit comparison but on FE simulation, not data-scarcity sweep
    dict(label="Chahar &\nMukhopadhyay (2026)",  x=0.54, y=0.55, group="mid"),
]

THESIS = dict(label="This thesis\n(Ilio, 2026)", x=0.10, y=0.85)

# ── Quadrant shading ─────────────────────────────────────────────────────────
# Four quadrants: split at x=0.45, y=0.50
# Labels placed at interior corners well away from data clusters.
QUAD_ALPHA = 0.07
QUADS = [
    # (x0, y0, w, h, color, label, label_xy_axes, ha, va)
    (0.00, 0.50, 0.45, 0.50, C["bootstrap"],  "Narrow /\nExplicit",   (0.40, 0.97), "right", "top"),
    (0.45, 0.50, 0.55, 0.50, C["violet"],     "Broad /\nExplicit",    (0.96, 0.97), "right", "top"),
    (0.00, 0.00, 0.45, 0.50, C["mid"],        "Narrow /\nNo sweep",   (0.40, 0.54), "right", "bottom"),
    (0.45, 0.00, 0.55, 0.50, C["bayesian"],   "Broad /\nNo sweep",    (0.96, 0.54), "right", "bottom"),
]

# ── Text offsets: (dx, dy) in data units added to marker position ────────────
# Each entry is (ha_anchor, (dx, dy)) — ha_anchor drives alignment.
OFFSETS = {
    # lower-right cluster: fan outward to avoid overlap
    "Tang et al.\n(2025)":             ("left",  ( 0.03,   0.03)),   # northeast of marker
    "Hiremath et al.\n(2025)":         ("right", (-0.03,  -0.07)),   # below-left
    "Champa-Bujaico\net al. (2022)":   ("left",  ( 0.03,   0.04)),   # northeast
    # lower-left cluster
    "Catalanotti\n(2022)":             ("left",  ( 0.03,  -0.02)),   # right
    "Catalanotti\n(2024)":             ("left",  ( 0.03,   0.03)),   # upper-right
    "Thomas et al.\n(2022)":           ("left",  ( 0.03,   0.03)),   # upper-right
    "Doh et al.\n(2020)":              ("left",  ( 0.03,   0.03)),   # upper-right
    "De Yro et al.\n(2023)":           ("left",  ( 0.03,  -0.07)),   # below-right
    # middle
    "Chahar &\nMukhopadhyay (2026)":   ("left",  ( 0.03,   0.03)),   # upper-right
    # upper-left
    "This thesis\n(Ilio, 2026)":       ("left",  ( 0.04,   0.03)),   # right of star
}

GROUP_COLOR = {
    "broad":  C["bayesian"],   # coral-red — broad pool papers
    "narrow": C["mid"],        # medium gray — narrow-focus papers
    "mid":    C["amber"],      # amber — intermediate / partial
}
GROUP_MARKER = {
    "broad":  "o",
    "narrow": "s",
    "mid":    "D",
}


def main() -> None:
    apply_style()

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # ── Quadrant backgrounds ─────────────────────────────────────────────────
    for x0, y0, w, h, color, qlabel, lxy, ha, va in QUADS:
        rect = mpatches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="square,pad=0",
            facecolor=color, alpha=QUAD_ALPHA,
            edgecolor="none", zorder=0,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            lxy[0], lxy[1], qlabel,
            transform=ax.transAxes,
            fontsize=FONTSIZE["annotation"], color=color,
            ha=ha, va=va, alpha=0.70, style="italic",
            zorder=1,
        )

    # ── Quadrant dividers ────────────────────────────────────────────────────
    ax.axvline(0.45, color=C["light"], lw=LW["grid"], ls="--", zorder=1)
    ax.axhline(0.50, color=C["light"], lw=LW["grid"], ls="--", zorder=1)

    # ── Anchor papers ────────────────────────────────────────────────────────
    for p in PAPERS:
        color  = GROUP_COLOR[p["group"]]
        marker = GROUP_MARKER[p["group"]]
        ax.scatter(p["x"], p["y"],
                   s=60, marker=marker,
                   color=color, edgecolors=C["dark"],
                   linewidths=LW["marker_edge"],
                   zorder=4)
        ha_anchor, (dx, dy) = OFFSETS[p["label"]]
        ax.text(
            p["x"] + dx, p["y"] + dy,
            p["label"],
            fontsize=FONTSIZE["annotation"],
            color=C["dark"],
            ha=ha_anchor,
            va="center",
            zorder=5,
            linespacing=1.3,
        )

    # ── This thesis (star marker, prominent) ────────────────────────────────
    ax.scatter(THESIS["x"], THESIS["y"],
               s=200, marker="*",
               color=C["bootstrap"],
               edgecolors=C["dark"],
               linewidths=LW["marker_edge"],
               zorder=6)
    ha_t, (dx, dy) = OFFSETS[THESIS["label"]]
    ax.text(
        THESIS["x"] + dx, THESIS["y"] + dy,
        THESIS["label"],
        fontsize=FONTSIZE["axis_label"],
        fontweight="bold",
        color=C["bootstrap"],
        ha=ha_t, va="bottom",
        zorder=7,
        linespacing=1.3,
    )

    # ── "Empty intersection" annotation ─────────────────────────────────────
    ax.annotate(
        "Previously\nunoccupied",
        xy=(THESIS["x"] + 0.02, THESIS["y"] - 0.02),
        xytext=(0.26, 0.72),
        fontsize=FONTSIZE["annotation"],
        color=C["bootstrap"], alpha=0.75,
        ha="left", va="center",
        arrowprops=dict(
            arrowstyle="->",
            color=C["bootstrap"],
            lw=0.9,
            connectionstyle="arc3,rad=0.2",
        ),
        zorder=5,
    )

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel(
        r"Methodological breadth  $\rightarrow$" "\n"
        r"(narrow / single material-system $\longleftarrow\!\longrightarrow$ broad / multi-polymer benchmark)",
        fontsize=FONTSIZE["axis_label"],
        labelpad=8,
    )
    ax.set_ylabel(
        r"Data-scarcity engagement  $\rightarrow$" "\n"
        r"(no quantitative sweep $\longleftarrow\!\longrightarrow$ explicit $n$-ladder)",
        fontsize=FONTSIZE["axis_label"],
        labelpad=8,
    )

    # Remove numeric ticks — conceptual axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Keep only bottom + left spines (style bible)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        plt.scatter([], [], s=60, marker="o", color=C["bayesian"],
                    edgecolors=C["dark"], lw=LW["marker_edge"],
                    label="Broad-pool benchmark"),
        plt.scatter([], [], s=60, marker="s", color=C["mid"],
                    edgecolors=C["dark"], lw=LW["marker_edge"],
                    label="Narrow-focus precedent"),
        plt.scatter([], [], s=60, marker="D", color=C["amber"],
                    edgecolors=C["dark"], lw=LW["marker_edge"],
                    label="Intermediate / partial"),
        plt.scatter([], [], s=200, marker="*", color=C["bootstrap"],
                    edgecolors=C["dark"], lw=LW["marker_edge"],
                    label="This thesis"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.48, 0.98),
        **LEGEND_KW,
    )

    out = ROOT / "figures" / "fig2_positioning"
    save_figure(fig, out)


if __name__ == "__main__":
    main()
