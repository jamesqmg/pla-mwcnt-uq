"""
Figure 5 — Dataset Filter Attrition Flow (PRISMA-style)
Thesis §3.1 / §2.3 — transparent account of how the literature-extracted master
dataset is filtered to the n=26 analysis pairs.

Run:  python figures/fig5_attrition_flow.py
Outputs: figures/fig5_attrition_flow.svg
         figures/fig5_attrition_flow.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from figures.style import (  # noqa: E402
    PALETTE as C, SIZES, LW, DPI_PNG,
    apply_style, save_figure, arrow_main, arrow_secondary,
    FONTSIZE, LEGEND_KW,
)

# ── Parameterization ─────────────────────────────────────────────────────────
# Box geometry (figure-fraction coordinates)
BOX_W          = 0.42   # main-chain box width
BOX_H          = 0.075  # box height
SIDE_BOX_W     = 0.30   # excluded/branching box width
SIDE_BOX_H     = 0.065

# Colors — from style bible
COLOR_MAIN     = C["include"]    # emerald teal — main inclusion chain
COLOR_EXCLUDE  = C["exclude"]    # vivid coral  — excluded / dropped
COLOR_BRANCH   = C["branch"]     # electric blue — side use (subsampling / hierarchical parent)
COLOR_TEXT_LT  = "white"
ALPHA_BOX      = 0.92

# Arrow — from style bible
ARROW_STYLE    = arrow_main(C["dark"])
SIDE_ARROW     = arrow_secondary(C["mid"])

# Figure
FIG_W, FIG_H   = SIZES["tall_flow"]
# ─────────────────────────────────────────────────────────────────────────────


def _box(ax, x, y, w, h, color, text, fontsize=None, textcolor=None):
    """Draw a rounded rectangle with centred text. (x, y) = bottom-left corner."""
    tc = textcolor if textcolor else COLOR_TEXT_LT
    fs = fontsize if fontsize is not None else FONTSIZE["axis_label"]
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010",
        linewidth=LW["spine"], edgecolor=C["dark"],
        facecolor=color, alpha=ALPHA_BOX, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs,
            color=tc, zorder=4, wrap=False,
            multialignment="center")


def _arrow(ax, x0, y0, x1, y1, props=None):
    if props is None:
        props = ARROW_STYLE
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=props, zorder=2)


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(-0.02, 1.10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Layout constants ────────────────────────────────────────────────────
    cx = 0.29                        # main-chain box left edge
    sx = cx + BOX_W + 0.04          # side-box left edge
    steps = [                        # (y_bottom of each main box, label text)
        (0.855, "Phase A: Scopus literature search\n100 papers screened"),
        (0.730, "Phase B: detail extraction\n~21 papers extracted\nmaster_dataset.csv — 45 rows\n(44 complete)"),
        (0.585, "Quality filter: HIGH tier only\nmaster_dataset_strict.csv — 35 rows"),
        (0.415, "data_loader filter:\nPLA + FDM/FFF + D638/ISO 527\ncomplete (x, y) pairs"),
        (0.255, "Analysis dataset\n$n$ = 26 pairs\nPhase 1 Bootstrap · Phase 2 NUTS"),
    ]

    for x_b, label in steps:
        c = COLOR_MAIN if x_b > 0.27 else C["violet"]   # violet marks the final analysis endpoint
        _box(ax, cx, x_b, BOX_W, BOX_H, c, label)

    # Arrows between main boxes — from bottom of current to top of next
    for i in range(len(steps) - 1):
        mid_x = cx + BOX_W / 2
        _arrow(ax,
               mid_x, steps[i][0],              # bottom of current box
               mid_x, steps[i + 1][0] + BOX_H)  # top of next box

    # ── Exclusion / branching side boxes ───────────────────────────────────
    # Phase A exclusions
    excl_y = 0.850
    _box(ax, sx, excl_y, SIDE_BOX_W, SIDE_BOX_H, COLOR_EXCLUDE,
         "45 EXCLUDED (wrong system,\nno data, non-peer-reviewed)\n54 FLAG for detail review\n1 direct INCLUDE",
         fontsize=FONTSIZE["annotation"], textcolor=COLOR_TEXT_LT)
    mid_main_A = steps[0][0] + BOX_H / 2
    _arrow(ax, cx + BOX_W, mid_main_A, sx, excl_y + SIDE_BOX_H / 2, SIDE_ARROW)

    # master_dataset.csv branches to subsampling
    sub_y = 0.700
    _box(ax, sx, sub_y, SIDE_BOX_W, SIDE_BOX_H, COLOR_BRANCH,
         "-> Subsampling n-ladder\nparent pool (§3.3)\n$n$=44 all-processing rows",
         fontsize=FONTSIZE["annotation"], textcolor=COLOR_TEXT_LT)
    mid_main_B = steps[1][0] + BOX_H / 2
    _arrow(ax, cx + BOX_W, mid_main_B, sx, sub_y + SIDE_BOX_H / 2, SIDE_ARROW)

    # master_dataset_strict.csv branches to hierarchical
    hier_y = 0.555
    _box(ax, sx, hier_y, SIDE_BOX_W, SIDE_BOX_H, COLOR_BRANCH,
         "-> Hierarchical model\nparent pool (§3.5b)\nFDM $n$=34 / non-FDM $n$=10",
         fontsize=FONTSIZE["annotation"], textcolor=COLOR_TEXT_LT)
    mid_main_C = steps[2][0] + BOX_H / 2
    _arrow(ax, cx + BOX_W, mid_main_C, sx, hier_y + SIDE_BOX_H / 2, SIDE_ARROW)

    # ── Legend ──────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=COLOR_MAIN,    label="Inclusion chain",   edgecolor=C["dark"], alpha=ALPHA_BOX),
        Patch(facecolor=COLOR_EXCLUDE, label="Excluded records",  edgecolor=C["dark"], alpha=ALPHA_BOX),
        Patch(facecolor=COLOR_BRANCH,  label="Secondary uses",    edgecolor=C["dark"], alpha=ALPHA_BOX),
    ]
    ax.legend(handles=legend_els, loc="lower left", handlelength=1.2, **LEGEND_KW)

    save_figure(fig, Path(__file__).with_suffix(""))


if __name__ == "__main__":
    main()
