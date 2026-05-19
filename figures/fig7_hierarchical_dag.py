"""
Figure 9 — Hierarchical Bayesian Model DAG
Thesis §3.5b — directed acyclic graph showing partial-pooling structure.
Conference paper: thesis-only (conference paper uses fig3_hierarchical_shrinkage instead).

Shows three levels:
  1. Population hyperpriors (β_pop, τ_β, σ_obs)
  2. Group-level parameters: non-centered reparameterisation (z_g ~ N(0,1),
     β_g = β_pop + τ_β · z_g) for FDM (n=34) and non-FDM (n=10) groups
  3. Observation likelihood: y_i ~ Normal(α + β_g · x_i, σ_obs)

Non-centered form is shown because that is the actual implementation in
src/uq/hierarchical.py — the centered form was abandoned due to funnel geometry.

Run:  python figures/fig9_hierarchical_dag.py
Outputs: figures/fig9_hierarchical_dag.svg
         figures/fig9_hierarchical_dag.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from figures.style import (  # noqa: E402
    PALETTE as C, SIZES, LW, DPI_PNG, FONTSIZE,
    apply_style, save_figure, LEGEND_KW,
)

# ── Layout constants ─────────────────────────────────────────────────────────
FIG_W, FIG_H = SIZES["dag"]   # (5.5, 4.2)

# Colors for each DAG level
COL_HYPER  = C["violet"]     # population hyperpriors
COL_FDM    = C["bootstrap"]  # FDM group
COL_NFDM   = C["bayesian"]   # non-FDM group
COL_OBS    = C["hier"]       # observation likelihood (shaded = observed)
COL_SHARED = C["amber"]      # shared parameters (σ_obs)

ALPHA_BOX   = 0.88
ALPHA_OBS   = 0.70           # slightly more transparent — "observed" nodes
BOX_RADIUS  = 0.04           # FancyBboxPatch corner radius
PLATE_ALPHA = 0.10           # plate boundary fill alpha


def _rounded_box(
    ax: plt.Axes,
    cx: float, cy: float,
    w: float, h: float,
    color: str,
    text: str,
    fontsize: int | None = None,
    alpha: float = ALPHA_BOX,
    edgecolor: str | None = None,
    textcolor: str = "white",
    lw: float = LW["spine"],
) -> None:
    """Draw a rounded rectangle centred at (cx, cy) with text."""
    fs = fontsize or FONTSIZE["annotation"]
    ec = edgecolor or C["dark"]
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={BOX_RADIUS}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=color,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text,
            ha="center", va="center",
            fontsize=fs, color=textcolor,
            zorder=4, multialignment="center")


def _arrow(
    ax: plt.Axes,
    x0: float, y0: float,
    x1: float, y1: float,
    color: str = C["dark"],
) -> None:
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=LW["schematic"],
            shrinkA=5,
            shrinkB=5,
        ),
        zorder=2,
    )


def _plate(
    ax: plt.Axes,
    x0: float, y0: float,
    w: float, h: float,
    label: str,
    color: str,
) -> None:
    """Draw a plate boundary (dashed rectangle with corner label)."""
    rect = plt.Rectangle(
        (x0, y0), w, h,
        linewidth=LW["schematic"] * 0.8,
        edgecolor=color,
        facecolor=color,
        alpha=PLATE_ALPHA,
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(rect)
    ax.text(x0 + w - 0.01, y0 + 0.01, label,
            ha="right", va="bottom",
            fontsize=FONTSIZE["annotation"] - 1,
            color=color, style="italic", zorder=5)


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(-0.22, 1.05)   # negative left margin gives room for row labels
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Level positions (y-centres of each row) ──────────────────────────────
    Y_HYPER  = 0.82   # row 1: population hyperpriors
    Y_NC     = 0.58   # row 2: non-centred auxiliary z_g
    Y_GROUP  = 0.37   # row 3: group-level β_g
    Y_OBS    = 0.12   # row 4: observation likelihood

    # ── Box widths / heights ──────────────────────────────────────────────────
    BW_HYPER = 0.22   # hyperprior box width
    BH_HYPER = 0.11
    BW_NC    = 0.23
    BH_NC    = 0.10
    BW_GROUP = 0.23
    BH_GROUP = 0.10
    BW_OBS   = 0.34
    BH_OBS   = 0.12

    # ── Row 1: Population hyperpriors ─────────────────────────────────────────
    # β_pop | τ_β | σ_obs (shared)
    CX_BPOP  = 0.22
    CX_TAU   = 0.50
    CX_SIGMA = 0.80

    _rounded_box(ax, CX_BPOP, Y_HYPER, BW_HYPER, BH_HYPER, COL_HYPER,
                 r"$\beta_{\mathrm{pop}} \sim \mathcal{N}(0,1)$",
                 fontsize=FONTSIZE["annotation"])
    _rounded_box(ax, CX_TAU, Y_HYPER, BW_HYPER, BH_HYPER, COL_HYPER,
                 r"$\tau_\beta \sim \mathrm{HalfNormal}(0.5)$",
                 fontsize=FONTSIZE["annotation"])
    _rounded_box(ax, CX_SIGMA, Y_HYPER, BW_HYPER - 0.02, BH_HYPER, COL_SHARED,
                 r"$\sigma_{\mathrm{obs}} \sim \mathrm{HalfNormal}(1)$",
                 fontsize=FONTSIZE["annotation"])

    # Level label
    ax.text(-0.01, Y_HYPER, "Population\nhyperpriors",
            ha="right", va="center", fontsize=FONTSIZE["annotation"] - 1,
            color=COL_HYPER, fontweight="bold")

    # ── Group plates ──────────────────────────────────────────────────────────
    # FDM plate: x ∈ [0.03, 0.48]
    _plate(ax, 0.03, Y_OBS - BH_OBS / 2 - 0.04,
           0.43, (Y_NC + BH_NC / 2 + 0.06) - (Y_OBS - BH_OBS / 2 - 0.04),
           r"FDM  $n$=34", COL_FDM)
    # non-FDM plate: x ∈ [0.53, 0.97]
    _plate(ax, 0.53, Y_OBS - BH_OBS / 2 - 0.04,
           0.43, (Y_NC + BH_NC / 2 + 0.06) - (Y_OBS - BH_OBS / 2 - 0.04),
           r"non-FDM  $n$=10", COL_NFDM)

    # ── Row 2: Non-centred auxiliary z_g ─────────────────────────────────────
    CX_ZFDM  = 0.24
    CX_ZNFDM = 0.76

    _rounded_box(ax, CX_ZFDM, Y_NC, BW_NC, BH_NC, COL_FDM,
                 r"$z_{\mathrm{FDM}} \sim \mathcal{N}(0,1)$",
                 fontsize=FONTSIZE["annotation"])
    _rounded_box(ax, CX_ZNFDM, Y_NC, BW_NC, BH_NC, COL_NFDM,
                 r"$z_{\mathrm{non\text{-}FDM}} \sim \mathcal{N}(0,1)$",
                 fontsize=FONTSIZE["annotation"])

    ax.text(-0.01, Y_NC, "Non-centred\nauxiliary",
            ha="right", va="center", fontsize=FONTSIZE["annotation"] - 1,
            color=C["mid"], fontstyle="italic")

    # ── Row 3: Group-level β_g ────────────────────────────────────────────────
    CX_BGFDM  = 0.24
    CX_BGNFDM = 0.76

    _rounded_box(ax, CX_BGFDM, Y_GROUP, BW_GROUP, BH_GROUP, COL_FDM,
                 r"$\beta_{\mathrm{FDM}} = \beta_{\mathrm{pop}} + \tau_\beta z_{\mathrm{FDM}}$",
                 fontsize=FONTSIZE["annotation"] - 1)
    _rounded_box(ax, CX_BGNFDM, Y_GROUP, BW_GROUP + 0.06, BH_GROUP, COL_NFDM,
                 r"$\beta_{\mathrm{nFDM}} = \beta_{\mathrm{pop}} + \tau_\beta z_{\mathrm{nFDM}}$",
                 fontsize=FONTSIZE["annotation"] - 1)

    ax.text(-0.01, Y_GROUP, "Group slope\n(non-centred)",
            ha="right", va="center", fontsize=FONTSIZE["annotation"] - 1,
            color=C["mid"])

    # ── Row 4: Observation likelihood ─────────────────────────────────────────
    CX_OFDM  = 0.24
    CX_ONFDM = 0.76

    # Shaded (observed) boxes — darker edgecolor signals observed nodes
    _rounded_box(ax, CX_OFDM, Y_OBS, BW_OBS, BH_OBS, COL_OBS,
                 r"$y_i \sim \mathcal{N}(\alpha + \beta_{\mathrm{FDM}}\, x_i,\; \sigma_{\mathrm{obs}})$",
                 fontsize=FONTSIZE["annotation"] - 1,
                 alpha=ALPHA_OBS, edgecolor=C["dark"], lw=LW["data"] * 0.8)
    _rounded_box(ax, CX_ONFDM, Y_OBS, BW_OBS, BH_OBS, COL_OBS,
                 r"$y_j \sim \mathcal{N}(\alpha + \beta_{\mathrm{nFDM}}\, x_j,\; \sigma_{\mathrm{obs}})$",
                 fontsize=FONTSIZE["annotation"] - 1,
                 alpha=ALPHA_OBS, edgecolor=C["dark"], lw=LW["data"] * 0.8)

    ax.text(-0.01, Y_OBS, "Observation\nlikelihood\n(observed)",
            ha="right", va="center", fontsize=FONTSIZE["annotation"] - 1,
            color=C["mid"])

    # ── Arrows: hyperpriors → z_g ─────────────────────────────────────────────
    # β_pop → z_FDM (through group equations)
    _arrow(ax, CX_BPOP, Y_HYPER - BH_HYPER / 2,
               CX_ZFDM, Y_NC + BH_NC / 2, C["mid"])
    _arrow(ax, CX_BPOP, Y_HYPER - BH_HYPER / 2,
               CX_ZNFDM, Y_NC + BH_NC / 2, C["mid"])
    # τ_β → z_FDM and z_nFDM
    _arrow(ax, CX_TAU, Y_HYPER - BH_HYPER / 2,
               CX_ZFDM, Y_NC + BH_NC / 2, C["mid"])
    _arrow(ax, CX_TAU, Y_HYPER - BH_HYPER / 2,
               CX_ZNFDM, Y_NC + BH_NC / 2, C["mid"])

    # ── Arrows: z_g → β_g ────────────────────────────────────────────────────
    _arrow(ax, CX_ZFDM, Y_NC - BH_NC / 2,
               CX_BGFDM, Y_GROUP + BH_GROUP / 2, COL_FDM)
    _arrow(ax, CX_ZNFDM, Y_NC - BH_NC / 2,
               CX_BGNFDM, Y_GROUP + BH_GROUP / 2, COL_NFDM)

    # ── Arrows: β_g → observation ────────────────────────────────────────────
    _arrow(ax, CX_BGFDM, Y_GROUP - BH_GROUP / 2,
               CX_OFDM, Y_OBS + BH_OBS / 2, COL_FDM)
    _arrow(ax, CX_BGNFDM, Y_GROUP - BH_GROUP / 2,
               CX_ONFDM, Y_OBS + BH_OBS / 2, COL_NFDM)

    # ── Arrow: σ_obs → both observation boxes (shared noise) ─────────────────
    _arrow(ax, CX_SIGMA, Y_HYPER - BH_HYPER / 2,
               CX_OFDM + BW_OBS / 2, Y_OBS + BH_OBS / 2, COL_SHARED)
    _arrow(ax, CX_SIGMA, Y_HYPER - BH_HYPER / 2,
               CX_ONFDM + BW_OBS / 2, Y_OBS + BH_OBS / 2, COL_SHARED)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=COL_HYPER, alpha=ALPHA_BOX, label="Population hyperpriors"),
        mpatches.Patch(color=COL_FDM,   alpha=ALPHA_BOX, label=r"FDM group ($n$=34)"),
        mpatches.Patch(color=COL_NFDM,  alpha=ALPHA_BOX, label=r"non-FDM group ($n$=10)"),
        mpatches.Patch(color=COL_SHARED, alpha=ALPHA_BOX,
                       label=r"Shared noise $\sigma_\mathrm{obs}$"),
        mpatches.Patch(color=COL_OBS,   alpha=ALPHA_OBS,
                       label="Observed data (shaded)", linewidth=LW["data"] * 0.8,
                       edgecolor=C["dark"]),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.04),
              ncol=3, handlelength=1.2, **LEGEND_KW)

    save_figure(fig, Path(__file__).with_suffix(""), pad=1.8)


if __name__ == "__main__":
    main()
