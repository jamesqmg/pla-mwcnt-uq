"""
figures/style.py — Thesis 1 Figure Style Bible
================================================
The SINGLE SOURCE OF TRUTH for all figure aesthetics. Every figure script
MUST import from here. Never hardcode colors, sizes, line weights, or font
choices in individual figure scripts.

ANCHOR PAPERS (read PDFs in data/pdfs/key_anchors/ before editing this file):
  Catalanotti 2024  catalanotti_2024_composites-sci-tech.pdf
    → Sans-serif throughout (Helvetica/Arial body).
    → Only left + bottom spines (clean open axes).
    → Variable names (X_i, n, θ) in math italic.
    → Blue filled boxes for main flow; red/pink dashed for stochastic/Bayesian blocks.
    → Minimal captions: "Fig. N. [noun phrase]." — no title inside figure.
    → Panel labels bold (a)/(b) upper-left outside axes.

  Tang 2025  tang_2025_jcim.pdf
    → Two-column ACS journal; clean white background.
    → Bar charts: categorical color semantics (blue=accuracy, green=rank, red=calibration).
    → Error bars on aggregated bar/scatter.
    → Variable n in math italic within formulae and captions.
    → Caption: "Figure N. [noun phrase]. Panel (a) ... Panel (b) ... [takeaway]."

  Thomas 2022  thomas_2022_composites-sci-tech.pdf
    → Parity plots and hierarchical DAG anchor.
    → Scatter + PI band style for parity plots.

  Catalanotti 2022  catalanotti_2022_composites-parta.pdf
    → Bootstrap PI band style anchor.

MATH ITALIC NOTE (critical):
  Variable names like n, n*, β, τ, σ MUST be wrapped in $...$ in all
  ax.set_xlabel/ylabel, ax.text, and annotation calls. The mathtext.fontset
  is "dejavusans" (changed from "stixsans" — STIXSans is not bundled in
  the local matplotlib install; DejaVu Sans Math is always available and
  renders clean sans-serif italic that is visibly distinct from upright text).

CONVENTION (always enforced):
  - No embedded captions. ax.set_title() / fig.suptitle() for "Figure N. ..."
    is FORBIDDEN. Only panel labels (a)/(b) go inside the figure.
  - Top and right spines are OFF by default (Catalanotti 2024 style).
  - Use LEGEND_KW for all legends (frameless, compact).
  - Use MATH dict for common variable strings.

FIGURE NUMBERING (thesis v6-final, locked 2026-05-17):
  Thesis Fig 1  → PSPP Framework                       (fig1_pspp_diagram.png — user-created)
  Thesis Fig 2  → Methodological Positioning (9 anchors) (user-created)
  Thesis Fig 3  → End-to-end Computational Methodology  (fig3_methodology_overview.png)
  Thesis Fig 4  → Data Attrition Flow                   (fig5_attrition_flow.py — legacy name kept)
  Thesis Fig 5  → Parity Plots Bootstrap vs Bayesian    (build_fig5_parity.py)
  Thesis Fig 6  → Reliability / Calibration Curve       (build_fig6_reliability.py)
  Thesis Fig 7  → Hierarchical DAG                      (fig9_hierarchical_dag.py — legacy name kept)
  Thesis Fig 8  → Hierarchical Shrinkage                (fig3_hierarchical_shrinkage.py — legacy name)
  Thesis Fig 9  → Divergence Curves PICP(n)/MPIW(n)    (fig8_divergence_curves.py — legacy name)
  Thesis Fig —  → Specimen-Budget Inversion             (fig_inversion.py — §3.6.5 only, no standalone)

  Conference paper mapping (do NOT renumber paper assets):
  Paper  Fig 1  → Attrition Flow          = Thesis Fig 4
  Paper  Fig 2  → Divergence Curves       = Thesis Fig 9
  Paper  Fig 3  → Hierarchical Shrinkage  = Thesis Fig 8

Usage:
    from figures.style import PALETTE, SIZES, LW, apply_style, label_panel, save_figure
    from figures.style import LEGEND_KW, MATH, STYLE_ANCHORS, MULTI_SERIES, FLOWCHART
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── 1. COLOR PALETTE ─────────────────────────────────────────────────────────
# 5 named neon-vivid colors + 2 neutrals. High chroma but not pure RGB
# saturated — readable on white, distinct in greyscale print.
PALETTE: dict[str, str] = {
    # Primary method colors
    "bootstrap": "#0091FF",   # electric blue  — Bootstrap / FDM group
    "bayesian":  "#FF4757",   # vivid coral-red — Bayesian NUTS / non-FDM group
    # Hierarchical / structural
    "hier":      "#00D2A0",   # emerald teal    — hierarchical / population level
    # Annotation / accent
    "amber":     "#FFB300",   # vivid amber     — population mean / threshold markers
    "violet":    "#A855F7",   # electric violet  — secondary accent / calibration
    # Neutrals (arrows, spines, excluded boxes)
    "dark":      "#333333",   # near-black       — spines, tick labels
    "mid":       "#777777",   # medium gray      — arrows, secondary lines
    "light":     "#CCCCCC",   # light gray       — grid, shading
    # Semantic aliases (use these in figure scripts for readability)
    "include":   "#00D2A0",   # = hier — main inclusion path in attrition flow
    "exclude":   "#FF4757",   # = bayesian — excluded / dropped boxes
    "branch":    "#0091FF",   # = bootstrap — side-use path (subsampling parent)
}

# Convenience references used across scripts
C = PALETTE  # short alias: C["bootstrap"], C["bayesian"], etc.

# ── 1b. MULTI-SERIES SEQUENCE ────────────────────────────────────────────────
# Ordered list for plots with 4–7 distinct series (CDFs, calibration curves).
# Anchored to Catalanotti 2022 Fig. 4 (tension panel) and Fig. 6 (CDF panel):
# series order black → blue → forest green → deep magenta → cyan/teal → coral.
# Use zip(MULTI_SERIES, series_labels) in figure scripts.
MULTI_SERIES: list[str] = [
    "#1A1A1A",  # near-black  — primary / unnotched baseline
    "#0091FF",  # electric blue (= bootstrap) — first variant
    "#00875A",  # forest green — second variant
    "#C2185B",  # deep magenta — third variant
    "#00ACC1",  # cyan-teal    — fourth variant
    "#FF4757",  # vivid coral  (= bayesian) — fifth variant
    "#A855F7",  # violet       (= accent)   — sixth variant if needed
]

# ── 1c. FLOWCHART COLOR FILLS ────────────────────────────────────────────────
# Cat2024 Fig. 1 convention: light-blue boxes for the main bootstrap pipeline;
# light-green boxes for all Bayesian analysis steps; red dashed border for the
# entire Bayesian block; white diamond for YES/NO decision nodes.
FLOWCHART: dict[str, str] = {
    "main_fill":    "#AED6F1",   # light blue  — bootstrap / main pipeline boxes
    "bayes_fill":   "#A9DFBF",   # light green — Bayesian analysis step boxes
    "terminal":     "#A9DFBF",   # light green — START / END ovals
    "decision":     "#FFFFFF",   # white       — YES/NO diamond interiors
    "main_border":  "#333333",   # near-black  — standard box outlines
    "bayes_border": "#FF4757",   # coral-red   — dashed outline of Bayesian block
    "label_text":   "#000000",   # black       — all in-box label text
}


# ── 2. FIGURE SIZES (inches) ──────────────────────────────────────────────────
# LNCS single-column text width ≈ 122 mm ≈ 4.8 in
# Thesis A4 single-column text width ≈ 150 mm ≈ 5.9 in
SIZES: dict[str, tuple[float, float]] = {
    "single":    (4.8, 3.2),   # LNCS single-column; also fits thesis margin
    "wide":      (6.7, 3.5),   # full-page width for 2-panel figures
    "square":    (3.5, 3.5),   # scatter plots / parity
    "tall_flow": (6.5, 7.5),   # attrition flow / tall flowcharts
    "dag":       (5.5, 4.2),   # hierarchical DAG / schematic
}

DPI_PNG: int = 300       # publication-quality raster export
DPI_SCREEN: int = 150    # quick preview


# ── 3. LINE WEIGHTS ──────────────────────────────────────────────────────────
LW: dict[str, float] = {
    "spine":      0.8,   # axis spines and tick marks
    "data":       1.8,   # plotted data lines
    "schematic":  1.4,   # schematic arrows and connector lines
    "marker_edge": 0.8,  # marker edge width
    "grid":       0.5,   # grid lines (use sparingly)
    "ref":        1.0,   # reference / calibration diagonal lines
    "threshold":  1.2,   # threshold / n* marker lines
}

MARKER_SIZE: float = 7.0       # default scatter marker size (pts)
ARROW_SHRINK: float = 8.0      # default shrinkA / shrinkB for annotate arrows
BOX_EDGE_LW: float = 0.8       # schematic box edge line weight


# ── 4. TYPOGRAPHY ─────────────────────────────────────────────────────────────
# One sans-serif family for all labels. stixsans mathtext matches it visually
# for inline equations without requiring a full LaTeX install.
FONT_FAMILY: list[str] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
# "dejavusans" is always bundled with matplotlib; renders clean sans-serif math italic.
# "stixsans" requires STIX Two Sans fonts which are NOT in the standard matplotlib
# bundle — on this machine only STIXGeneral (serif) is installed, causing $n$ to
# render in serif italic instead of sans-serif italic.
MATH_FONTSET: str = "dejavusans"

FONTSIZE: dict[str, int] = {
    "title":      10,   # subplot/figure title
    "axis_label": 9,    # x/y axis labels
    "tick":       8,    # tick labels
    "annotation": 7,    # in-figure annotations and shrinkage Δ labels
    "legend":     8,    # legend text
    "panel":      11,   # panel label (a), (b), (c) — bold
    "caption":    8,    # not rendered in matplotlib; kept for reference
}


# ── 5. PANEL LABEL SPEC ───────────────────────────────────────────────────────
# Upper-left corner of every panel, same size and weight throughout.
PANEL_LABEL_KW: dict = dict(
    fontsize=FONTSIZE["panel"],
    fontweight="bold",
    ha="left",
    va="bottom",
    transform=None,   # must be set per-axes (see label_panel())
)
PANEL_LABEL_XY: tuple[float, float] = (-0.08, 1.02)   # axes-fraction coords


# ── 6. FIGURE PADDING ─────────────────────────────────────────────────────────
TIGHT_PAD: float = 1.5      # pad argument for tight_layout
TIGHT_WSPACE: float = 0.35  # default horizontal space between subplots
TIGHT_HSPACE: float = 0.40  # default vertical space between subplots


# ── 7. APPLY STYLE ────────────────────────────────────────────────────────────

def apply_style() -> None:
    """Apply the global rcParams for the thesis figure style.

    Call once at the top of every figure script, before any plt.subplots() call.
    Handles optional scienceplots gracefully.
    """
    try:
        import scienceplots  # noqa: F401
        try:
            plt.style.use(["science", "no-latex"])
        except OSError:
            pass
    except ImportError:
        pass

    matplotlib.rcParams.update({
        # Typography
        "font.family":           "sans-serif",
        "font.sans-serif":       FONT_FAMILY,
        "mathtext.fontset":      MATH_FONTSET,
        "font.size":             FONTSIZE["tick"],
        "axes.titlesize":        FONTSIZE["title"],
        "axes.labelsize":        FONTSIZE["axis_label"],
        "xtick.labelsize":       FONTSIZE["tick"],
        "ytick.labelsize":       FONTSIZE["tick"],
        "legend.fontsize":       FONTSIZE["legend"],
        # Spines — left + bottom only (Catalanotti 2024 open-axes style)
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "xtick.top":             False,   # no ticks on top edge
        "ytick.right":           False,   # no ticks on right edge
        "axes.linewidth":        LW["spine"],
        "xtick.major.width":     LW["spine"],
        "ytick.major.width":     LW["spine"],
        "xtick.minor.width":     LW["spine"] * 0.75,
        "ytick.minor.width":     LW["spine"] * 0.75,
        "xtick.direction":       "out",
        "ytick.direction":       "out",
        # Grid — off by default; enable per-figure if needed
        "axes.grid":             False,
        "grid.linewidth":        LW["grid"],
        "grid.color":            PALETTE["light"],
        "grid.alpha":            0.7,
        # Lines and markers
        "lines.linewidth":       LW["data"],
        "lines.markersize":      MARKER_SIZE,
        "patch.linewidth":       BOX_EDGE_LW,
        # Figure
        "figure.dpi":            DPI_SCREEN,
        "savefig.dpi":           DPI_PNG,
        "savefig.bbox":          "tight",
        "figure.facecolor":      "white",
        "axes.facecolor":        "white",
    })


# ── 8. PANEL LABEL HELPER ────────────────────────────────────────────────────

def label_panel(
    ax: plt.Axes,
    label: str,
    *,
    xy: tuple[float, float] = PANEL_LABEL_XY,
) -> None:
    """Add a bold panel label (a), (b), … to the upper-left of ax.

    Args:
        ax:    Target axes.
        label: Single character, e.g. "a" or "b". Parentheses added automatically.
        xy:    Axes-fraction coordinates of the label anchor.
    """
    ax.text(
        xy[0], xy[1],
        f"({label})",
        transform=ax.transAxes,
        fontsize=FONTSIZE["panel"],
        fontweight="bold",
        ha="left",
        va="bottom",
    )


# ── 9. SAVE FIGURE HELPER ─────────────────────────────────────────────────────

def save_figure(
    fig: plt.Figure,
    out_base: Path,
    *,
    pad: float = TIGHT_PAD,
    formats: tuple[str, ...] = ("svg", "png"),
) -> None:
    """Apply tight_layout and save to svg + png (or specified formats).

    Args:
        fig:      The figure to save.
        out_base: Path without extension, e.g. Path("figures/fig5_attrition_flow").
        pad:      tight_layout pad value.
        formats:  Tuple of extension strings to write.
    """
    fig.tight_layout(pad=pad)
    for ext in formats:
        out = out_base.with_suffix(f".{ext}")
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


# ── 10. ARROW STYLE HELPERS ──────────────────────────────────────────────────
# Standardised arrowprops dicts — use these for annotate() calls so all
# schematic arrows use identical weights and head sizes.

def arrow_main(color: str = C["dark"]) -> dict:
    """Primary schematic arrow (solid, filled head)."""
    return dict(
        arrowstyle="-|>",
        color=color,
        lw=LW["schematic"],
        shrinkA=ARROW_SHRINK,
        shrinkB=ARROW_SHRINK,
    )


def arrow_secondary(color: str = C["mid"]) -> dict:
    """Secondary / side-branch schematic arrow (lighter)."""
    return dict(
        arrowstyle="-|>",
        color=color,
        lw=LW["schematic"] * 0.75,
        shrinkA=ARROW_SHRINK,
        shrinkB=ARROW_SHRINK,
    )


def arrow_shrinkage(color: str = C["mid"]) -> dict:
    """Shrinkage arrow (open head, used in hierarchical shrinkage plot)."""
    return dict(
        arrowstyle="->",
        color=color,
        lw=LW["schematic"],
        shrinkA=ARROW_SHRINK,
        shrinkB=ARROW_SHRINK,
    )


# ── 11. LEGEND PRESET ────────────────────────────────────────────────────────
# Catalanotti/Tang style: frameless, compact spacing.
# Usage: ax.legend(**LEGEND_KW) or ax.legend(handles=..., **LEGEND_KW)
LEGEND_KW: dict = dict(
    frameon=False,
    fontsize=FONTSIZE["legend"],
    labelspacing=0.3,
)


# ── 12. MATH STRING CONSTANTS ────────────────────────────────────────────────
# Canonical math-italic strings for frequently used variables.
# Always wrap variable names in $...$ so they render in math italic (stixsans).
# Usage: f"Sample size {MATH['n']}", f"FDM ({MATH['n']}=34)"
MATH: dict[str, str] = {
    "n":        r"$n$",               # sample size — math italic
    "n_star":   r"$n^*$",             # threshold sample size
    "n_approx": r"$n^* \approx$",     # with approx — for threshold annotations
    "beta_pop": r"$\beta_\mathrm{pop}$",
    "sigma_obs": r"$\sigma_\mathrm{obs}$",
    "sim":      r"$\sim$",            # distributed-as symbol
    "normal":   r"$\mathcal{N}$",
}


# ── CONVENTION NOTE ───────────────────────────────────────────────────────────
# No embedded figure captions.
# ax.set_title() and fig.suptitle() must NOT be used for "Figure X. ..." text.
# Captions belong in the manuscript, not in the saved image file.
# Only panel labels (a), (b), ... go inside the figure via label_panel().


# ── 13. STYLE ANCHORS (read PDFs in data/pdfs/key_anchors/ before editing) ───
# Documented mapping: style rule → anchor paper + figure reference.
# This dict is the persistent record of WHY each convention was chosen.
# Any future figure script should import STYLE_ANCHORS and reference it
# when deviating from a convention (justify the deviation in a comment).
STYLE_ANCHORS: dict[str, dict] = {
    "open_axes": {
        "rule":   "Top and right spines OFF; only left + bottom visible.",
        "anchor": "Catalanotti 2024 Fig. 3 — all panels use open-axis style.",
        "pdf":    "catalanotti_2024_composites-sci-tech.pdf p.6",
    },
    "sans_serif_font": {
        "rule":   "All text in sans-serif: Helvetica Neue -> Helvetica -> Arial -> DejaVu Sans.",
        "anchor": "Catalanotti 2024 throughout — body and figure labels clearly sans-serif.",
        "pdf":    "catalanotti_2024_composites-sci-tech.pdf p.1-7",
    },
    "math_italic_variables": {
        "rule":   "All scalar variable names (n, n*, beta, tau, sigma, x, y) in $...$. "
                  "mathtext.fontset='dejavusans' (always bundled; stixsans NOT installed here).",
        "anchor": "Catalanotti 2024 Fig. 2 (w, d, c_x, c_y all italic); "
                  "Tang 2025 Fig. 4d caption ('n: number of observation' italic n).",
        "pdf":    "catalanotti_2024_composites-sci-tech.pdf p.4; tang_2025_jcim.pdf p.7",
    },
    "no_embedded_caption": {
        "rule":   "NEVER put 'Figure N. ...' inside the figure. Caption = manuscript only.",
        "anchor": "Both Catalanotti 2024 and Tang 2025 — zero title text inside figure bodies.",
        "pdf":    "catalanotti_2024_composites-sci-tech.pdf; tang_2025_jcim.pdf",
    },
    "panel_labels": {
        "rule":   "Bold (a)/(b)/(c) at (-0.08, 1.02) axes-fraction, 11pt. Use label_panel().",
        "anchor": "Tang 2025 Fig. 5 — bold (a)/(b)/(c)/(d) upper-left outside each panel.",
        "pdf":    "tang_2025_jcim.pdf p.8",
    },
    "frameless_legend": {
        "rule":   "frameon=False, labelspacing=0.3. Always use LEGEND_KW.",
        "anchor": "Catalanotti 2024 — no legend box frame in any figure.",
        "pdf":    "catalanotti_2024_composites-sci-tech.pdf",
    },
    "palette_semantics": {
        "rule":   "bootstrap=#0091FF; bayesian=#FF4757; hier/inclusion=#00D2A0; "
                  "threshold/target=#FFB300; accent/hyperpriors=#A855F7. "
                  "Multi-series (≥4 curves): use MULTI_SERIES list in order. "
                  "Flowcharts: use FLOWCHART dict (light-blue main, green Bayesian, red dashed border).",
        "anchor": "Catalanotti 2024 Fig. 1: blue=main flow, green=Bayesian steps, red dashed=Bayesian block. "
                  "Catalanotti 2022 Fig. 4 & 6: ordered multi-hue (black→blue→green→magenta→cyan→coral). "
                  "Tang 2025 Fig. 5: blue=accuracy, green=rank, red=calibration.",
        "pdf":    "catalanotti_2024_composites-sci-tech.pdf p.3; "
                  "catalanotti_2022_composites-parta.pdf p.4,p.6; tang_2025_jcim.pdf p.8",
    },
    "multi_series_order": {
        "rule":   "Use MULTI_SERIES list (index 0→6) when plotting 4+ curves. "
                  "Never reuse PALETTE bootstrap/bayesian colors for unrelated series — "
                  "semantic confusion (bootstrap=blue, bayesian=red) overrides sequential aesthetics.",
        "anchor": "Catalanotti 2022 Fig. 6: 7 CDF curves use distinct sequential hues with no repeat.",
        "pdf":    "catalanotti_2022_composites-parta.pdf p.6",
    },
    "flowchart_fills": {
        "rule":   "Main pipeline boxes: FLOWCHART['main_fill'] (#AED6F1 light blue). "
                  "Bayesian steps: FLOWCHART['bayes_fill'] (#A9DFBF light green). "
                  "Bayesian block outline: FLOWCHART['bayes_border'] dashed, lw=1.2. "
                  "Decision nodes (diamonds): white fill, dark border.",
        "anchor": "Catalanotti 2024 Fig. 1 — entire methodology flowchart with Bayesian block.",
        "pdf":    "catalanotti_2024_composites-sci-tech.pdf p.3",
    },
    "caption_format": {
        "rule":   "Caption: 'Figure N. [Noun phrase]. Panel (a) ... Panel (b) ... [Takeaway].' "
                  "Never starts with a verb. Variable names italic in caption body.",
        "anchor": "Tang 2025 Fig. 4: 'Figure 4. Metrics for evaluating UQ models: (a) ...'",
        "pdf":    "tang_2025_jcim.pdf p.7",
    },
    "error_bands": {
        "rule":   "+-1 std shown as translucent fill_between band (alpha=0.13) on line plots. "
                  "Error bars on aggregated bar charts.",
        "anchor": "Tang 2025 Fig. 5 — all bars have std error bars across 5-fold CV.",
        "pdf":    "tang_2025_jcim.pdf p.8",
    },
    "flowchart_boxes": {
        "rule":   "Teal/blue fill = main inclusion chain. Coral fill = excluded/stochastic. "
                  "Amber = shared/population nodes. Violet = hyperpriors.",
        "anchor": "Catalanotti 2024 Fig. 1 — blue boxes main pipeline, red dashed Bayesian block.",
        "pdf":    "catalanotti_2024_composites-sci-tech.pdf p.3",
    },
}
