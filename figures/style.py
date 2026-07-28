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
    # Primary method colors — ALLOTROPE-LIGHT CANON (locked 2026-06-10).
    # Violet hero + warm-orange complement: a split-complementary, colour-blind-
    # safe pair shared with iaapt_viz so thesis + quant + lab figures read as one
    # visual family. Hues tuned for legibility on white at print DPI.
    "bootstrap": "#8B5CF6",   # violet  — Bootstrap / FDM group (HERO)
    "bayesian":  "#F97316",   # orange  — Bayesian NUTS / non-FDM group (warm complement)
    # Hierarchical / structural
    "hier":      "#2DD4BF",   # teal    — hierarchical / population level
    # Annotation / accent
    "sky":       "#38BDF8",   # sky     — secondary series / benchmark
    "orchid":    "#E879F9",   # orchid  — threshold / n* / target markers
    "amber":     "#C9A227",   # gold (still reads "amber") — population mean β_pop
    "violet":    "#A78BFA",   # soft violet — tertiary accent / calibration
    # Neutrals (arrows, spines, excluded boxes)
    "dark":      "#2A2433",   # warm near-black  — spines, tick labels, text
    "mid":       "#777777",   # medium gray      — arrows, secondary lines
    "light":     "#CCCCCC",   # light gray       — grid, shading
    # Semantic aliases (use these in figure scripts for readability)
    "include":   "#2DD4BF",   # = hier — main inclusion path in attrition flow
    "exclude":   "#F97316",   # = bayesian — excluded / dropped boxes
    "branch":    "#8B5CF6",   # = bootstrap — side-use path (subsampling parent)
    # ── Petrol & Gold register — FE-RVE / INCON figures (locked 2026-07-15) ──
    # Discreet journal palette for the INCON FE-RVE figures ONLY. Kept SEPARATE
    # from the bootstrap/bayesian semantics above (which stay violet/orange for
    # the main-thesis figures) so nothing here cascades to them. FE-RVE scripts
    # opt in explicitly via these names + apply_style(serif=True). See
    # Thesis2/fe-rve-viz-strategy.md §1. cividis is the field/ordered-data cmap.
    "petrol":       "#1E6E86",  # deep teal-blue — primary FE-RVE series (measured/axial)
    "gold":         "#C0912C",  # antique gold   — second striking series / analytical
    "purple_bayes": "#4A3A9E",  # deep violet    — RESERVED: Bayesian/MCMC semantic only
    "charcoal":     "#23242B",  # near-black neutral — text / reference lines
    "slate":        "#5B6672",  # muted blue-gray — sober 3rd/4th series in multi-line panels
    "rust":         "#9E4A2C",  # muted terracotta — sober series color (NOT the girlish set)
}

# Convenience references used across scripts
C = PALETTE  # short alias: C["bootstrap"], C["bayesian"], etc.

# Ordered SOBER series for multi-line FE-RVE panels (Petrol & Gold register).
# Differentiate series by COLOR, not by marker/linestyle (Jimbo, 2026-07-15) — the
# anchor papers pack multiple comparisons per panel using colored lines. Use in
# order; all print-safe and distinct from the retired "girlish" MULTI_SERIES set.
FE_RVE_SERIES: list[str] = [
    PALETTE["petrol"], PALETTE["gold"], PALETTE["rust"], PALETTE["slate"], PALETTE["charcoal"],
]

# ── 1b. MULTI-SERIES SEQUENCE ────────────────────────────────────────────────
# Ordered list for plots with 4–7 distinct series (CDFs, calibration curves).
# Anchored to Catalanotti 2022 Fig. 4 (tension panel) and Fig. 6 (CDF panel):
# series order black → blue → forest green → deep magenta → cyan/teal → coral.
# Use zip(MULTI_SERIES, series_labels) in figure scripts.
MULTI_SERIES: list[str] = [
    "#1A1A1A",  # near-black — baseline
    "#8B5CF6",  # violet     — series 2 (= bootstrap hero)
    "#38BDF8",  # sky        — series 3
    "#2DD4BF",  # teal       — series 4
    "#F97316",  # orange     — series 5 (warm pop)
    "#E879F9",  # orchid     — series 6
    "#6366F1",  # indigo     — series 7 (CEILING — see note below)
]
# >7-CATEGORY ESCAPE HATCH (Results/Discussion future-proofing):
# saturated-distinct tops out at 7. Beyond that, hues collide and stop being
# colour-blind / greyscale separable. For a figure with >7 categories do NOT
# extend this list — switch that figure to SEQ_CMAP (ordered/sequential) and
# lean on marker/linestyle redundancy, or facet into small multiples.
SEQ_CMAP: str = "viridis"

# ── 1c. FLOWCHART COLOR FILLS ────────────────────────────────────────────────
# Cat2024 Fig. 1 convention: light-blue boxes for the main bootstrap pipeline;
# light-green boxes for all Bayesian analysis steps; red dashed border for the
# entire Bayesian block; white diamond for YES/NO decision nodes.
FLOWCHART: dict[str, str] = {
    "main_fill":    "#E4DAF8",   # light violet — bootstrap / main pipeline boxes
    "bayes_fill":   "#FBE2CE",   # light orange — Bayesian analysis step boxes
    "terminal":     "#D6F7F1",   # light teal   — START / END ovals
    "decision":     "#FFFFFF",   # white        — YES/NO diamond interiors
    "main_border":  "#2A2433",   # warm near-black — standard box outlines
    "bayes_border": "#F97316",   # orange       — dashed outline of Bayesian block
    "label_text":   "#000000",   # black        — all in-box label text
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

# FE-RVE / INCON register (serif). Opt-in via apply_style(serif=True) — NOT the
# default, because the main-thesis figures follow a sans-serif anchor
# (Catalanotti 2024). Source Serif (the strategy-doc choice) is NOT installed on
# this machine; STIX Two Text is the doc's named fallback (a refined journal
# serif) and STIX Two Math pairs with it for inline math. See
# fe-rve-viz-strategy.md §1.
FONT_FAMILY_SERIF: list[str] = [
    "STIX Two Text", "Source Serif 4", "Source Serif Pro", "Charter", "DejaVu Serif",
]
MATH_FONTSET_SERIF: str = "stix"

FONTSIZE: dict[str, int] = {
    "title":      11,   # subplot/figure title
    "axis_label": 11,   # x/y axis labels   (bumped 2026-07-16 — fonts read small vs figure)
    "tick":       10,   # tick labels        (bumped)
    "annotation": 8,    # in-figure annotations and shrinkage Δ labels (bumped)
    "legend":     9,    # legend text        (bumped)
    "panel":      12,   # panel label (a), (b), (c) — bold (bumped)
    "caption":    9,    # not rendered in matplotlib; kept for reference
}

# ── Per-figure font scaling ───────────────────────────────────────────────────
# Set by apply_style(axis_scale=, inside_scale=); read by fs() and label_panel.
# axis_scale  → axis labels, ticks, colorbar labels ("outside" / X-Y).
# inside_scale → titles, panel labels, annotations, legends ("inside").
_AXIS_SCALE: float = 1.0
_INSIDE_SCALE: float = 1.0


def fs(key: str, kind: str = "inside") -> int:
    """FONTSIZE[key] scaled by the current apply_style scale.

    kind='axis' for axis/tick/colorbar labels; 'inside' for titles, panel
    labels, annotations, and legends.
    """
    scale = _AXIS_SCALE if kind == "axis" else _INSIDE_SCALE
    return max(1, round(FONTSIZE[key] * scale))


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

def apply_style(*, serif: bool = False, axis_scale: float = 1.0,
                inside_scale: float = 1.0) -> None:
    """Apply the global rcParams for the thesis figure style.

    Call once at the top of every figure script, before any plt.subplots() call.
    Handles optional scienceplots gracefully.

    Args:
        serif: Opt into the FE-RVE / INCON serif register (STIX Two Text + stix
            mathtext). Default False keeps the main-thesis sans-serif anchor
            (Catalanotti 2024) — so existing callers are unaffected.
    """
    try:
        import scienceplots  # noqa: F401
        try:
            plt.style.use(["science", "no-latex"])
        except OSError:
            pass
    except ImportError:
        pass

    family = "serif" if serif else "sans-serif"
    math_fontset = MATH_FONTSET_SERIF if serif else MATH_FONTSET
    global _AXIS_SCALE, _INSIDE_SCALE
    _AXIS_SCALE, _INSIDE_SCALE = axis_scale, inside_scale

    matplotlib.rcParams.update({
        # Typography
        "font.family":           family,
        "font.sans-serif":       FONT_FAMILY,
        "font.serif":            FONT_FAMILY_SERIF,
        "mathtext.fontset":      math_fontset,
        "font.size":             round(FONTSIZE["tick"] * axis_scale),
        "axes.titlesize":        round(FONTSIZE["title"] * inside_scale),
        "axes.labelsize":        round(FONTSIZE["axis_label"] * axis_scale),
        "xtick.labelsize":       round(FONTSIZE["tick"] * axis_scale),
        "ytick.labelsize":       round(FONTSIZE["tick"] * axis_scale),
        "legend.fontsize":       round(FONTSIZE["legend"] * inside_scale),
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
        fontsize=fs("panel"),
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
    "fe_field_and_mc": {
        "rule":   "FE scalar fields: triangulated domain, BoundaryNorm-banded "
                  "sequential colormap (FIELD_CMAP), thin white mesh overlay (MESH_EDGE), "
                  "dark inclusions with white edge, labelled colorbar (add_field_colorbar), "
                  "Fixed-end hatch + tensile-load arrows. Monte-Carlo ensembles: translucent "
                  "'spaghetti' paths (MC_PATHS) + bold median (MC_MEDIAN) + percentile band (BAND_KW).",
        "anchor": "Chahar 2026 Fig. 5 (matrix-damage contours: meshed domain, discrete banded "
                  "colorbar) and Fig. 4B (Fixed-end / Tensile-load BC schematic).",
        "pdf":    "chahar_2026_composite-structures.pdf p.8-9",
    },
}


# ── 14. SIMULATION / FIELD AESTHETIC (Phase-4 validation figures) ─────────────
# FE field contours follow Chahar 2026 (Composite Structures): a triangulated
# domain, a discrete-banded sequential colormap with a labelled colorbar, a thin
# mesh overlay, and Fixed-end/Tensile-load boundary markers. Monte-Carlo
# ensembles render as translucent "spaghetti" paths beneath a bold median and a
# percentile band. Keep an "illustrative" note on synthetic/2D fields — the full
# 3D periodic homogenisation is Thesis-2 work.
FIELD_CMAP: str = "inferno"        # perceptually-uniform scalar field (von Mises)
FIELD_LEVELS: int = 12             # discrete colorbar/contour bands (Chahar-style)
MESH_EDGE: dict = dict(color="#FFFFFF", linewidth=0.12, alpha=0.45)
INCLUSION_FILL: str = "#10131A"    # stiff inclusion interior (near-black)
INCLUSION_EDGE: dict = dict(edgecolor="#FFFFFF", linewidth=0.6)

# Monte-Carlo ensemble styling
MC_PATHS: dict = dict(color=PALETTE["bootstrap"], linewidth=0.4, alpha=0.05)
MC_MEDIAN: dict = dict(color="#1A1A1A", linewidth=2.0)
MC_BAND: dict = dict(color=PALETTE["hier"], alpha=0.30, linewidth=0)
MC_N_PATHS: int = 400              # ensemble size to draw


def add_field_colorbar(fig, mappable, ax, label: str):
    """Chahar-style labelled colorbar for an FE field panel."""
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=FONTSIZE["axis_label"])
    cb.ax.tick_params(labelsize=FONTSIZE["tick"])
    cb.outline.set_linewidth(LW["spine"])
    return cb


# ── 15. ALLOTROPE-LIGHT TREATMENT HELPERS (canon 2026-06-10) ──────────────────
# Ported from iaapt_viz so thesis figures share the IAAPT visual DNA on the
# white/print profile. THREE tools, with a rule for each:
#   glow_line  — soft neon halo under a data line. Use on the 1–2 primary method
#                lines per panel (Bootstrap / Bayesian). Keep subtle.
#   band       — flat ±1σ / PI uncertainty band with a faint edge. THE default for
#                uncertainty regions: crisp edge so the reader can read the spread.
#   gradient_fill — vertical alpha gradient from a curve to a baseline. RESERVE for
#                genuinely cumulative / equity-like curves where area-to-baseline is
#                meaningful. Do NOT use it for ±σ bands (the edge goes fuzzy).

def glow_line(ax, x, y, color, *, lw: float = 1.9, zorder: int = 3,
              glow=((6.5, 0.05), (4.2, 0.09), (2.6, 0.15))):
    """Plot a line with a soft glow (stacked translucent strokes). Subtle on white."""
    for gw, ga in glow:
        ax.plot(x, y, color=color, lw=gw, alpha=ga, zorder=zorder,
                solid_capstyle="round")
    return ax.plot(x, y, color=color, lw=lw, zorder=zorder + 1,
                   solid_capstyle="round")


def band(ax, x, y, std, color, *, alpha: float = 0.15, edge_alpha: float = 0.45,
         zorder: int = 2):
    """Flat ±1σ uncertainty band with a faint edge — the canonical uncertainty fill.

    Crisp edges (unlike a gradient) so the reader can read the spread magnitude.
    """
    import numpy as np
    y = np.asarray(y, float); std = np.asarray(std, float)
    ax.fill_between(x, y - std, y + std, color=color, alpha=alpha, lw=0, zorder=zorder)
    ax.plot(x, y - std, color=color, lw=0.6, alpha=edge_alpha, zorder=zorder)
    ax.plot(x, y + std, color=color, lw=0.6, alpha=edge_alpha, zorder=zorder)


def gradient_fill(ax, x, y, color, *, baseline: float = 0.0,
                  alpha_peak: float = 0.30, direction: str = "auto",
                  zorder: int = 2):
    """Vertical alpha-gradient fill from a curve to a baseline (equity-curve look).

    RESERVED for cumulative / monotone-ish curves. Strongest next to the line,
    fading to transparent toward the baseline. No-op on empty / all-NaN input.
    """
    import numpy as np
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon as _Polygon

    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or not np.any(np.isfinite(y)):
        return None
    if direction == "auto":
        direction = "down" if np.nanmean(y) < baseline else "up"
    rgb = mcolors.to_rgb(color)
    grad = np.empty((256, 1, 4)); grad[:, :, :3] = rgb
    ramp = np.linspace(0.0, alpha_peak, 256)
    grad[:, :, 3] = (ramp if direction == "up" else ramp[::-1])[:, None]
    ymin = min(baseline, float(np.nanmin(y))); ymax = max(baseline, float(np.nanmax(y)))
    if ymax == ymin:
        return None
    im = ax.imshow(grad, aspect="auto", origin="lower",
                   extent=[float(x.min()), float(x.max()), ymin, ymax], zorder=zorder)
    verts = np.column_stack([np.concatenate([x, x[::-1]]),
                             np.concatenate([y, np.full_like(x, baseline)[::-1]])])
    clip = _Polygon(verts, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(clip); im.set_clip_path(clip)
    return im
