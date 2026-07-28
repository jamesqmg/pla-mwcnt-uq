"""
figures/deck_style.py — Thesis 1 DEFENSE-DECK figure style (dark / neon Allotrope)
==================================================================================
The presentation profile. Distinct from `style.py` (white publication) and from
IAAPT `iaapt_viz.py` (quant/social) — but it shares the SAME Allotrope DNA so the
deck reads as one visual family with the manuscript and the quant work.

Why a third profile:
  - `style.py`  = white bg, print DPI, no in-figure titles, LNCS column widths.
  - deck        = BLACK bg + NEON glow, slide aspect ratios, larger type for a
                  projected/Zoom-shared 20-min defense where colours must POP.
  - Method-colour SEMANTICS are inherited verbatim from `style.py` so a viewer who
    saw the manuscript reads the same hues here: violet = Bootstrap, orange =
    Bayesian, teal = hierarchical, etc.

Background = Allotrope violet-graphite #0F0D15 (carbon → graphite), the same dark
ground as the IAAPT social profile.

Usage:
    from figures.deck_style import (
        PALETTE, NEUTRAL, SIZES, T, apply_deck, deck_axes, save_deck,
        glow_line, band, LEGEND_KW,
    )
    apply_deck()
    fig, ax = plt.subplots(figsize=SIZES["wide2"])
    deck_axes(ax)
    glow_line(ax, x, y, PALETTE["bootstrap"], label="Bootstrap")
    save_deck(fig, OUT / "divergence")        # writes divergence.png on dark canvas
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as _fm


# ── 0. FONT (Inter if present, else Helvetica) ───────────────────────────────
def _register_inter() -> bool:
    fd = os.path.expanduser("~/Library/Fonts")
    for ttf in glob.glob(f"{fd}/Inter-*.ttf"):
        try:
            _fm.fontManager.addfont(ttf)
        except Exception:
            continue
    return "Inter" in {f.name for f in _fm.fontManager.ttflist}


_HAVE_INTER = _register_inter()
FONT_SANS: list[str] = (["Inter"] if _HAVE_INTER else []) + [
    "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
MATH_FONTSET = "dejavusans"   # always bundled; clean sans-serif math italic


# ── 1. THEME (dark only — the deck ground) ───────────────────────────────────
T: dict[str, str] = {
    "bg":    "#0F0D15",   # figure facecolor — Allotrope violet-graphite
    "panel": "#0F0D15",   # axes facecolor — same as bg (full-bleed slide look)
    "fg":    "#F3F1F8",   # primary text (titles, key numbers)
    "sub":   "#9B93AD",   # secondary text (axis labels, ticks)
    "grid":  "#2A2536",   # spines + faint gridlines
}

# ── 2. METHOD PALETTE (semantics inherited from style.py) ────────────────────
# Same hues as the manuscript so colour meaning carries over. On #0F0D15 these
# read bright; glow_line supplies the neon halo that makes them pop.
PALETTE: dict[str, str] = {
    "bootstrap": "#A78BFA",   # violet  — Bootstrap (HERO) — lifted vs print #8B5CF6 for dark legibility
    "bayesian":  "#FB923C",   # orange  — Bayesian NUTS — lifted vs print #F97316
    "hier":      "#2DD4BF",   # teal    — hierarchical / population
    "sky":       "#38BDF8",   # sky     — secondary series / benchmark
    "orchid":    "#E879F9",   # orchid  — threshold / n* / target
    "amber":     "#FCD34D",   # amber   — population mean β_pop
}
C = PALETTE

# Neutrals for reference lines / annotations on dark (NEVER use style.py's dark ink).
NEUTRAL: dict[str, str] = {
    "ref":   "#6B6478",   # nominal/reference dashed lines
    "thresh": "#9B93AD",  # n* threshold marker
    "text":  T["fg"],
    "sub":   T["sub"],
}

# ── 3. SLIDE SIZES (16:9-friendly, inches) ───────────────────────────────────
SIZES: dict[str, tuple[float, float]] = {
    "wide2":   (11.0, 4.7),   # full-width 2-panel figure on a slide
    "single":  (7.6, 4.6),    # single-panel, ~half slide
    "square":  (5.4, 5.4),    # parity / scatter
    "wide1":   (10.5, 4.3),   # single wide strip
}
DPI: int = 200   # crisp on projector + Zoom share, compresses well

# ── 4. TYPE + LINE WEIGHTS (bumped for projection legibility) ────────────────
FONTSIZE: dict[str, int] = {
    "title": 15, "axis_label": 13, "tick": 11,
    "annotation": 11, "legend": 11, "panel": 14,
}
LW: dict[str, float] = {
    "spine": 1.0, "data": 2.4, "ref": 1.2, "threshold": 1.4, "grid": 0.6,
}
MARKER_SIZE: float = 7.5

LEGEND_KW: dict = dict(frameon=False, fontsize=FONTSIZE["legend"], labelspacing=0.35)


# ── 5. APPLY (rcParams) ──────────────────────────────────────────────────────
def apply_deck() -> dict[str, str]:
    """Set global rcParams for the dark/neon deck profile. Returns theme dict T."""
    matplotlib.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   FONT_SANS,
        "mathtext.fontset":  MATH_FONTSET,
        "font.size":         FONTSIZE["tick"],
        "axes.titlesize":    FONTSIZE["title"],
        "axes.labelsize":    FONTSIZE["axis_label"],
        "xtick.labelsize":   FONTSIZE["tick"],
        "ytick.labelsize":   FONTSIZE["tick"],
        "legend.fontsize":   FONTSIZE["legend"],
        # open axes (shared DNA)
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.top":         False,
        "ytick.right":       False,
        "axes.linewidth":    LW["spine"],
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.width": LW["spine"],
        "ytick.major.width": LW["spine"],
        # dark theme colours
        "figure.facecolor":  T["bg"],
        "axes.facecolor":    T["panel"],
        "savefig.facecolor": T["bg"],
        "text.color":        T["fg"],
        "axes.labelcolor":   T["sub"],
        "axes.edgecolor":    T["grid"],
        "xtick.color":       T["sub"],
        "ytick.color":       T["sub"],
        "grid.color":        T["grid"],
        "grid.linewidth":    LW["grid"],
        "axes.grid":         False,
        "lines.linewidth":   LW["data"],
        "figure.dpi":        120,
        "savefig.dpi":       DPI,
        "savefig.bbox":      "tight",
    })
    return T


# ── 6. PER-AXES THEME ────────────────────────────────────────────────────────
def deck_axes(ax: plt.Axes, *, grid: bool = False) -> None:
    """Apply dark facecolor, spine/tick colours, optional faint y-grid."""
    ax.set_facecolor(T["panel"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["grid"])
        ax.spines[side].set_linewidth(LW["spine"])
    ax.tick_params(colors=T["sub"], labelsize=FONTSIZE["tick"], length=3)
    if grid:
        ax.grid(True, axis="y", color=T["grid"], linewidth=LW["grid"], alpha=0.8)
        ax.set_axisbelow(True)


def label_panel(ax: plt.Axes, label: str, *, xy=(-0.09, 1.03)) -> None:
    ax.text(xy[0], xy[1], f"({label})", transform=ax.transAxes,
            fontsize=FONTSIZE["panel"], fontweight="bold", color=T["fg"],
            ha="left", va="bottom")


# ── 7. NEON TREATMENTS (stronger than the white profile) ─────────────────────
def glow_line(ax, x, y, color, *, lw: float = None, zorder: int = 3,
              glow=((9.0, 0.06), (5.5, 0.10), (3.0, 0.18)), **plot_kw):
    """Neon line: stacked translucent halo strokes under a crisp core line.

    Glow is tuned brighter than style.py's print version (dark bg eats halo).
    Extra kwargs (label, ls, marker, ms) pass to the core ax.plot.
    """
    lw = lw if lw is not None else LW["data"]
    for gw, ga in glow:
        ax.plot(x, y, color=color, lw=gw, alpha=ga, zorder=zorder,
                solid_capstyle="round")
    return ax.plot(x, y, color=color, lw=lw, zorder=zorder + 1,
                   solid_capstyle="round", **plot_kw)


def band(ax, x, lo, hi, color, *, alpha: float = 0.16, edge_alpha: float = 0.5,
         zorder: int = 2):
    """Translucent uncertainty band with a faint neon edge (crisp, readable spread)."""
    x = np.asarray(x, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=zorder)
    ax.plot(x, lo, color=color, lw=0.8, alpha=edge_alpha, zorder=zorder)
    ax.plot(x, hi, color=color, lw=0.8, alpha=edge_alpha, zorder=zorder)


# ── 8. SAVE (always paints dark canvas) ──────────────────────────────────────
def save_deck(fig: plt.Figure, out_base: Path | str, *,
              formats: tuple[str, ...] = ("png",), pad: float = 0.8) -> list[Path]:
    """tight_layout + save on the dark canvas (so it never lands on white)."""
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad)
    written: list[Path] = []
    for ext in formats:
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, dpi=DPI, facecolor=T["bg"])
        written.append(out)
        print(f"Wrote: {out}")
    plt.close(fig)
    return written
