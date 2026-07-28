"""
figures/deck_assets/concept_intro.py — conceptual right-column visuals for the
text-only INTRODUCTION slides (S2 Problem, S3 Purpose, S4 Significance,
S5 Research Questions). Fixes the "empty slide" problem (BEAUTIFY-PLAN item B).

These are SCHEMATIC diagrams — no quantitative claims, no data points, so they
carry no honesty burden (unlike figures/deck/*, which are JSON-bound). They share
the deck's neon Allotrope DNA (palette, glow) so they read as one family.

Saved with a transparent background so the slide's atmosphere shows through.
Run: ~/miniconda3/bin/python figures/deck_assets/concept_intro.py
Out: Presentations/current/figures/concept_{problem,purpose,significance,rq}.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "Presentations" / "current" / "figures"

# palette (matches build_deck.py / deck_style.py)
BG = "#0F0D15"
FG = "#F3F1F8"
SUB = "#9C97B0"
VIOLET = "#A78BFA"
ORANGE = "#FB923C"
TEAL = "#2DD4BF"
SKY = "#7DD3FC"
PANEL = "#1A1724"
LINE = "#39334A"
SANS = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": SANS,
                     "mathtext.fontset": "dejavusans"})

FIGSIZE = (6.0, 5.4)   # portrait-ish, fills a right column
DPI = 200


def _fig():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _node(ax, cx, cy, w, h, label, color, *, sub=None, dashed=False,
          fs=14, label_color=None, glow=True):
    """Glowing rounded node centred at (cx, cy)."""
    if glow:
        for pad, alpha in ((3.2, 0.05), (1.6, 0.10)):
            ax.add_patch(FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h,
                boxstyle=f"round,pad={pad},rounding_size=4",
                fc=color, ec="none", alpha=alpha, zorder=2))
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.6,rounding_size=4",
        fc=PANEL, ec=color, lw=2.0,
        linestyle=(0, (4, 3)) if dashed else "solid", zorder=3))
    ty = cy + (4.5 if sub else 0)
    ax.text(cx, ty, label, color=label_color or color, fontsize=fs,
            fontweight="bold", ha="center", va="center", zorder=4,
            linespacing=1.05)
    if sub:
        ax.text(cx, cy - 5.5, sub, color=SUB, fontsize=10.5,
                ha="center", va="center", zorder=4)


def _arrow(ax, x0, y0, x1, y1, *, color=SUB, lw=2.0, glow_color=None):
    """Soft connector arrow, optional neon glow underlay."""
    if glow_color:
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
            color=glow_color, lw=lw + 3, alpha=0.10, zorder=1,
            shrinkA=6, shrinkB=6))
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
        color=color, lw=lw, zorder=3, shrinkA=6, shrinkB=6))


def _save(fig, name):
    out = OUT / f"{name}.png"
    fig.savefig(out, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Wrote: {out}")


# ── B1 · Problem: prediction → unquantified gap → decision ────────────────────
def problem():
    fig, ax = _fig()
    _node(ax, 50, 85, 56, 18, "Point prediction", VIOLET,
          sub="a single number, no error bar", label_color=FG)

    _arrow(ax, 50, 74, 50, 62, glow_color=VIOLET)

    # the gap — dashed, unknown (caption carried as the node's sub, inside)
    _node(ax, 50, 49, 56, 22, "How much do we\ntrust it?", SKY,
          sub="uncertainty unquantified", dashed=True, label_color=SKY, fs=15)

    _arrow(ax, 50, 36, 50, 25, glow_color=SKY)

    _node(ax, 50, 14, 56, 18, "Engineering decision", TEAL,
          sub="indefensible without it", label_color=FG)
    _save(fig, "concept_problem")


# ── B2 · Purpose: two UQ methods → calibrated error bars (which wins?) ─────────
def purpose():
    fig, ax = _fig()
    _node(ax, 27, 80, 40, 18, "Bootstrap", VIOLET, sub="resample the data", fs=15)
    _node(ax, 73, 80, 40, 18, "Bayesian", ORANGE, sub="prior × data", fs=15)

    _arrow(ax, 27, 70, 43, 52, glow_color=VIOLET)
    _arrow(ax, 73, 70, 57, 52, glow_color=ORANGE)

    _node(ax, 50, 42, 60, 22, "Calibrated\nerror bars", TEAL,
          sub="honest coverage, not accuracy", label_color=FG, fs=15)

    _arrow(ax, 50, 29, 50, 19, glow_color=TEAL)
    # the question that drives the study
    ax.text(50, 11, "Which wins under scarcity?", color=SKY, fontsize=14,
            fontweight="bold", ha="center", va="center")
    _save(fig, "concept_purpose")


# ── B3 · Significance: specimens → calibrated method → certification ──────────
def _coupon(ax, cx, cy, w, h, color):
    from matplotlib.patches import Polygon
    g, k, t = 0.46 * w, 0.20 * w, 0.5 * h
    pts = [(g, t), (g, 0.62 * t), (k, 0.42 * t), (k, -0.42 * t),
           (g, -0.62 * t), (g, -t), (-g, -t), (-g, -0.62 * t),
           (-k, -0.42 * t), (-k, 0.42 * t), (-g, 0.62 * t), (-g, t)]
    pts = [(cx + px, cy + py) for px, py in pts]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=0.28,
                         edgecolor=color, lw=1.4, joinstyle="round", zorder=4))


def significance():
    fig, ax = _fig()
    # a few specimens (the budget question)
    for i, dx in enumerate((-14, 0, 14)):
        _coupon(ax, 50 + dx, 82, 9, 17, VIOLET)
    ax.text(50, 96, "few specimens", color=SUB, fontsize=11, ha="center")

    _arrow(ax, 50, 70, 50, 61, glow_color=VIOLET)

    _node(ax, 50, 51, 60, 16, "Minimum n to certify", VIOLET,
          label_color=FG, fs=15)
    _arrow(ax, 50, 42, 50, 32, glow_color=TEAL)

    # the standard, with a check badge at the top-right corner (clear of text)
    _node(ax, 50, 21, 60, 18, "PNS-AM standard", TEAL, label_color=FG,
          sub="AM certification input")
    ax.add_patch(Circle((78, 29), 4.6, fc=TEAL, ec="none", alpha=0.20, zorder=5))
    ax.plot([75.9, 77.5, 80.4], [29, 26.9, 31.5], color=TEAL, lw=2.4,
            solid_capstyle="round", zorder=6)
    _save(fig, "concept_significance")


# ── B4 · Research Questions: two questions + one hypothesis, on a scarcity axis ─
def rq():
    fig, ax = _fig()
    badges = [
        ("RQ1", SKY, "At n = 26, which method\nis better-calibrated?"),
        ("RQ2", SKY, "How does the gap scale\nwith sample size?"),
        ("H", ORANGE, "Bayesian priors win\nwhen data is scarce."),
    ]
    y0, dy = 80, 27
    for i, (tag, col, text) in enumerate(badges):
        y = y0 - i * dy
        # badge
        ax.add_patch(Circle((14, y), 8.2, fc=PANEL, ec=col, lw=2.2, zorder=3))
        ax.add_patch(Circle((14, y), 8.2, fc=col, ec="none", alpha=0.10, zorder=2))
        ax.text(14, y, tag, color=col, fontsize=14 if len(tag) > 1 else 17,
                fontweight="bold", ha="center", va="center", zorder=4)
        # card
        ax.add_patch(FancyBboxPatch((28, y - 9.5), 64, 19,
                     boxstyle="round,pad=0.6,rounding_size=3",
                     fc=PANEL, ec=LINE, lw=1.4, zorder=3))
        ax.text(32, y, text, color=FG, fontsize=12, ha="left", va="center",
                zorder=4, linespacing=1.15)
    _save(fig, "concept_rq")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    problem()
    purpose()
    significance()
    rq()


if __name__ == "__main__":
    main()
