"""
DECK figure — HEADLINE specimen budget (dark/neon Allotrope).
Slide 9 (Finding 2): to reach an 88% honest-coverage target, Bayesian needs n=5
specimens vs Bootstrap's n=28 — a ~5.6x specimen saving. Rendered as a same-size
tensile-coupon (dogbone) array so the disparity is visceral to a materials panel.

Honest by construction:
  - both coupon icons are identical size (area = honest count comparison);
  - the 5.6x is read straight from results/inversion_table.json (PICP@90 target 0.88);
  - the 90% ceiling caveat (neither method reaches a full 90% on the pool) is stated.

Data: results/inversion_table.json
Run:  ~/miniconda3/bin/python figures/deck/fig_specimen_budget_deck.py
Out:  Presentations/current/figures/specimen_budget_deck.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from figures.deck_style import (  # noqa: E402
    PALETTE as C, NEUTRAL, FONTSIZE, T,
    apply_deck, save_deck,
)

DATA_PATH = ROOT / "results" / "inversion_table.json"
OUT_BASE  = ROOT / "Presentations" / "current" / "figures" / "specimen_budget_deck"

TARGET = 0.88  # honest-coverage target where the gap opens


def _n_min(block, target):
    for row in block["picp_90"]:
        if abs(row["target"] - target) < 1e-9:
            return int(row["n_min"])
    raise KeyError(target)


def _coupon(ax, cx, cy, w, h, color):
    """Draw one vertical tensile dogbone centered at (cx, cy)."""
    g = 0.46 * w   # grip half-width
    k = 0.20 * w   # gauge half-width
    t = 0.5 * h
    pts = [
        ( g,  t), ( g,  0.62 * t), ( k,  0.42 * t), ( k, -0.42 * t),
        ( g, -0.62 * t), ( g, -t),
        (-g, -t), (-g, -0.62 * t), (-k, -0.42 * t), (-k,  0.42 * t),
        (-g,  0.62 * t), (-g,  t),
    ]
    pts = [(cx + px, cy + py) for px, py in pts]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=0.28,
                         edgecolor=color, lw=1.3, joinstyle="round", zorder=3))


def _grid(ax, n, ncols, x0, y_top, cw, ch, color):
    """Lay n coupons in a top-down grid; return (x_extent, y_bottom)."""
    for i in range(n):
        r, c = divmod(i, ncols)
        cx = x0 + c * cw + cw / 2
        cy = y_top - r * ch - ch / 2
        _coupon(ax, cx, cy, cw * 0.62, ch * 0.78, color)
    nrows = int(np.ceil(n / ncols))
    return x0 + ncols * cw, y_top - nrows * ch


def main():
    d = json.loads(DATA_PATH.read_text())
    n_bayes = _n_min(d["bayesian"], TARGET)     # 5
    n_boot  = _n_min(d["bootstrap"], TARGET)    # 28
    ratio   = n_boot / n_bayes                  # 5.6

    apply_deck()
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    CW, CH = 8.0, 12.0  # identical coupon cell for both groups (honest area)
    Y_TOP = 78

    # ── Bayesian group (left): n=5 in a single column ────────────────────────
    _grid(ax, n_bayes, ncols=1, x0=10, y_top=Y_TOP, cw=CW, ch=CH, color=C["bootstrap"])
    ax.text(10 + CW / 2, Y_TOP + 5, f"Bayesian NUTS\nn = {n_bayes}",
            color=C["bootstrap"], fontsize=FONTSIZE["panel"], ha="center",
            va="bottom", fontweight="bold", linespacing=1.3)

    # ── Bootstrap group (right): n=28 in a 7-wide grid ───────────────────────
    _grid(ax, n_boot, ncols=7, x0=44, y_top=Y_TOP, cw=CW, ch=CH, color=C["bayesian"])
    ax.text(44 + 7 * CW / 2, Y_TOP + 5, f"Bootstrap\nn = {n_boot}",
            color=C["bayesian"], fontsize=FONTSIZE["panel"], ha="center",
            va="bottom", fontweight="bold", linespacing=1.3)

    # ── Center headline ──────────────────────────────────────────────────────
    # Center horizontally in the gap between the two coupon masses (Bayesian
    # column ends ~x=17, Bootstrap grid starts ~x=44 → gap center ~30.5), and
    # vertically on the middle of the coupon band.
    HX, HY = 30.5, 48
    ax.text(HX, HY, f"{ratio:.1f}×", color=T["fg"], fontsize=58, ha="center",
            va="center", fontweight="bold")
    ax.text(HX, HY - 12, "fewer specimens", color=NEUTRAL["sub"],
            fontsize=FONTSIZE["panel"], ha="center", va="center")
    ax.text(HX, HY - 19, "to reach 88% honest coverage", color=NEUTRAL["sub"],
            fontsize=FONTSIZE["annotation"], ha="center", va="center")

    # ── Honest ceiling caveat ────────────────────────────────────────────────
    ax.text(50, 6,
            "Neither method stably reaches a full 90% target on the heterogeneous pool — "
            "that ceiling is itself the data-scarcity finding.",
            color=NEUTRAL["sub"], fontsize=FONTSIZE["annotation"], ha="center", va="center")

    save_deck(fig, OUT_BASE)


if __name__ == "__main__":
    main()
