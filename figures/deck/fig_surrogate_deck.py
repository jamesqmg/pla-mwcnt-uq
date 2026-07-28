"""
DECK figure — Tier-2 physics surrogate cross-check (dark/neon Allotrope).
Slide 12 (Finding 5): the data-alone strength slope CI CROSSES ZERO, yet the
data-blind Halpin-Tsai physics check confirms the POSITIVE sign. Claim = sign +
order of magnitude only (modulus != strength; ~31x magnitude gap).

Single panel, honest by construction:
  - the n=26 data strength slope drawn as a point + 95% CI bar straddling 0;
  - a zero reference line (the "CNTs do nothing" null);
  - an annotation that the independent physics surrogate agrees on SIGN, not magnitude.

Data: results/surrogate_validation.json
Run:  ~/miniconda3/bin/python figures/deck/fig_surrogate_deck.py
Out:  Presentations/current/figures/surrogate_deck.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from figures.deck_style import (  # noqa: E402
    PALETTE as C, NEUTRAL, SIZES, FONTSIZE, T,
    apply_deck, deck_axes, save_deck,
)

DATA_PATH = ROOT / "results" / "surrogate_validation.json"
OUT_BASE  = ROOT / "Presentations" / "current" / "figures" / "surrogate_deck"


def main():
    d = json.loads(DATA_PATH.read_text())
    slope = d["data_slope_raw_mpa_per_wt"]
    lo, hi = d["data_slope_ci"]
    ratio = d["magnitude_ratio_modulus_to_strength"]

    apply_deck()
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    deck_axes(ax)

    Y_DATA, Y_PHYS = 0.0, 1.05

    # x-axis = effect of CNTs on strength. Zero = "no effect".
    # Draw the zero reference as a bounded segment (only where the data bar and
    # physics arrow live) so it doesn't run through the takeaway block below.
    ZERO_LO, ZERO_HI = -0.30, 1.58
    ax.plot([0, 0], [ZERO_LO, ZERO_HI], color=NEUTRAL["ref"], lw=1.6, ls="--",
            alpha=0.85, zorder=1)
    ax.text(0, 1.62, "no effect", color=NEUTRAL["sub"],
            fontsize=FONTSIZE["annotation"], ha="center", va="bottom")

    # DATA plausible range (violet bar) — straddles zero.
    ax.fill_betweenx([Y_DATA - 0.13, Y_DATA + 0.13], lo, hi,
                     color=C["bootstrap"], alpha=0.30, lw=0, zorder=2)
    ax.plot([lo, hi], [Y_DATA, Y_DATA], color=C["bootstrap"], lw=2.4, zorder=3)
    for xb in (lo, hi):
        ax.plot([xb, xb], [Y_DATA - 0.13, Y_DATA + 0.13],
                color=C["bootstrap"], lw=2.4, zorder=3)
    ax.plot(slope, Y_DATA, "o", color=C["bootstrap"], ms=14, zorder=4)
    ax.text((lo + hi) / 2, Y_DATA + 0.27,
            "Our 26 data points: the effect could be anywhere in this range",
            color=C["bootstrap"], fontsize=FONTSIZE["annotation"],
            ha="center", va="bottom")

    # PHYSICS direction (teal arrow) — independent, positive.
    ax.annotate("", xy=(hi * 0.9, Y_PHYS), xytext=(0.05, Y_PHYS),
                arrowprops=dict(arrowstyle="-|>", color=C["hier"], lw=2.8))
    ax.text((hi * 0.9) / 2, Y_PHYS + 0.14,
            "Halpin–Tsai surrogate (data-blind): predicts positive reinforcement",
            color=C["hier"], fontsize=FONTSIZE["annotation"], ha="center", va="bottom")

    # Plain-English takeaway — anchored to the lower-left clear zone so it sits
    # off the zero reference line (which now stops above this block).
    ax.text(min(lo, 0) - 0.45, -0.82,
            "The data alone can't rule out 'no effect' — its range crosses zero.\n"
            "The physics confirms the direction: CNTs do strengthen.\n"
            f"We claim direction only — physics predicts stiffness, not strength "
            f"(~{ratio:.0f}× larger).",
            color=T["fg"], fontsize=FONTSIZE["annotation"], ha="left", va="top",
            linespacing=1.5)

    ax.set_yticks([])
    ax.set_ylim(-1.7, 1.9)
    ax.set_xlim(min(lo, 0) - 0.5, hi + 0.5)
    ax.set_xlabel("Effect of CNTs on tensile strength  (MPa per wt%)")
    ax.spines["left"].set_visible(False)
    save_deck(fig, OUT_BASE)


if __name__ == "__main__":
    main()
