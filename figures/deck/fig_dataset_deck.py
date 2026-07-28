"""
DECK figure — Primary dataset (n=26 PLA+FDM+D638), dark/neon Allotrope.
Slide 4 (Research question + dataset): the actual material system the thesis is
about — few, scattered points = the data-scarce regime. Data-safe (the Phase 1/2
source master_dataset_strict.csv), NOT the 200-row Hiremath pool.

Run:  ~/miniconda3/bin/python figures/deck/fig_dataset_deck.py
Out:  Presentations/current/figures/dataset_deck.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from figures.deck_style import (  # noqa: E402
    PALETTE as C, NEUTRAL, SIZES, FONTSIZE, T,
    apply_deck, deck_axes, save_deck, glow_line,
)

OUT_BASE = ROOT / "Presentations" / "current" / "figures" / "dataset_deck"


def load_xy():
    from src.uq.data_loader import load_dataset, filter_pla_fdm
    df = load_dataset(ROOT / "data" / "extraction" / "master_dataset_strict.csv")
    df = filter_pla_fdm(df)
    return (df["cnt_wt_pct"].values.astype(float),
            df["tensile_mpa"].values.astype(float))


def main():
    x, y = load_xy()
    n = len(x)
    cv = y.std() / y.mean() * 100.0

    apply_deck()
    fig, ax = plt.subplots(figsize=SIZES["single"])
    deck_axes(ax)

    # OLS trend (sky) — gentle, with the scatter dominating visually.
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    glow_line(ax, xx, m * xx + b, C["sky"], lw=2.0)

    # The 26 points (violet, glow markers).
    ax.scatter(x, y, s=80, color=C["bootstrap"], alpha=0.92,
               edgecolors=T["bg"], linewidths=0.9, zorder=4)

    ax.text(0.97, 0.06, f"$n$ = {n}\nscatter (CV) = {cv:.0f}%\nthe data-scarce regime",
            transform=ax.transAxes, ha="right", va="bottom", color=T["fg"],
            fontsize=FONTSIZE["annotation"],
            bbox=dict(boxstyle="round,pad=0.4", fc=T["panel"], ec=NEUTRAL["sub"]))

    ax.set_xlabel("CNT loading (wt%)")
    ax.set_ylabel("Tensile strength (MPa)")
    save_deck(fig, OUT_BASE)


if __name__ == "__main__":
    main()
