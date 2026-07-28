"""
DECK figure — Hierarchical shrinkage + per-paper intercepts (dark/neon Allotrope).
Slide 11 (Finding 4): route-slope shrinkage toward β_pop (a) + per-paper random
intercepts with τ_α > τ_β (b). Content == manuscript Fig 8.

Data: results/hierarchical_phase2b.json, results/hierarchical_phase2c_paper_intercept.json,
      data/extraction/master_dataset.csv
Run:  ~/miniconda3/bin/python figures/deck/fig_shrinkage_deck.py
Out:  Presentations/current/figures/shrinkage_deck.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from figures.deck_style import (  # noqa: E402
    PALETTE as C, NEUTRAL, SIZES, FONTSIZE, T,
    apply_deck, deck_axes, label_panel, save_deck, LEGEND_KW,
)
from src.uq.bayesian import standardize  # noqa: E402
from src.uq.hierarchical import GROUP_FDM, GROUP_NON_FDM, load_hierarchical_data  # noqa: E402

COLOR_FDM, COLOR_NON, COLOR_POP = C["bootstrap"], C["bayesian"], C["amber"]
HIER_JSON    = ROOT / "results" / "hierarchical_phase2b.json"
PHASE2C_JSON = ROOT / "results" / "hierarchical_phase2c_paper_intercept.json"
DATA_CSV     = ROOT / "data" / "extraction" / "master_dataset.csv"
OUT_BASE     = ROOT / "Presentations" / "current" / "figures" / "shrinkage_deck"


def unpooled_ols_slope(x, y):
    x_std, y_std, _ = standardize(x, y)
    X = np.column_stack([np.ones_like(x_std), x_std])
    beta, *_ = np.linalg.lstsq(X, y_std, rcond=None)
    return float(beta[1])


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=NEUTRAL["sub"], lw=1.6,
                                shrinkA=9, shrinkB=9))


def panel_a(ax):
    hier = json.loads(HIER_JSON.read_text())
    beta_pop = float(hier["beta_pop_posterior_mean"][1])
    bg = np.array(hier["beta_g_posterior_mean"])[:, 1]
    s_fdm_pool, s_non_pool = float(bg[GROUP_FDM]), float(bg[GROUP_NON_FDM])

    x, y, g = load_hierarchical_data(DATA_CSV)
    mf, mn = g == GROUP_FDM, g == GROUP_NON_FDM
    s_fdm_unp, s_non_unp = unpooled_ols_slope(x[mf], y[mf]), unpooled_ols_slope(x[mn], y[mn])

    xu, xp = 0.0, 1.0
    ax.axhline(beta_pop, color=COLOR_POP, lw=1.4, ls="--",
               label=fr"$\beta_{{\mathrm{{pop}}}}$ = {beta_pop:+.3f}")
    # FDM (violet)
    ax.scatter([xu], [s_fdm_unp], s=130, marker="o", edgecolors=COLOR_FDM,
               facecolors=T["bg"], lw=1.8, zorder=4)
    ax.scatter([xp], [s_fdm_pool], s=130, marker="o", color=COLOR_FDM, zorder=4,
               label=f"FDM ($n$={int(mf.sum())})")
    _arrow(ax, xu, s_fdm_unp, xp, s_fdm_pool)
    # non-FDM (orange)
    ax.scatter([xu], [s_non_unp], s=130, marker="s", edgecolors=COLOR_NON,
               facecolors=T["bg"], lw=1.8, zorder=4)
    ax.scatter([xp], [s_non_pool], s=130, marker="s", color=COLOR_NON, zorder=4,
               label=f"non-FDM ($n$={int(mn.sum())})")
    _arrow(ax, xu, s_non_unp, xp, s_non_pool)

    ax.set_xticks([xu, xp])
    ax.set_xticklabels(["Unpooled\n(group OLS)", "Hierarchical\nposterior"])
    ax.set_xlim(-0.4, 1.5)
    ax.set_ylabel("Slope on standardised CNT wt%")
    ax.legend(loc="lower right", **LEGEND_KW)
    ax.text(xp + 0.06, s_fdm_pool, fr"$\Delta$={abs(s_fdm_unp - s_fdm_pool):.2f}",
            color=COLOR_FDM, fontsize=FONTSIZE["annotation"], va="center")
    ax.text(xp + 0.06, s_non_pool, fr"$\Delta$={abs(s_non_unp - s_non_pool):.2f}",
            color=COLOR_NON, fontsize=FONTSIZE["annotation"], va="center")


def panel_b(ax):
    d = json.loads(PHASE2C_JSON.read_text())
    ap = d["alpha_paper_posterior_mean"]
    tau_a, tau_b = d["tau_alpha_posterior_mean"], d["tau_beta_posterior_mean"]
    short = {k: k.split("_")[0].capitalize() + " " + k.split("_")[1] for k in ap}
    items = sorted(ap.items(), key=lambda kv: kv[1])
    labels = [short[k] for k, _ in items]
    vals = [v for _, v in items]
    yy = np.arange(len(vals))
    ax.axvline(0, color=NEUTRAL["ref"], lw=1.2, ls="--", alpha=0.7)
    for yi, v in zip(yy, vals):
        col = COLOR_FDM if v >= 0 else COLOR_NON
        ax.plot([0, v], [yi, yi], color=col, lw=3.0, alpha=0.6, solid_capstyle="round")
        ax.plot(v, yi, "o", color=col, ms=9, zorder=5)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\alpha_\mathrm{paper}$ offset (std units)")
    ax.set_xlim(min(vals) - 0.3, max(vals) + 0.3)
    ax.text(0.97, 0.06,
            rf"$\tau_\alpha$ = {tau_a:.2f} (study)" + "\n" + rf"$\tau_\beta$ = {tau_b:.2f} (route)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONTSIZE["annotation"], color=T["fg"],
            bbox=dict(boxstyle="round,pad=0.4", fc=T["panel"], ec=NEUTRAL["sub"]))


def main():
    apply_deck()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["wide2"])
    deck_axes(ax1); deck_axes(ax2)
    panel_a(ax1); panel_b(ax2)
    label_panel(ax1, "a"); label_panel(ax2, "b")
    save_deck(fig, OUT_BASE)


if __name__ == "__main__":
    main()
