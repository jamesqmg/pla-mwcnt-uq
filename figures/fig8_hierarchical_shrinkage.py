"""
Figure 3 (conference paper) — Hierarchical shrinkage plot
Phase 2b — partial-pooling Bayesian regression with FDM vs non-FDM grouping (n=44 pool).

Shows for each group: the unpooled OLS slope vs the hierarchical posterior mean slope,
with arrows to the population mean β_pop. This visualises partial pooling — groups with
fewer observations (non-FDM, n=10) shrink more strongly toward β_pop than groups with
more data (FDM, n=34).

Data sources:
  - results/hierarchical_phase2b.json (β_pop, β_g posterior means)
  - data/extraction/master_dataset.csv (for unpooled OLS slopes)

Run:  python figures/fig3_hierarchical_shrinkage.py
Outputs:
  figures/fig3_hierarchical_shrinkage.svg
  figures/fig3_hierarchical_shrinkage.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from figures.style import (  # noqa: E402
    PALETTE as C, SIZES, LW, MARKER_SIZE, DPI_PNG,
    apply_style, label_panel, save_figure, arrow_shrinkage, LEGEND_KW,
)
from src.uq.bayesian import standardize  # noqa: E402
from src.uq.hierarchical import (         # noqa: E402
    GROUP_FDM,
    GROUP_NON_FDM,
    load_hierarchical_data,
)

# ── Parameterization ─────────────────────────────────────────────────────────
COLOR_FDM   = C["bootstrap"]   # electric blue — FDM group
COLOR_NON   = C["bayesian"]    # vivid coral   — non-FDM group
COLOR_POP   = C["amber"]       # amber          — population mean β_pop
COLOR_ARROW = C["mid"]
FIG_W, FIG_H = SIZES["single"]

HIER_JSON = ROOT / "results" / "hierarchical_phase2b.json"
DATA_CSV  = ROOT / "data" / "extraction" / "master_dataset.csv"
OUT_BASE  = ROOT / "figures" / "fig3_hierarchical_shrinkage"


def unpooled_ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Plain least-squares slope on the standardised pair (intercept allowed)."""
    x_std, y_std, _ = standardize(x, y)
    X = np.column_stack([np.ones_like(x_std), x_std])
    beta, *_ = np.linalg.lstsq(X, y_std, rcond=None)
    return float(beta[1])


def main() -> None:
    with open(HIER_JSON) as f:
        hier = json.load(f)

    beta_pop_slope = float(hier["beta_pop_posterior_mean"][1])
    beta_g_slopes  = np.array(hier["beta_g_posterior_mean"])[:, 1]
    slope_fdm_pool = float(beta_g_slopes[GROUP_FDM])
    slope_non_pool = float(beta_g_slopes[GROUP_NON_FDM])

    x, y, g = load_hierarchical_data(DATA_CSV)
    # Unpooled slopes are computed on each group's data alone — this is what
    # the hierarchy shrinks *toward β_pop* away from.
    mask_fdm = g == GROUP_FDM
    mask_non = g == GROUP_NON_FDM
    slope_fdm_unp = unpooled_ols_slope(x[mask_fdm], y[mask_fdm])
    slope_non_unp = unpooled_ols_slope(x[mask_non], y[mask_non])

    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    x_unp, x_pool = 0.0, 1.0
    ax.axhline(beta_pop_slope, color=COLOR_POP, lw=1.0, ls="--",
               label=fr"$\beta_{{\mathrm{{pop}}}}$ = {beta_pop_slope:+.3f}")

    # FDM
    ax.scatter([x_unp], [slope_fdm_unp], s=MARKER_SIZE * 10, marker="o",
               edgecolors=COLOR_FDM, facecolors="white", lw=LW["marker_edge"], zorder=4)
    ax.scatter([x_pool], [slope_fdm_pool], s=MARKER_SIZE * 10, marker="o",
               color=COLOR_FDM, lw=LW["marker_edge"], zorder=4,
               label=f"FDM ($n$={int(mask_fdm.sum())})")
    ax.annotate("", xy=(x_pool, slope_fdm_pool), xytext=(x_unp, slope_fdm_unp),
                arrowprops=arrow_shrinkage(COLOR_ARROW))

    # non-FDM
    ax.scatter([x_unp], [slope_non_unp], s=MARKER_SIZE * 10, marker="s",
               edgecolors=COLOR_NON, facecolors="white", lw=LW["marker_edge"], zorder=4)
    ax.scatter([x_pool], [slope_non_pool], s=MARKER_SIZE * 10, marker="s",
               color=COLOR_NON, lw=LW["marker_edge"], zorder=4,
               label=f"non-FDM ($n$={int(mask_non.sum())})")
    ax.annotate("", xy=(x_pool, slope_non_pool), xytext=(x_unp, slope_non_unp),
                arrowprops=arrow_shrinkage(COLOR_ARROW))

    ax.set_xticks([x_unp, x_pool])
    ax.set_xticklabels(["Unpooled\n(group OLS)", "Hierarchical\nposterior"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Slope on standardised CNT wt% (per std)")
    ax.legend(loc="lower right", **LEGEND_KW)

    # Annotate the shrinkage magnitude
    fdm_shr = abs(slope_fdm_unp - slope_fdm_pool)
    non_shr = abs(slope_non_unp - slope_non_pool)
    from figures.style import FONTSIZE  # noqa: E402
    ax.text(x_pool + 0.05, slope_fdm_pool,
            fr"shrink $\Delta$={fdm_shr:.2f}", color=COLOR_FDM,
            fontsize=FONTSIZE["annotation"], va="center")
    ax.text(x_pool + 0.05, slope_non_pool,
            fr"shrink $\Delta$={non_shr:.2f}", color=COLOR_NON,
            fontsize=FONTSIZE["annotation"], va="center")

    save_figure(fig, OUT_BASE)

    print(f"FDM      unpooled slope = {slope_fdm_unp:+.4f}  →  hierarchical = {slope_fdm_pool:+.4f}  (Δ={fdm_shr:.4f})")
    print(f"non-FDM  unpooled slope = {slope_non_unp:+.4f}  →  hierarchical = {slope_non_pool:+.4f}  (Δ={non_shr:.4f})")
    print(f"β_pop slope = {beta_pop_slope:+.4f}")


if __name__ == "__main__":
    main()
