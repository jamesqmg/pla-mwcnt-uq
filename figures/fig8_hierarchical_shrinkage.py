"""
Figure 8 — Hierarchical partial pooling (2-panel)
Phase 2b/2c — partial-pooling Bayesian regression on the n=44 pool.

Panel (a): route-slope shrinkage. For each processing route, the unpooled OLS
slope vs the hierarchical posterior mean slope, with arrows to β_pop. Groups with
fewer observations (non-FDM, n=10) shrink more strongly toward β_pop.

Panel (b): per-paper random intercepts (Phase 2c crossed model). α_paper offset
for each of the 6 source studies; τ_α (between-study) vs τ_β (between-route)
annotated — between-study scatter dominates.

Data sources:
  - results/hierarchical_phase2b.json (β_pop, β_g posterior means)
  - results/hierarchical_phase2c_paper_intercept.json (α_paper, τ_α, τ_β)
  - data/extraction/master_dataset.csv (for unpooled OLS slopes)

Outputs: figures/fig8_hierarchical_shrinkage.svg + .png
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
    PALETTE as C, SIZES, LW, MARKER_SIZE, FONTSIZE,
    apply_style, label_panel, save_figure, arrow_shrinkage, LEGEND_KW,
)
from src.uq.bayesian import standardize  # noqa: E402
from src.uq.hierarchical import (         # noqa: E402
    GROUP_FDM,
    GROUP_NON_FDM,
    load_hierarchical_data,
)

# Groups get their own slots — NOT the bootstrap/bayesian method colors
# (style.py palette_semantics: never reuse method colors for unrelated series).
COLOR_FDM = C["sky"]      # blue    — FDM group
COLOR_NON = C["orchid"]   # orchid  — non-FDM group
COLOR_POP = C["dark"]     # neutral — population mean β_pop reference line
COLOR_ARROW = C["mid"]

HIER_JSON   = ROOT / "results" / "hierarchical_phase2b.json"
PHASE2C_JSON = ROOT / "results" / "hierarchical_phase2c_paper_intercept.json"
DATA_CSV    = ROOT / "data" / "extraction" / "master_dataset.csv"
OUT_BASE    = ROOT / "figures" / "fig8_hierarchical_shrinkage"


def unpooled_ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_std, y_std, _ = standardize(x, y)
    X = np.column_stack([np.ones_like(x_std), x_std])
    beta, *_ = np.linalg.lstsq(X, y_std, rcond=None)
    return float(beta[1])


def panel_a_slope_shrinkage(ax) -> None:
    with open(HIER_JSON) as f:
        hier = json.load(f)
    beta_pop_slope = float(hier["beta_pop_posterior_mean"][1])
    beta_g_slopes = np.array(hier["beta_g_posterior_mean"])[:, 1]
    slope_fdm_pool = float(beta_g_slopes[GROUP_FDM])
    slope_non_pool = float(beta_g_slopes[GROUP_NON_FDM])

    x, y, g = load_hierarchical_data(DATA_CSV)
    mask_fdm = g == GROUP_FDM
    mask_non = g == GROUP_NON_FDM
    slope_fdm_unp = unpooled_ols_slope(x[mask_fdm], y[mask_fdm])
    slope_non_unp = unpooled_ols_slope(x[mask_non], y[mask_non])

    x_unp, x_pool = 0.0, 1.0
    ax.axhline(beta_pop_slope, color=COLOR_POP, lw=1.0, ls="--",
               label=fr"$\beta_{{\mathrm{{pop}}}}$ = {beta_pop_slope:+.3f}")
    ax.scatter([x_unp], [slope_fdm_unp], s=MARKER_SIZE * 10, marker="o",
               edgecolors=COLOR_FDM, facecolors="white", lw=LW["marker_edge"], zorder=4)
    ax.scatter([x_pool], [slope_fdm_pool], s=MARKER_SIZE * 10, marker="o",
               color=COLOR_FDM, lw=LW["marker_edge"], zorder=4,
               label=f"FDM ($n$={int(mask_fdm.sum())})")
    ax.annotate("", xy=(x_pool, slope_fdm_pool), xytext=(x_unp, slope_fdm_unp),
                arrowprops=arrow_shrinkage(COLOR_ARROW))
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
    fdm_shr = abs(slope_fdm_unp - slope_fdm_pool)
    non_shr = abs(slope_non_unp - slope_non_pool)
    ax.text(x_pool + 0.05, slope_fdm_pool, fr"shrink $\Delta$={fdm_shr:.2f}",
            color=COLOR_FDM, fontsize=FONTSIZE["annotation"], va="center")
    ax.text(x_pool + 0.05, slope_non_pool, fr"shrink $\Delta$={non_shr:.2f}",
            color=COLOR_NON, fontsize=FONTSIZE["annotation"], va="center")


def panel_b_paper_intercepts(ax) -> None:
    d = json.loads(PHASE2C_JSON.read_text())
    ap = d["alpha_paper_posterior_mean"]
    tau_a, tau_b = d["tau_alpha_posterior_mean"], d["tau_beta_posterior_mean"]
    short = {k: k.split("_")[0].capitalize() + " " + k.split("_")[1] for k in ap}
    items = sorted(ap.items(), key=lambda kv: kv[1])
    labels = [short[k] for k, _ in items]
    vals = [v for _, v in items]
    yy = np.arange(len(vals))
    ax.axvline(0, color=C["dark"], lw=1.0, ls="--", alpha=.6)
    for yi, v in zip(yy, vals):
        col = COLOR_FDM if v >= 0 else COLOR_NON
        ax.plot([0, v], [yi, yi], color=col, lw=2.0, alpha=.55, solid_capstyle="round")
        ax.plot(v, yi, "o", color=col, ms=7, zorder=5)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\alpha_\mathrm{paper}$ offset (std units)")
    ax.set_xlim(min(vals) - 0.25, max(vals) + 0.25)
    ax.text(0.97, 0.05,
            rf"$\tau_\alpha$ = {tau_a:.2f} (study)" + "\n" + rf"$\tau_\beta$ = {tau_b:.2f} (route)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONTSIZE["annotation"],
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C["light"]))


def main() -> None:
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["wide"])
    panel_a_slope_shrinkage(ax1)
    panel_b_paper_intercepts(ax2)
    label_panel(ax1, "a")
    label_panel(ax2, "b")
    save_figure(fig, OUT_BASE)
    print("wrote 2-panel fig8 (slope shrinkage + per-paper intercepts)")


if __name__ == "__main__":
    main()
