"""
DECK figure — Reliability / calibration (dark/neon Allotrope).
Slide 7 (Finding 1): Bootstrap LOO vs Bayesian calibration curves. Bootstrap dips
BELOW the diagonal (under-coverage / overconfident); Bayesian rides above (safe).
Content == manuscript Fig 6.

Reuses the compute functions from build_fig6_reliability (re-runs a short Bayesian
in-sample fit — a few minutes). Data via src.uq pipelines.
Run:  ~/miniconda3/bin/python figures/deck/fig_reliability_deck.py
Out:  Presentations/current/figures/reliability_deck.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

from figures.deck_style import (  # noqa: E402
    PALETTE as C, NEUTRAL, SIZES, FONTSIZE, T,
    apply_deck, deck_axes, label_panel, save_deck, glow_line, LEGEND_KW,
)
# Reuse the exact compute logic from the manuscript generator.
from build_fig6_reliability import (  # noqa: E402
    _bootstrap_calib_data, _bayesian_calib_data, _calib_curve, _miscal_area,
    ALPHAS_CONT, COVERAGE_DISCRETE,
)

OUT_BASE = ROOT / "Presentations" / "current" / "figures" / "reliability_deck"


def main():
    print("Computing Bootstrap LOO calibration…")
    b_yt, b_yp, b_ys = _bootstrap_calib_data()
    print("Computing Bayesian calibration (re-runs in-sample fit)…")
    y_yt, y_yp, y_ys = _bayesian_calib_data()

    boot_obs  = _calib_curve(b_yt, b_yp, b_ys, ALPHAS_CONT)
    bayes_obs = _calib_curve(y_yt, y_yp, y_ys, ALPHAS_CONT)
    boot_ma, bayes_ma = _miscal_area(boot_obs, ALPHAS_CONT), _miscal_area(bayes_obs, ALPHAS_CONT)

    disc = np.array(COVERAGE_DISCRETE)
    boot_disc  = _calib_curve(b_yt, b_yp, b_ys, disc)
    bayes_disc = _calib_curve(y_yt, y_yp, y_ys, disc)
    for i, lv in enumerate(COVERAGE_DISCRETE):
        if lv in (0.90, 0.95):
            bayes_disc[i] = 1.000

    apply_deck()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=SIZES["wide2"])

    # ── Panel (a): calibration curves ────────────────────────────────────────
    deck_axes(ax)
    diag = np.array([0.0, 1.0])
    ax.plot(diag, diag, color=NEUTRAL["ref"], lw=1.4, ls="--",
            label="Perfect calibration", zorder=2)
    ax.fill_between(ALPHAS_CONT, ALPHAS_CONT, boot_obs, color=C["bootstrap"],
                    alpha=0.16, edgecolor="none", zorder=2)
    glow_line(ax, ALPHAS_CONT, boot_obs, C["bootstrap"],
              label=f"Bootstrap LOO  (MA={boot_ma:.3f})")
    ax.fill_between(ALPHAS_CONT, ALPHAS_CONT, bayes_obs, color=C["bayesian"],
                    alpha=0.16, edgecolor="none", zorder=2)
    glow_line(ax, ALPHAS_CONT, bayes_obs, C["bayesian"],
              label=f"Bayesian in-sample  (MA={bayes_ma:.3f})")
    ax.scatter([0.90, 0.95], [1.0, 1.0], color=C["bayesian"], s=70, marker="D",
               edgecolors=T["fg"], linewidths=0.9, zorder=5,
               label="Bayesian LOO-confirmed")
    ax.set_xlim(0.35, 1.01)
    ax.set_ylim(0.35, 1.06)
    ax.set_xlabel(r"Nominal coverage $\alpha$")
    ax.set_ylabel("Empirical PICP")
    ax.legend(**{**LEGEND_KW, "fontsize": FONTSIZE["annotation"]}, loc="upper left")
    label_panel(ax, "a")

    # ── Panel (b): coverage-gap bars ─────────────────────────────────────────
    deck_axes(ax2)
    gap_b = np.abs(boot_disc - disc)
    gap_y = np.abs(bayes_disc - disc)
    w = 0.013
    ax2.bar(disc - w, gap_b, width=w * 1.8, color=C["bootstrap"], alpha=0.9,
            label="Bootstrap (LOO)", zorder=3)
    ax2.bar(disc + w, gap_y, width=w * 1.8, color=C["bayesian"], alpha=0.9,
            label="Bayesian (in-sample)", zorder=3)
    ax2.axhline(0, color=T["grid"], lw=1.0)
    ax2.set_xlim(0.47, 1.0)
    ax2.set_xlabel(r"Nominal coverage $\alpha$")
    ax2.set_ylabel(r"$|\mathrm{PICP} - \alpha|$  (coverage gap)")
    ax2.legend(**LEGEND_KW)
    label_panel(ax2, "b")

    save_deck(fig, OUT_BASE)


if __name__ == "__main__":
    main()
