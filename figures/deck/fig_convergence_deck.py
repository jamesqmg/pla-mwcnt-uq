"""
DECK figure — MCMC convergence / "can we trust the sampler?" (dark/neon Allotrope).
Slide 8: replaces the cut `nuts-chain-trace.mp4` (which animated a decorative
Metropolis-Hastings walk, not the real NUTS posterior). This figure is honest by
construction: it plots the REAL 4-chain NUTS trace of the CNT-loading slope from a
re-run of the primary model (degree-1, flat prior, 4x1000, seed=42), and quotes the
canonical primary diagnostics from results/phase2_informed_prior.json.

The 5-second takeaway for a non-stats panel: four independent chains overlap into one
band -> the sampler converged -> the intervals are trustworthy even at n=26.

Left panel : 4-chain trace of beta1 (CNT effect slope, MPa per wt%) -> "they agree".
Right panel: diagnostics card -> R-hat < 1.01 and ESS > 400, both passed.

Data : results/phase2_informed_prior.json (canonical diagnostics) + live NUTS re-run.
Run  : ~/miniconda3/bin/python figures/deck/fig_convergence_deck.py
Out  : Presentations/current/figures/convergence_deck.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

from figures.deck_style import (  # noqa: E402
    PALETTE as C, NEUTRAL, SIZES, FONTSIZE, T,
    apply_deck, deck_axes, label_panel, save_deck, glow_line,
)
from build_fig6_reliability import _load_xy  # noqa: E402
from src.uq.bayesian import standardize, build_model, run_nuts  # noqa: E402

DIAG_PATH = ROOT / "results" / "phase2_informed_prior.json"
OUT_BASE  = ROOT / "Presentations" / "current" / "figures" / "convergence_deck"

# Chain colors: hero violet, orange, teal, plus a soft sky for the 4th.
CHAIN_COLORS = [C["bootstrap"], C["bayesian"], C["hier"], "#7DD3FC"]


def _fit_real_trace():
    """Re-run the primary model and return per-chain beta1 draws in raw MPa/wt%."""
    x, y = _load_xy()
    x_std, y_std, params = standardize(x, y)
    model = build_model(x_std, y_std, degree=1, slope_prior=None)
    idata = run_nuts(model, draws=1000, tune=1000, chains=4, seed=42)

    # beta has shape (chain, draw, 2); index 1 is the slope. Convert to raw units.
    beta1_std = idata.posterior["beta"].values[:, :, 1]          # (chain, draw)
    slope_raw = beta1_std * params["y_std"] / params["x_std"]    # MPa per wt%

    # Self-consistency check (printed, not displayed): should match canonical.
    rhat_run = float(az.rhat(idata).to_array().max())
    ess_run  = float(az.ess(idata).to_array().min())
    print(f"  re-run check: max R-hat={rhat_run:.4f}  min ESS={ess_run:.0f}")
    return slope_raw


def main():
    diag = json.loads(DIAG_PATH.read_text())
    prim = diag["flat_prior"]                      # canonical primary result
    rhat = prim["rhat_max"]
    ess  = prim["ess_min"]
    cfg  = diag["sampler"]                          # draws/tune/chains

    print("Re-running primary NUTS for the real trace (4x1000)…")
    slope_raw = _fit_real_trace()
    n_chains, n_draws = slope_raw.shape
    draw_idx = np.arange(n_draws)
    post_mean = slope_raw.mean()

    apply_deck()
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=SIZES["wide2"], gridspec_kw={"width_ratios": [1.85, 1.0]}
    )

    # ── Panel (a): 4-chain trace of the CNT-effect slope ─────────────────────
    deck_axes(ax)
    for k in range(n_chains):
        ax.plot(draw_idx, slope_raw[k], color=CHAIN_COLORS[k], lw=0.5,
                alpha=0.45, zorder=2)
    # Posterior mean reference — the common answer all chains settle on.
    ax.axhline(post_mean, color=T["fg"], lw=1.2, ls="--", alpha=0.7, zorder=3)
    ax.text(n_draws * 0.985, post_mean, "  posterior mean", color=T["fg"],
            fontsize=FONTSIZE["annotation"], ha="right", va="bottom")

    ax.set_xlim(0, n_draws)
    ax.set_xlabel("Post-warmup draw")
    ax.set_ylabel("CNT effect on strength  (MPa per wt%)")
    ax.set_title("Four independent chains, same answer",
                 color=T["fg"], fontsize=FONTSIZE["panel"], pad=8)
    label_panel(ax, "a")

    # Tiny inline chain legend (top-left, inside the panel).
    for k in range(n_chains):
        ax.plot([], [], color=CHAIN_COLORS[k], lw=2.4, label=f"chain {k + 1}")
    ax.legend(loc="upper left", frameon=False, ncol=2,
              fontsize=FONTSIZE["annotation"], labelcolor=T["fg"],
              handlelength=1.3, columnspacing=1.1, handletextpad=0.5)

    # ── Panel (b): diagnostics "trust" card ──────────────────────────────────
    ax2.axis("off")

    def row(y, name, value, target, ok=True):
        mark = r"$\checkmark$" if ok else r"$\times$"
        mc = C["hier"] if ok else C["bayesian"]
        ax2.text(0.0, y, name, color=NEUTRAL["sub"],
                 fontsize=FONTSIZE["annotation"], ha="left", va="center")
        ax2.text(0.0, y - 0.085, value, color=T["fg"],
                 fontsize=FONTSIZE["panel"] + 6, ha="left", va="center",
                 fontweight="bold")
        ax2.text(1.0, y - 0.085, mark, color=mc,
                 fontsize=FONTSIZE["panel"] + 6, ha="right", va="center",
                 fontweight="bold")
        ax2.text(1.0, y + 0.005, target, color=NEUTRAL["sub"],
                 fontsize=FONTSIZE["annotation"] - 1, ha="right", va="center")

    row(0.86, r"Gelman–Rubin  $\hat{R}$  (max)", f"{rhat:.4f}", "want < 1.01", ok=rhat < 1.01)
    row(0.55, "Effective sample size (min)", f"{ess:,.0f}", "want > 400", ok=ess > 400)

    ax2.text(0.0, 0.26,
             f"{cfg['chains']} chains × {cfg['draws']:,} draws\n"
             f"(+{cfg['tune']:,} warm-up, discarded)",
             color=NEUTRAL["sub"], fontsize=FONTSIZE["annotation"], va="top")
    ax2.text(0.0, 0.04,
             "Chains that overlap are mixing —\nthe intervals are trustworthy at n = 26.",
             color=T["fg"], fontsize=FONTSIZE["annotation"], va="top",
             linespacing=1.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    save_deck(fig, OUT_BASE)


if __name__ == "__main__":
    main()
