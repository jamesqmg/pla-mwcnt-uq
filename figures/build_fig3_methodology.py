"""
Figure 3 — End-to-end Computational Methodology  (v2 — embedded art icons)
Thesis §3.1

Layout philosophy
-----------------
Inspired by fig3_p1p2_comparison.png: each step is a horizontally-split box
with a TEXT LABEL on the left and a SCHEMATIC MINI-ICON on the right.
No equations or Greek letters live in box bodies — they belong in captions.
The icons carry the visual semantics of the operation.

Structure
---------
  START                                (terminal, top)
  ── Shared inputs (centre column) ──
  Dataset                              icon: data-cell grid
  PLA+FDM+D638 filter                  icon: funnel
  Linear regression model              icon: scatter + fitted line
  ── Split: Phase 1 left, Phase 2 right ──
  P1: Resample              | P2: Prior              (bell curve)
  P1: Refit                 | P2: Likelihood         (data + arrow into curve)
  P1: Empirical PI          | P2: NUTS               (4 chain traces)
                            | P2: Convergence diamond (R-hat=1, ESS bar)
                            | P2: Posterior predictive (density + CI band)
  ── Merge (centre) ──
  Metrics                              icon: bar chart (PICP/MPIW/CRPS/NLPD/ECE)
  Subsampling experiment               icon: n-ladder step bars
  Divergence curves                    icon: PICP↑ + MPIW↓ vs n
  STOP                                 (terminal, bottom)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from figures.style import (
    PALETTE as C, SIZES, LW, DPI_PNG,
    apply_style, save_figure, arrow_main,
    FONTSIZE, FLOWCHART, MATH, LEGEND_KW, MULTI_SERIES,
)

# ── Figure dimensions ────────────────────────────────────────────────────────
FIG_W, FIG_H = 8.0, 11.0

# Box dimensions
BW_S = 0.46   # shared (centre column) box width
BW_C = 0.36   # column (Phase 1 or Phase 2) box width
BH   = 0.085  # standard box height

# Column centres
CX_C = 0.50
CX_L = 0.24   # Phase 1 column centre
CX_R = 0.76   # Phase 2 column centre

ARROW = arrow_main(C["dark"])


# ── Drawing primitives ───────────────────────────────────────────────────────

def _box(ax, cx, cy, w, h, fill, edge=None, ls="-", lw=None):
    """Plain rounded rectangle (no text)."""
    edge = edge or C["dark"]
    lw   = lw or LW["spine"]
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012",
        facecolor=fill, edgecolor=edge,
        linewidth=lw, linestyle=ls, alpha=0.92, zorder=3,
    )
    ax.add_patch(rect)


def _step(ax, cx, cy, w, h, fill, title, sub, icon_fn):
    """Box with left-aligned title + sub-text and right-side icon."""
    _box(ax, cx, cy, w, h, fill)
    # Left half: text
    tx = cx - w / 2 + 0.012
    ax.text(tx, cy + h * 0.13, title,
            ha="left", va="center",
            fontsize=FONTSIZE["annotation"],
            fontweight="bold", color="black", zorder=4)
    if sub:
        ax.text(tx, cy - h * 0.18, sub,
                ha="left", va="center",
                fontsize=FONTSIZE["annotation"] - 1,
                color=C["dark"], zorder=4,
                linespacing=1.2)
    # Right half: icon canvas (limit drawing area)
    icx = cx + w * 0.22
    icy = cy
    iw  = w * 0.36
    ih  = h * 0.72
    icon_fn(ax, icx, icy, iw, ih)


def _oval(ax, cx, cy, w, h, label, fill):
    ell = mpatches.Ellipse((cx, cy), w, h,
                           facecolor=fill, edgecolor=C["dark"],
                           linewidth=LW["spine"], alpha=0.92, zorder=3)
    ax.add_patch(ell)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=FONTSIZE["axis_label"], fontweight="bold", zorder=4)


def _arrow(ax, x0, y0, x1, y1, color=None, label=None, lw_mult=1.0):
    props = dict(ARROW)
    if color:
        props["color"] = color
    if lw_mult != 1.0:
        props["lw"] = props["lw"] * lw_mult
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=props, zorder=2)
    if label:
        ax.text((x0 + x1) / 2 + 0.012, (y0 + y1) / 2, label,
                fontsize=FONTSIZE["annotation"] - 1,
                ha="left", va="center", color=C["mid"], zorder=5)


# ── Mini-icon library ────────────────────────────────────────────────────────
# Each function draws into a centred (cx, cy) canvas of (w, h).
# Coords are figure-fraction; keep drawing inside the bounding box.

def _ic_data(ax, cx, cy, w, h):
    """Data table: 2x6 cell grid → 'n=26 pairs'."""
    cols, rows = 6, 2
    cell_w = w / cols
    cell_h = h / rows
    x0 = cx - w / 2
    y0 = cy - h / 2
    for r in range(rows):
        for k in range(cols):
            ax.add_patch(mpatches.Rectangle(
                (x0 + k * cell_w, y0 + r * cell_h),
                cell_w * 0.9, cell_h * 0.85,
                facecolor=C["light"] if (r + k) % 2 else C["mid"],
                edgecolor=C["dark"], linewidth=0.4, zorder=5,
            ))


def _ic_funnel(ax, cx, cy, w, h):
    """Filter funnel."""
    pts = np.array([
        [cx - w / 2,         cy + h / 2],
        [cx + w / 2,         cy + h / 2],
        [cx + w * 0.10,      cy - h * 0.15],
        [cx + w * 0.10,      cy - h / 2],
        [cx - w * 0.10,      cy - h / 2],
        [cx - w * 0.10,      cy - h * 0.15],
    ])
    ax.add_patch(plt.Polygon(pts, closed=True,
                              facecolor=C["light"],
                              edgecolor=C["dark"],
                              linewidth=0.7, zorder=5))
    # Small descending dots above funnel mouth
    for dy in [0.65, 0.45, 0.25]:
        ax.plot(cx - w * 0.18, cy + h * dy / 2,
                marker="o", ms=2.5, color=C["dark"], zorder=6)


def _ic_scatter_line(ax, cx, cy, w, h):
    """Scatter with fitted regression line."""
    rng = np.random.default_rng(11)
    xs = np.linspace(cx - w * 0.40, cx + w * 0.40, 7)
    slope = h * 0.65 / w
    ys = cy - h * 0.05 + slope * (xs - cx) + rng.normal(0, h * 0.045, 7)
    ax.scatter(xs, ys, s=6, color=C["dark"], zorder=6)
    ax.plot([cx - w * 0.42, cx + w * 0.42],
            [cy - h * 0.05 - slope * w * 0.42, cy - h * 0.05 + slope * w * 0.42],
            color=C["bootstrap"], lw=0.9, zorder=5)


def _ic_resample(ax, cx, cy, w, h):
    """3 overlapping copies of the dataset (blue, fading) + '×B' label."""
    rng = np.random.default_rng(7)
    for i, alpha in enumerate([0.30, 0.55, 0.95]):
        dx = (1 - i) * w * 0.04
        dy = (1 - i) * h * 0.06
        for r in range(2):
            for k in range(5):
                ax.add_patch(mpatches.Rectangle(
                    (cx - w * 0.42 + dx + k * w * 0.10,
                     cy - h * 0.30 + dy + r * h * 0.20),
                    w * 0.08, h * 0.16,
                    facecolor=C["bootstrap"], alpha=alpha,
                    edgecolor=C["dark"], linewidth=0.3, zorder=5 + i,
                ))
    ax.text(cx + w * 0.42, cy + h * 0.30, "×B",
            ha="right", va="top",
            fontsize=FONTSIZE["annotation"] - 1,
            color=C["bootstrap"], fontweight="bold", zorder=8)


def _ic_refit(ax, cx, cy, w, h):
    """Multiple thin fitted lines (bootstrap regression family)."""
    rng = np.random.default_rng(13)
    xs = np.array([cx - w * 0.42, cx + w * 0.42])
    base_slope = h * 0.55 / w
    for _ in range(8):
        slope = base_slope + rng.normal(0, h * 0.10 / w)
        intercept = cy - h * 0.05 + rng.normal(0, h * 0.06)
        ys = intercept + slope * (xs - cx)
        ax.plot(xs, ys, color=C["bootstrap"],
                lw=0.5, alpha=0.55, zorder=5)
    # Mean line on top
    ys_mean = cy - h * 0.05 + base_slope * (xs - cx)
    ax.plot(xs, ys_mean, color=C["bootstrap"],
            lw=1.1, alpha=1.0, zorder=6)


def _ic_pi_band(ax, cx, cy, w, h):
    """Regression line + shaded 90% prediction band."""
    xs = np.linspace(cx - w * 0.42, cx + w * 0.42, 40)
    slope = h * 0.45 / w
    mid = cy - h * 0.05 + slope * (xs - cx)
    band = h * 0.18
    ax.fill_between(xs, mid - band, mid + band,
                    facecolor=C["bootstrap"], alpha=0.30,
                    edgecolor="none", zorder=4)
    ax.plot(xs, mid, color=C["bootstrap"], lw=0.9, zorder=5)
    # A few dots
    rng = np.random.default_rng(15)
    xs_pt = rng.uniform(cx - w * 0.38, cx + w * 0.38, 5)
    ys_pt = cy - h * 0.05 + slope * (xs_pt - cx) + rng.normal(0, h * 0.10, 5)
    ax.scatter(xs_pt, ys_pt, s=5, color=C["dark"], zorder=6)


def _ic_prior(ax, cx, cy, w, h):
    """Bell curve — Normal(0, 1) prior."""
    xs = np.linspace(cx - w * 0.42, cx + w * 0.42, 60)
    sigma = w * 0.15
    ys = cy - h * 0.30 + h * 0.60 * np.exp(-((xs - cx) / sigma) ** 2)
    ax.fill_between(xs, cy - h * 0.30, ys,
                    facecolor=C["bayesian"], alpha=0.25, edgecolor="none",
                    zorder=4)
    ax.plot(xs, ys, color=C["bayesian"], lw=1.0, zorder=5)
    ax.plot([cx - w * 0.42, cx + w * 0.42],
            [cy - h * 0.30, cy - h * 0.30],
            color=C["dark"], lw=0.5, zorder=5)


def _ic_likelihood(ax, cx, cy, w, h):
    """Data points with vertical likelihood ribbons."""
    xs = np.linspace(cx - w * 0.36, cx + w * 0.36, 5)
    slope = h * 0.45 / w
    ys = cy - h * 0.05 + slope * (xs - cx)
    # Line
    ax.plot([cx - w * 0.42, cx + w * 0.42],
            [cy - h * 0.05 - slope * w * 0.42, cy - h * 0.05 + slope * w * 0.42],
            color=C["bayesian"], lw=0.7, alpha=0.6, zorder=4)
    # Vertical 'likelihood ribbons' (mini-Gaussian density at each x)
    for xi, yi in zip(xs, ys):
        gy = np.linspace(yi - h * 0.20, yi + h * 0.20, 20)
        gx = xi + (h * 0.20 - np.abs(gy - yi)) * 0.04 / h
        ax.plot(gx, gy, color=C["bayesian"], lw=0.6, alpha=0.7, zorder=5)
        ax.plot(2 * xi - gx, gy, color=C["bayesian"], lw=0.6, alpha=0.7, zorder=5)
    ax.scatter(xs, ys, s=6, color=C["dark"], zorder=6)


def _ic_nuts(ax, cx, cy, w, h):
    """Trace plot — 4 MCMC chains."""
    rng = np.random.default_rng(19)
    xs = np.linspace(cx - w * 0.42, cx + w * 0.42, 80)
    chain_colors = [MULTI_SERIES[0], MULTI_SERIES[1],
                    MULTI_SERIES[2], MULTI_SERIES[5]]
    for i, col in enumerate(chain_colors):
        offset = (i - 1.5) * h * 0.08
        trace = (cy + offset
                 + h * 0.06 * np.cumsum(rng.normal(0, 0.5, len(xs))) / np.sqrt(len(xs)))
        ax.plot(xs, trace, color=col, lw=0.55, alpha=0.85, zorder=5)


def _ic_converge(ax, cx, cy, w, h):
    """4 chains converged to flat band + check mark."""
    rng = np.random.default_rng(23)
    xs = np.linspace(cx - w * 0.42, cx + w * 0.05, 50)
    band = h * 0.06
    for i, col in enumerate([MULTI_SERIES[0], MULTI_SERIES[1],
                              MULTI_SERIES[2], MULTI_SERIES[5]]):
        trace = cy + rng.normal(0, band * 0.6, len(xs))
        ax.plot(xs, trace, color=col, lw=0.5, alpha=0.8, zorder=5)
    # Check mark on the right
    cx_chk = cx + w * 0.25
    ax.plot([cx_chk - w * 0.08, cx_chk - w * 0.02, cx_chk + w * 0.12],
            [cy, cy - h * 0.18, cy + h * 0.22],
            color=C["hier"], lw=2.0, solid_capstyle="round", zorder=7)


def _ic_post_pred(ax, cx, cy, w, h):
    """Smooth posterior predictive density with 90% CI shaded."""
    xs = np.linspace(cx - w * 0.45, cx + w * 0.45, 100)
    sigma = w * 0.16
    ys = cy - h * 0.30 + h * 0.55 * np.exp(-((xs - cx) / sigma) ** 2)
    # CI band shading inside ±1.645*sigma
    mask = np.abs(xs - cx) < sigma * 1.4
    ax.fill_between(xs[mask], cy - h * 0.30, ys[mask],
                    facecolor=C["bayesian"], alpha=0.35, edgecolor="none",
                    zorder=4)
    ax.fill_between(xs[~mask], cy - h * 0.30, ys[~mask],
                    facecolor=C["bayesian"], alpha=0.12, edgecolor="none",
                    zorder=4)
    ax.plot(xs, ys, color=C["bayesian"], lw=1.0, zorder=5)
    ax.plot([cx - w * 0.45, cx + w * 0.45],
            [cy - h * 0.30, cy - h * 0.30],
            color=C["dark"], lw=0.5, zorder=5)


def _ic_metrics(ax, cx, cy, w, h):
    """Mini bar chart: 5 metric bars in alternating method colors."""
    labels = ["PICP", "MPIW", "CRPS", "NLPD", "ECE"]
    heights = [0.85, 0.55, 0.65, 0.45, 0.50]
    bar_xs = np.linspace(cx - w * 0.40, cx + w * 0.40, len(labels))
    bar_w = w * 0.10
    for i, (x, hgt) in enumerate(zip(bar_xs, heights)):
        col = C["bootstrap"] if i % 2 == 0 else C["bayesian"]
        ax.add_patch(mpatches.Rectangle(
            (x - bar_w / 2, cy - h * 0.30),
            bar_w, hgt * h * 0.55,
            facecolor=col, edgecolor=C["dark"], linewidth=0.4,
            alpha=0.85, zorder=5,
        ))
    ax.plot([cx - w * 0.45, cx + w * 0.45],
            [cy - h * 0.30, cy - h * 0.30],
            color=C["dark"], lw=0.5, zorder=5)


def _ic_nladder(ax, cx, cy, w, h):
    """Stair-step bars: n-ladder from small to large."""
    ns = [5, 8, 10, 15, 20, 26, 30, 35, 44]
    bar_xs = np.linspace(cx - w * 0.42, cx + w * 0.42, len(ns))
    bar_w = w * 0.06
    for x, n in zip(bar_xs, ns):
        ax.add_patch(mpatches.Rectangle(
            (x - bar_w / 2, cy - h * 0.30),
            bar_w, (n / 50) * h * 0.62,
            facecolor=C["amber"], edgecolor=C["dark"],
            linewidth=0.3, alpha=0.85, zorder=5,
        ))


def _ic_divergence(ax, cx, cy, w, h):
    """Two curves: PICP rising, MPIW falling vs n."""
    xs = np.linspace(cx - w * 0.42, cx + w * 0.42, 60)
    # PICP: sigmoid rising
    norm_x = (xs - (cx - w * 0.42)) / (w * 0.84)
    picp = cy - h * 0.25 + h * 0.45 * (1 - np.exp(-norm_x * 3))
    # MPIW: decaying
    mpiw = cy - h * 0.05 + h * 0.40 * np.exp(-norm_x * 3)
    ax.plot(xs, picp, color=C["bootstrap"], lw=1.1, zorder=5)
    ax.plot(xs, mpiw, color=C["bayesian"], lw=1.1, ls="--", zorder=5)
    # Threshold marker (n*)
    n_star_x = cx - w * 0.42 + w * 0.84 * 0.30
    ax.axvline(n_star_x, ymin=0, ymax=1,
               color=C["amber"], lw=0.7, ls=":", zorder=6,
               clip_on=True)


# ── Decision diamond ─────────────────────────────────────────────────────────

def _diamond(ax, cx, cy, w, h, title, sub):
    pts = np.array([
        [cx,         cy + h / 2],
        [cx + w / 2, cy        ],
        [cx,         cy - h / 2],
        [cx - w / 2, cy        ],
    ])
    ax.add_patch(plt.Polygon(pts, closed=True,
                              facecolor=FLOWCHART["decision"],
                              edgecolor=C["dark"],
                              linewidth=LW["spine"], zorder=3))
    ax.text(cx, cy + h * 0.10, title,
            ha="center", va="center",
            fontsize=FONTSIZE["annotation"],
            fontweight="bold", zorder=4)
    ax.text(cx, cy - h * 0.18, sub,
            ha="center", va="center",
            fontsize=FONTSIZE["annotation"] - 1,
            color=C["mid"], zorder=4)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Y-coordinates ────────────────────────────────────────────────────────
    y = dict(
        start    = 0.965,
        dataset  = 0.900,
        filter_  = 0.815,
        linreg   = 0.730,
        split    = 0.685,
        # Phase 1 column
        p1_resamp= 0.605,
        p1_refit = 0.515,
        p1_pi    = 0.425,
        # Phase 2 column
        p2_prior = 0.605,
        p2_lik   = 0.515,
        p2_nuts  = 0.425,
        p2_conv  = 0.335,
        p2_post  = 0.245,
        # Shared bottom
        merge    = 0.190,
        metrics  = 0.155,
        nlad     = 0.070,
        nlad_dec = -0.003,   # n-ladder complete? decision diamond
        diverge  = -0.085,
        stop     = -0.145,
    )
    # Extend axes lower to fit STOP + n-ladder decision diamond
    ax.set_ylim(-0.185, 1.0)

    half = BH / 2

    # ── START ────────────────────────────────────────────────────────────────
    _oval(ax, CX_C, y["start"], 0.20, BH * 0.85,
          "START", FLOWCHART["terminal"])

    # ── Shared top: dataset → filter → linear model ──────────────────────────
    _arrow(ax, CX_C, y["start"] - BH * 0.42,
                CX_C, y["dataset"] + half)
    _step(ax, CX_C, y["dataset"], BW_S, BH,
          FLOWCHART["main_fill"],
          title="Literature-extracted dataset",
          sub=f"master_dataset.csv  ({MATH['n']}=45)",
          icon_fn=_ic_data)

    _arrow(ax, CX_C, y["dataset"] - half,
                CX_C, y["filter_"] + half)
    _step(ax, CX_C, y["filter_"], BW_S, BH,
          FLOWCHART["main_fill"],
          title="Material-system filter",
          sub=f"PLA + FDM + ASTM D638 / ISO 527\n{MATH['n']}=26 wt%–UTS pairs",
          icon_fn=_ic_funnel)

    _arrow(ax, CX_C, y["filter_"] - half,
                CX_C, y["linreg"] + half)
    _step(ax, CX_C, y["linreg"], BW_S, BH,
          FLOWCHART["main_fill"],
          title="Linear regression model",
          sub="(held constant across phases)",
          icon_fn=_ic_scatter_line)

    # ── Split arrow into two columns ─────────────────────────────────────────
    _arrow(ax, CX_C, y["linreg"] - half,
                CX_C, y["split"])

    ax.plot([CX_L, CX_R], [y["split"], y["split"]],
            color=C["dark"], lw=LW["schematic"], zorder=2)
    _arrow(ax, CX_L, y["split"], CX_L, y["p1_resamp"] + half, lw_mult=1.0)
    _arrow(ax, CX_R, y["split"], CX_R, y["p2_prior"] + half, lw_mult=1.0)

    # Phase labels sit on the split bus, offset away from the column arrows
    ax.text(CX_L - BW_C / 2 - 0.005, y["split"] + 0.005,
            "Phase 1 — Bootstrap",
            ha="left", va="bottom",
            fontsize=FONTSIZE["annotation"],
            color=C["bootstrap"], fontweight="bold")
    ax.text(CX_R + BW_C / 2 + 0.005, y["split"] + 0.005,
            "Phase 2 — Bayesian NUTS",
            ha="right", va="bottom",
            fontsize=FONTSIZE["annotation"],
            color=C["bayesian"], fontweight="bold")

    # ── Phase 1 column (left) ────────────────────────────────────────────────
    _step(ax, CX_L, y["p1_resamp"], BW_C, BH,
          FLOWCHART["main_fill"],
          title="Bootstrap resample",
          sub=f"B = 1000, with\nreplacement",
          icon_fn=_ic_resample)

    _arrow(ax, CX_L, y["p1_resamp"] - half,
                CX_L, y["p1_refit"]  + half)
    _step(ax, CX_L, y["p1_refit"], BW_C, BH,
          FLOWCHART["main_fill"],
          title="Refit regression",
          sub="(per resample)",
          icon_fn=_ic_refit)

    _arrow(ax, CX_L, y["p1_refit"] - half,
                CX_L, y["p1_pi"]    + half)
    _step(ax, CX_L, y["p1_pi"], BW_C, BH,
          FLOWCHART["main_fill"],
          title="Empirical PI",
          sub=f"5th–95th %ile (90%)",
          icon_fn=_ic_pi_band)

    # ── Phase 2 column (right) ───────────────────────────────────────────────
    _step(ax, CX_R, y["p2_prior"], BW_C, BH,
          FLOWCHART["bayes_fill"],
          title="Priors",
          sub="weakly informative\non standardised inputs",
          icon_fn=_ic_prior)

    _arrow(ax, CX_R, y["p2_prior"] - half,
                CX_R, y["p2_lik"]   + half)
    _step(ax, CX_R, y["p2_lik"], BW_C, BH,
          FLOWCHART["bayes_fill"],
          title="Likelihood",
          sub="Gaussian, fixed model",
          icon_fn=_ic_likelihood)

    _arrow(ax, CX_R, y["p2_lik"] - half,
                CX_R, y["p2_nuts"] + half)
    _step(ax, CX_R, y["p2_nuts"], BW_C, BH,
          FLOWCHART["bayes_fill"],
          title="NUTS sampling",
          sub="4 chains × 1000 draws\n1000 tune",
          icon_fn=_ic_nuts)

    _arrow(ax, CX_R, y["p2_nuts"] - half,
                CX_R, y["p2_conv"] + BH * 0.55)
    _diamond(ax, CX_R, y["p2_conv"], BW_C * 1.05, BH * 1.25,
             "Converged?",
             r"$\hat{R}<1.01$, ESS>400")

    # No → loop back (kept inside figure: route to the LEFT of the column)
    loop_x = CX_R - BW_C * 0.65
    ax.plot([CX_R - BW_C * 0.55, loop_x],
            [y["p2_conv"], y["p2_conv"]],
            color=C["bayesian"], lw=LW["schematic"] * 0.8, zorder=2)
    ax.plot([loop_x, loop_x],
            [y["p2_conv"], y["p2_nuts"]],
            color=C["bayesian"], lw=LW["schematic"] * 0.8, zorder=2)
    ax.annotate("", xy=(CX_R - BW_C * 0.50, y["p2_nuts"]),
                xytext=(loop_x, y["p2_nuts"]),
                arrowprops=dict(arrowstyle="-|>", color=C["bayesian"],
                                lw=LW["schematic"] * 0.8,
                                shrinkA=0, shrinkB=0))
    ax.text(loop_x - 0.008, (y["p2_conv"] + y["p2_nuts"]) / 2,
            "No", color=C["bayesian"],
            fontsize=FONTSIZE["annotation"] - 1,
            ha="right", va="center", fontweight="bold")

    _arrow(ax, CX_R, y["p2_conv"] - BH * 0.55,
                CX_R, y["p2_post"] + half, label="Yes")

    _step(ax, CX_R, y["p2_post"], BW_C, BH,
          FLOWCHART["bayes_fill"],
          title="Posterior predictive",
          sub="burn-in discarded\n90% credible interval",
          icon_fn=_ic_post_pred)

    # ── Bayesian Analysis dashed border block ───────────────────────────────
    bx0 = CX_R - BW_C / 2 - 0.025
    by0 = y["p2_post"] - half - 0.020
    bw  = BW_C + 0.05
    bh  = (y["p2_prior"] + half + 0.025) - by0
    ax.add_patch(mpatches.FancyBboxPatch(
        (bx0, by0), bw, bh,
        boxstyle="round,pad=0.015",
        facecolor="none", edgecolor=FLOWCHART["bayes_border"],
        linewidth=1.2, linestyle="--", zorder=1,
    ))
    # (No duplicate "Bayesian Analysis" label — the dashed border + Phase 2
    #  column header already convey this.)

    # ── Merge to shared bottom ───────────────────────────────────────────────
    _arrow(ax, CX_L, y["p1_pi"]   - half, CX_L, y["merge"])
    _arrow(ax, CX_R, y["p2_post"] - half, CX_R, y["merge"])
    ax.plot([CX_L, CX_R], [y["merge"], y["merge"]],
            color=C["dark"], lw=LW["schematic"], zorder=2)
    _arrow(ax, CX_C, y["merge"], CX_C, y["metrics"] + half)

    # ── Shared bottom: metrics / subsampling / divergence ───────────────────
    _step(ax, CX_C, y["metrics"], BW_S, BH,
          FLOWCHART["main_fill"],
          title="UQ metrics",
          sub="PICP · MPIW · CRPS · NLPD · ECE\n(Uncertainty Toolbox + PSIS-LOO)",
          icon_fn=_ic_metrics)

    _arrow(ax, CX_C, y["metrics"] - half, CX_C, y["nlad"] + half)
    _step(ax, CX_C, y["nlad"], BW_S, BH,
          FLOWCHART["main_fill"],
          title="Subsampling experiment",
          sub=f"{MATH['n']}-ladder [5, 8, 10, 15, 20, 26, 30, 35, 44]\n30 repeats each",
          icon_fn=_ic_nladder)

    # ── n-ladder decision diamond ────────────────────────────────────────────
    _arrow(ax, CX_C, y["nlad"] - half, CX_C, y["nlad_dec"] + BH * 0.60)
    _diamond(ax, CX_C, y["nlad_dec"], BW_S * 0.90, BH * 1.20,
             "n-ladder complete?",
             "all n × 30 repeats done")

    # No → loop back to subsampling experiment (route to right of shared column)
    loop_x3 = CX_C + BW_S * 0.60
    ax.plot([CX_C + BW_S * 0.45, loop_x3],
            [y["nlad_dec"], y["nlad_dec"]],
            color=C["amber"], lw=LW["schematic"] * 0.8, zorder=2)
    ax.plot([loop_x3, loop_x3],
            [y["nlad_dec"], y["nlad"]],
            color=C["amber"], lw=LW["schematic"] * 0.8, zorder=2)
    ax.annotate("", xy=(CX_C + BW_S / 2, y["nlad"]),
                xytext=(loop_x3, y["nlad"]),
                arrowprops=dict(arrowstyle="-|>", color=C["amber"],
                                lw=LW["schematic"] * 0.8,
                                shrinkA=0, shrinkB=0))
    ax.text(loop_x3 + 0.008, (y["nlad_dec"] + y["nlad"]) / 2,
            "No", color=C["amber"],
            fontsize=FONTSIZE["annotation"] - 1,
            ha="left", va="center", fontweight="bold")

    # Yes → divergence curves
    _arrow(ax, CX_C, y["nlad_dec"] - BH * 0.60,
               CX_C, y["diverge"] + half, label="Yes")
    _step(ax, CX_C, y["diverge"], BW_S, BH,
          FLOWCHART["main_fill"],
          title="Divergence curves",
          sub=f"PICP({MATH['n']}) $\\uparrow$ / MPIW({MATH['n']}) $\\downarrow$\n" + r"$\rightarrow$ Decision framework",
          icon_fn=_ic_divergence)

    _arrow(ax, CX_C, y["diverge"] - half, CX_C, y["stop"] + BH * 0.42)
    _oval(ax, CX_C, y["stop"], 0.20, BH * 0.85,
          "STOP", FLOWCHART["terminal"])

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=FLOWCHART["main_fill"],  edgecolor=C["dark"],
                       lw=LW["spine"], label="Shared / Bootstrap step"),
        mpatches.Patch(facecolor=FLOWCHART["bayes_fill"], edgecolor=C["dark"],
                       lw=LW["spine"], label="Bayesian step"),
        mpatches.Patch(facecolor=FLOWCHART["decision"],   edgecolor=C["dark"],
                       lw=LW["spine"], label="Decision node"),
        mpatches.Patch(facecolor=FLOWCHART["terminal"],   edgecolor=C["dark"],
                       lw=LW["spine"], label="Start / Stop"),
        mpatches.Patch(facecolor="none", edgecolor=FLOWCHART["bayes_border"],
                       lw=1.2, linestyle="--", label="Bayesian Analysis block"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower left",
        bbox_to_anchor=(0.00, -0.11),
        ncol=3,
        **LEGEND_KW,
    )

    out = ROOT / "figures" / "fig3_methodology"
    save_figure(fig, out, pad=1.0)


if __name__ == "__main__":
    main()
