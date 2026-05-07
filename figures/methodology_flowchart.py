"""
methodology_flowchart.py — clean vertical methodology flow diagram
For MSE/ME audience: shows the full experimental design in one figure.
Style: phase boxes with colored headers, minimal text, downward flow.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "methodology_flowchart.png")

# ── colors ────────────────────────────────────────────────────────────────────
C = {
    "data_hdr":  "#4A4A4A",   # dark gray header
    "data_bg":   "#EBEBEB",   # light gray body
    "p1_hdr":    "#2E6B3E",   # forest green header
    "p1_bg":     "#D6EAD9",   # light green body
    "p2_hdr":    "#7D4A00",   # amber/brown header
    "p2_bg":     "#FDE9C9",   # light amber body
    "p3_hdr":    "#2C3E7A",   # navy header
    "p3_bg":     "#D5DCF5",   # light blue body
    "out_hdr":   "#1A1A1A",   # near-black header
    "out_bg":    "#F5F5F5",   # off-white body
    "val_hdr":   "#5C1F6B",   # purple header  (Phase 3b, Thesis 2)
    "val_bg":    "#EEE0F5",   # light purple body
    "arrow":     "#333333",
    "white":     "#FFFFFF",
    "text":      "#111111",
}

fig, ax = plt.subplots(figsize=(8.5, 16))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── drawing helpers ───────────────────────────────────────────────────────────

def box(ax, x, y, w, h, hdr_color, bg_color, header_text,
        body_lines, hdr_h=0.038, fontsize_hdr=9.5, fontsize_body=8.2,
        corner_r=0.012):
    """Draw a phase box: colored header strip + light body with bullet lines."""
    # body rectangle
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={corner_r}",
        facecolor=bg_color, edgecolor="#888888", linewidth=0.8, zorder=2
    ))
    # header rectangle (clipped at top of box)
    ax.add_patch(FancyBboxPatch(
        (x, y + h - hdr_h), w, hdr_h,
        boxstyle=f"round,pad=0,rounding_size={corner_r}",
        facecolor=hdr_color, edgecolor="#888888", linewidth=0.8, zorder=3
    ))
    # solid cover to square off the bottom of the header (remove rounded corners at join)
    ax.add_patch(mpatches.Rectangle(
        (x + corner_r, y + h - hdr_h), w - 2*corner_r, hdr_h * 0.5,
        facecolor=hdr_color, edgecolor="none", zorder=3
    ))
    # header text
    ax.text(x + w/2, y + h - hdr_h/2, header_text,
            ha="center", va="center", fontsize=fontsize_hdr,
            fontweight="bold", color=C["white"], zorder=4)
    # body bullet lines
    line_h = (h - hdr_h - 0.01) / max(len(body_lines), 1)
    for i, line in enumerate(body_lines):
        ty = y + h - hdr_h - 0.012 - i * line_h
        ax.text(x + 0.018, ty, line,
                ha="left", va="top", fontsize=fontsize_body,
                color=C["text"], zorder=4,
                fontfamily="sans-serif")


def arrow(ax, x, y_top, y_bot, color=C["arrow"]):
    """Draw a downward arrow between two y positions at horizontal x."""
    ax.annotate("",
        xy=(x, y_bot), xytext=(x, y_top),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=1.4, mutation_scale=14),
        zorder=5)


def merge_arrow(ax, x1, x2, y_from, y_mid, y_to, color=C["arrow"]):
    """Two branches merging: (x1,y_from) and (x2,y_from) → (mid,y_to)."""
    xmid = (x1 + x2) / 2
    # left branch down then right
    ax.plot([x1, x1, xmid], [y_from, y_mid, y_mid], color=color, lw=1.4, zorder=5)
    # right branch down then left
    ax.plot([x2, x2, xmid], [y_from, y_mid, y_mid], color=color, lw=1.4, zorder=5)
    # single arrow down to next box
    ax.annotate("",
        xy=(xmid, y_to), xytext=(xmid, y_mid),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=1.4, mutation_scale=14),
        zorder=5)


def split_arrow(ax, x_from, y_from, x1, x2, y_to, color=C["arrow"]):
    """Single arrow splitting into two: (x_from,y_from) → left (x1,y_to) + right (x2,y_to)."""
    ymid = (y_from + y_to) / 2
    ax.plot([x_from, x_from, x1, x1], [y_from, ymid, ymid, y_to],
            color=color, lw=1.4, zorder=5)
    ax.plot([x_from, x_from, x2, x2], [y_from, ymid, ymid, y_to],
            color=color, lw=1.4, zorder=5)
    # arrowheads
    ax.annotate("", xy=(x1, y_to), xytext=(x1, y_to + 0.002),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=14), zorder=5)
    ax.annotate("", xy=(x2, y_to), xytext=(x2, y_to + 0.002),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=14), zorder=5)


# ── layout constants ──────────────────────────────────────────────────────────
LM   = 0.06    # left margin
RM   = 0.94    # right margin
W    = RM - LM # full width
GAP  = 0.018   # vertical gap between boxes

# Vertical positions (bottom of each box), top-down
Y_DATA  = 0.840
H_DATA  = 0.130

Y_MODEL = 0.740
H_MODEL = 0.075

# Phase 1 and Phase 2 side by side
HALF_W  = (W - 0.015) / 2      # width of each half-box
X_P1    = LM
X_P2    = LM + HALF_W + 0.015
Y_PH12  = 0.560
H_PH12  = 0.155

Y_P3    = 0.370
H_P3    = 0.165

Y_OUT   = 0.220
H_OUT   = 0.120

Y_VAL   = 0.050
H_VAL   = 0.135


# ── 0. Data preparation ───────────────────────────────────────────────────────
box(ax, LM, Y_DATA, W, H_DATA,
    hdr_color=C["data_hdr"], bg_color=C["data_bg"],
    header_text="Data Preparation",
    hdr_h=0.042,
    body_lines=[
        "▸  Literature dataset: Hiremath et al. (2025) — 200 rows, CNT/polymer tensile",
        "▸  Three-axis filter: PLA matrix   |   FDM / 3D-print processing   |   ASTM D638 / ISO 527",
        "▸  Supplementary extraction: PLA+FDM papers from literature (master_dataset.csv, n=44)",
        "▸  Output: filtered training dataset, n = 26 rows (PLA + FDM + ASTM D638/ISO 527)",
    ])

# Arrow: data → shared model note
arrow(ax, LM + W/2, Y_DATA, Y_MODEL + H_MODEL)

# ── shared model note ─────────────────────────────────────────────────────────
box(ax, LM, Y_MODEL, W, H_MODEL,
    hdr_color="#555555", bg_color="#F0F0F0",
    header_text="Shared Prediction Model (fixed across all phases)",
    hdr_h=0.035,
    fontsize_hdr=8.8,
    body_lines=[
        "▸  Simple linear / polynomial regression   |   Input: CNT wt%   |   Target: Tensile strength (MPa)",
        "▸  Model held constant — UQ method is the only variable of comparison",
    ])

# ── split arrow from model to Phase 1 and Phase 2 ────────────────────────────
split_arrow(ax,
    x_from=LM + W/2, y_from=Y_MODEL,
    x1=X_P1 + HALF_W/2,
    x2=X_P2 + HALF_W/2,
    y_to=Y_PH12 + H_PH12)

# ── Phase 1 — Bootstrap ───────────────────────────────────────────────────────
box(ax, X_P1, Y_PH12, HALF_W, H_PH12,
    hdr_color=C["p1_hdr"], bg_color=C["p1_bg"],
    header_text="Phase 1 — Bootstrap Baseline",
    hdr_h=0.040,
    fontsize_hdr=8.8,
    body_lines=[
        "▸  Bootstrap resampling with replacement",
        "    B = 1,000 resamples",
        "▸  Prediction intervals at 90% nominal",
        "▸  Metrics: PICP, MPIW, CRPS",
        "▸  Tool: scikit-learn resample",
    ])

# ── Phase 2 — Bayesian MCMC ──────────────────────────────────────────────────
box(ax, X_P2, Y_PH12, HALF_W, H_PH12,
    hdr_color=C["p2_hdr"], bg_color=C["p2_bg"],
    header_text="Phase 2 — Bayesian MCMC",
    hdr_h=0.040,
    fontsize_hdr=8.8,
    body_lines=[
        "▸  Bayesian linear regression via PyMC",
        "    NUTS sampler: 4 chains × 1,000 draws (1,000 tune)",
        "▸  Priors: β ~ N(0,1), σ ~ HalfNormal(1)",
        "▸  Metrics: PICP, MPIW, CRPS",
        "▸  Diagnostics: R̂, ESS (arviz)",
    ])

# ── merge arrows Phase 1 + Phase 2 → Phase 3 ─────────────────────────────────
merge_arrow(ax,
    x1=X_P1 + HALF_W/2,
    x2=X_P2 + HALF_W/2,
    y_from=Y_PH12,
    y_mid=Y_P3 + H_P3 + GAP + 0.01,
    y_to=Y_P3 + H_P3)

# ── Phase 3 — Subsampling experiment ─────────────────────────────────────────
box(ax, LM, Y_P3, W, H_P3,
    hdr_color=C["p3_hdr"], bg_color=C["p3_bg"],
    header_text="Phase 3 — Subsampling Experiment & Decision Framework",
    hdr_h=0.042,
    body_lines=[
        "▸  Both methods applied at descending sample sizes (n-ladder):",
        "    n = 44  →  35  →  30  →  26  →  20  →  15  →  10  →  8  →  5",
        "▸  At each n: refit model, rerun UQ, evaluate PICP and MPIW",
        "▸  Output: divergence curves  —  PICP(n) and MPIW(n) for Bootstrap vs. Bayesian",
        "▸  Decision threshold: sample size at which Bayesian calibration diverges from Bootstrap",
    ])

# ── Arrow: Phase 3 → Output ───────────────────────────────────────────────────
arrow(ax, LM + W/2, Y_P3, Y_OUT + H_OUT)

# ── Output ────────────────────────────────────────────────────────────────────
box(ax, LM, Y_OUT, W, H_OUT,
    hdr_color=C["out_hdr"], bg_color=C["out_bg"],
    header_text="Output: UQ Method Selection Framework",
    hdr_h=0.040,
    body_lines=[
        "▸  Evidence-based guidance: which UQ method is appropriate at a given sample size",
        "▸  Calibrated prediction intervals for PLA/MWCNT tensile strength",
        "▸  Generalizable to other data-scarce nanocomposite systems",
    ])

# ── dashed arrow: → Experimental Validation (Thesis 2) ───────────────────────
ax.annotate("",
    xy=(LM + W/2, Y_VAL + H_VAL),
    xytext=(LM + W/2, Y_OUT),
    arrowprops=dict(arrowstyle="-|>", color="#777777",
                    lw=1.2, mutation_scale=12,
                    linestyle="dashed"),
    zorder=5)
ax.text(LM + W/2 + 0.03, (Y_OUT + Y_VAL + H_VAL) / 2,
        "(Thesis 2)", ha="left", va="center",
        fontsize=7.5, color="#777777", style="italic")

# ── Experimental Validation (Thesis 2) ───────────────────────────────────────
box(ax, LM, Y_VAL, W, H_VAL,
    hdr_color=C["val_hdr"], bg_color=C["val_bg"],
    header_text="Experimental Validation (Phase 3 — Thesis 2)",
    hdr_h=0.040,
    body_lines=[
        "▸  FDM-printed PLA/MWCNT specimens  (n ≥ 5)  |  ASTM D638 tensile testing",
        "▸  Measured tensile strength compared against framework's prediction interval",
        "▸  Confirms framework produces calibrated intervals for real in-house measurements",
    ])

# ── figure label ─────────────────────────────────────────────────────────────
ax.text(0.5, 0.01,
        "Figure 11. Computational methodology overview. "
        "Phase 1 and Phase 2 operate on the same filtered dataset and regression model; "
        "Phase 3 characterizes their calibration across descending sample sizes.",
        ha="center", va="bottom", fontsize=7.2,
        color="#444444", style="italic",
        wrap=True, transform=ax.transAxes)

# ── save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0)
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT}")
