#!/usr/bin/env python3
"""figures/build_fig_dag.py — Phase 2b hierarchical model DAG, Allotrope recolor.

Depicts the model the paper actually reports (results/hierarchical_phase2b.json):
a non-centred, partial-pooled GROUP-SLOPE model over g = 2 processing routes
(FDM, non-FDM). There is NO per-paper intercept — that belongs to the separate
Phase 2c model (hierarchical_phase2c_paper_intercept.json), which this paper does
not report, so it is deliberately excluded here.

    beta_pop ~ N(0,1),  tau_beta ~ HalfNormal(0.5)        (population hyperpriors)
    z_g      ~ N(0,1)                                     (non-centred auxiliary, g=2)
    beta_g   = beta_pop + tau_beta * z_g                  (group slope vector, g=2)
    sigma_obs~ HalfNormal(1)
    y_i      ~ N(beta_g[g[i]] . X_i, sigma_obs)           (observed)

On the Allotrope-light canon. Output: fig_dag.png/.svg/.pdf
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import FONTSIZE, apply_style, save_figure  # noqa: E402

# category -> (light fill, saturated edge) on the Allotrope canon
CAT = {
    "hyper": ("#F7DDFB", "#E879F9"),   # orchid — population hyperpriors
    "slope": ("#D6EFFB", "#38BDF8"),   # sky    — group-slope branch (g = 2)
    "noise": ("#F1E7C4", "#C9A227"),   # amber  — observation noise
    "obs":   ("#E3E3E3", "#5B5B66"),   # grey   — observed (shaded)
}

# id: (mathtext label, cx, cy, w, h, category)  — plot coords, y up
NODES = {
    "beta_pop": (r"$\beta_{\mathrm{pop}}\sim\mathcal{N}(0,1)$",         170, 432, 210, 50, "hyper"),
    "tau_b":    (r"$\tau_\beta\sim\mathrm{HalfNormal}(0.5)$",           452, 432, 224, 50, "hyper"),
    "z_g":      (r"$z_g\sim\mathcal{N}(0,1)$",                          311, 330, 180, 46, "slope"),
    "beta_g":   (r"$\beta_g=\beta_{\mathrm{pop}}+\tau_\beta\,z_g$",     311, 222, 252, 52, "slope"),
    "sigma":    (r"$\sigma_{\mathrm{obs}}\sim\mathrm{HalfNormal}(1)$",  612, 96,  196, 52, "noise"),
    "y":        (r"$y_i\sim\mathcal{N}(\beta_{g[i]}\,X_i,\ \sigma_{\mathrm{obs}})$", 300, 96, 330, 58, "obs"),
}

EDGES = [("beta_pop", "beta_g"), ("tau_b", "beta_g"), ("z_g", "beta_g"),
         ("beta_g", "y"), ("sigma", "y")]

LEGEND = [
    ("hyper", "population hyperpriors"),
    ("slope", "group slope, g = 2 (FDM, non-FDM)"),
    ("noise", r"observation noise $\sigma_{\mathrm{obs}}$"),
    ("obs",   "observed"),
]


def main(out: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_xlim(25, 745)
    ax.set_ylim(40, 480)
    ax.set_aspect("equal")
    ax.axis("off")

    # g = 2 plate around the per-group nodes (z_g, beta_g)
    plate = FancyBboxPatch((176, 190), 270, 172, boxstyle="round,pad=0,rounding_size=10",
                           facecolor="none", edgecolor="#9AA0AA", lw=1.1,
                           linestyle=(0, (5, 4)), zorder=1)
    ax.add_patch(plate)
    ax.text(438, 352, "g = 2", ha="right", va="top", fontsize=FONTSIZE["annotation"] - 1,
            color="#7A7F88", style="italic", zorder=2)

    patches = {}
    for nid, (label, cx, cy, w, h, cat) in NODES.items():
        fill, edge = CAT[cat]
        rect = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                              boxstyle="round,pad=0,rounding_size=8",
                              facecolor=fill, edgecolor=edge, linewidth=1.7, zorder=3)
        ax.add_patch(rect)
        patches[nid] = rect
        fs = FONTSIZE["annotation"] + 4 - (1 if nid == "y" else 0)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
                color="#1A1A1A", zorder=4)

    for s, t in EDGES:
        (_, sx, sy, *_), (_, tx, ty, *_) = NODES[s], NODES[t]
        arr = FancyArrowPatch((sx, sy), (tx, ty), patchA=patches[s], patchB=patches[t],
                              shrinkA=1, shrinkB=1, arrowstyle="-|>", mutation_scale=12,
                              lw=1.3, color="#4A4A55", zorder=2)
        ax.add_patch(arr)

    handles = [Patch(facecolor=CAT[c][0], edgecolor=CAT[c][1], label=lab)
               for c, lab in LEGEND]
    # legend BELOW the graph (2 columns) so it never overlaps the observed node
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=2, fontsize=FONTSIZE["annotation"] - 0.5, frameon=True,
              framealpha=1.0, edgecolor="#CCCCCC", handlelength=1.1,
              borderpad=0.6, labelspacing=0.5, columnspacing=1.6)

    save_figure(fig, out, formats=("png", "svg", "pdf"))


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "fig_dag")
