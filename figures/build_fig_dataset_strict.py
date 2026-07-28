"""
Fig 2 (ICES paper) — n = 26 strict PLA/MWCNT/FDM dataset characteristics.

Panel (a): tensile-strength distribution (histogram + KDE, mean, ±1σ, CV).
Panel (b): CNT wt% vs tensile strength coloured by SOURCE STUDY, with a linear fit.

Replaces the mistakenly-embedded Hiremath 200-row figure (`fig_dataset_dist.py`,
a thesis Ch-2 prior-art figure) that the ICES build was pointing at. On the
Allotrope-light canon (`style.py`). Output: fig_dataset_dist_strict.png

Layout note: legends are kept off the data — panel (a) uses an opaque card in
added headroom; panel (b)'s study legend sits outside the axes on the right.
The redundant marginal histograms were removed (the right marginal duplicated
panel (a); the top marginal added little the scatter does not already show).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from style import PALETTE, apply_style           # noqa: E402
from src.uq.data_loader import load_dataset, filter_pla_fdm  # noqa: E402

apply_style()
C = PALETTE

df = filter_pla_fdm(load_dataset(str(ROOT / "data/extraction/master_dataset_strict.csv")))
ts = df["tensile_mpa"].values.astype(float)
cnt = df["cnt_wt_pct"].values.astype(float)
papers = df["source_paper"].values

mean_ts, std_ts = ts.mean(), ts.std(ddof=1)
cv_ts = std_ts / mean_ts * 100
r_val = np.corrcoef(cnt, ts)[0, 1]

# Categorical (non-method) colours for the source studies; short labels.
study_order = sorted(np.unique(papers))
study_color = dict(zip(study_order, [C["sky"], C["orchid"], C["amber"]]))
def short(p):
    a = p.split("_")
    return f"{a[0].capitalize()} {a[1]}"

fig = plt.figure(figsize=(11, 4.6))
gs = GridSpec(1, 2, figure=fig, wspace=0.28)

# ── Panel (a): tensile strength distribution ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(ts, bins=12, color=C["violet"], alpha=0.55, edgecolor="white",
         linewidth=0.4, density=True, zorder=2)
kde = gaussian_kde(ts, bw_method="scott")
xk = np.linspace(ts.min() - 5, ts.max() + 5, 300)
ax1.plot(xk, kde(xk), color=C["bootstrap"], lw=1.8, zorder=3, label="KDE")
ax1.axvline(mean_ts, color=C["dark"], lw=1.4, label=f"Mean = {mean_ts:.1f} MPa")
ax1.axvline(mean_ts - std_ts, color=C["bayesian"], lw=1.1, ls="--",
            label=f"±1σ ({std_ts:.1f} MPa)")
ax1.axvline(mean_ts + std_ts, color=C["bayesian"], lw=1.1, ls="--")
# Headroom so the legend + stat box clear the bars and the ±1σ lines.
hist_top = np.histogram(ts, bins=12, density=True)[0].max()
ax1.set_ylim(0, max(hist_top, kde(xk).max()) * 1.42)
ax1.text(0.97, 0.96, f"CV = {cv_ts:.1f}%\nn = {len(ts)}", transform=ax1.transAxes,
         ha="right", va="top", fontsize=9, zorder=6,
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                   edgecolor=C["light"], alpha=1.0))
ax1.set_xlabel("Tensile Strength [MPa]")
ax1.set_ylabel("Density")
leg1 = ax1.legend(fontsize=8.5, loc="upper left", framealpha=1.0,
                  edgecolor=C["dark"])
leg1.set_zorder(6)
ax1.set_title("(a)", loc="left", fontweight="bold", fontsize=11, pad=6)

# ── Panel (b): CNT wt% vs TS, coloured by study ───────────────────────────────
axm = fig.add_subplot(gs[0, 1])
for p in study_order:
    m = papers == p
    axm.scatter(cnt[m], ts[m], s=44, color=study_color[p], alpha=0.9,
                edgecolors="white", linewidth=0.5, label=short(p), zorder=3)
mf = np.polyfit(cnt, ts, 1)
xf = np.linspace(cnt.min(), cnt.max(), 100)
axm.plot(xf, np.polyval(mf, xf), color=C["dark"], lw=1.5, ls="--", zorder=2,
         label=f"Linear fit (r = {r_val:.2f})")
axm.set_xlim(-0.4, 6.6)
axm.set_xlabel("CNT Weight Fraction [wt%]")
axm.set_ylabel("Tensile Strength [MPa]")
# Legend outside the axes (right) so it never sits on a data point.
leg2 = axm.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  framealpha=1.0, edgecolor=C["dark"], borderaxespad=0.0)
leg2.set_zorder(6)
axm.set_title("(b)", loc="left", fontweight="bold", fontsize=11, pad=6)

out = str(HERE / "fig_dataset_dist_strict.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out} | n={len(ts)} CV={cv_ts:.1f}% r={r_val:.3f} "
      f"studies={len(study_order)} ts=[{ts.min():.1f},{ts.max():.1f}] "
      f"cnt=[{cnt.min():.1f},{cnt.max():.1f}]")
