"""
F3 — Hiremath Dataset Distribution (Ch2 §2.5 after "r = 0.03")
Panel (a): Histogram of Tensile Strength [MPa] with KDE and ±1σ annotation.
Panel (b): Scatter CNT wt% vs Tensile Strength with marginal histograms.
Live data from Hiremath XLSX.
Output: fig_dataset_dist.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

DATA_PATH = '../../data/raw/hiremath/File_A1_A2_Encoded_Variables_and_Descriptions.xlsx'

df = pd.read_excel(DATA_PATH, sheet_name=0)
ts  = df['Tensile Strength Output (MPa)'].values
cnt = df['CNT Weight Fraction (%)'].values

mean_ts = ts.mean()
std_ts  = ts.std()
cv_ts   = std_ts / mean_ts * 100
r_val   = np.corrcoef(cnt, ts)[0, 1]

fig = plt.figure(figsize=(11, 4.5))
gs  = GridSpec(1, 2, figure=fig, wspace=0.32)

# ── Panel (a): Histogram + KDE ───────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

n_bins = 22
counts, edges, patches = ax1.hist(ts, bins=n_bins, color='#2176AE', alpha=0.55,
                                    edgecolor='white', linewidth=0.4, density=True, zorder=2)

kde = gaussian_kde(ts, bw_method='scott')
x_kde = np.linspace(ts.min() - 10, ts.max() + 10, 300)
ax1.plot(x_kde, kde(x_kde), color='#1A5FA8', lw=1.8, zorder=3, label='KDE')

ax1.axvline(mean_ts, color='#1A1A2E', lw=1.4, ls='-',  zorder=4, label=f'Mean = {mean_ts:.1f} MPa')
ax1.axvline(mean_ts - std_ts, color='#E07B39', lw=1.1, ls='--', zorder=4, label=f'±1σ ({std_ts:.1f} MPa)')
ax1.axvline(mean_ts + std_ts, color='#E07B39', lw=1.1, ls='--', zorder=4)

# CV annotation
ax1.text(0.97, 0.96, f'CV = {cv_ts:.1f}%\nn = {len(ts)}',
         transform=ax1.transAxes, ha='right', va='top', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                   edgecolor='#AAAAAA', alpha=0.9))

ax1.set_xlabel('Tensile Strength [MPa]', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.grid(True, linestyle=':', linewidth=0.5, color='#BBBBBB')
ax1.spines[['top', 'right', 'bottom', 'left']].set_color('black')
ax1.tick_params(labelsize=9)
ax1.legend(fontsize=8.5, loc='upper right', framealpha=0.9, edgecolor='black')
ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes,
         fontsize=10, fontweight='bold', va='top')

# ── Panel (b): Scatter CNT wt% vs TS with marginal hists ─────────────────────
# Using GridSpec sub-grid for main + marginals
gs_b = gs[0, 1].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                              hspace=0.04, wspace=0.04)
ax_main  = fig.add_subplot(gs_b[1, 0])
ax_top   = fig.add_subplot(gs_b[0, 0], sharex=ax_main)
ax_right = fig.add_subplot(gs_b[1, 1], sharey=ax_main)

ax_main.scatter(cnt, ts, s=20, color='#888888', alpha=0.50,
                edgecolors='none', zorder=2)

# Flat linear fit showing near-zero slope
m_fit = np.polyfit(cnt, ts, 1)
x_fit = np.linspace(cnt.min(), cnt.max(), 100)
ax_main.plot(x_fit, np.polyval(m_fit, x_fit),
             color='#C03A3A', lw=1.4, ls='-', zorder=3,
             label=f'Linear fit (r = {r_val:.3f})')

ax_main.set_xlabel('CNT Weight Fraction [wt%]', fontsize=11)
ax_main.set_ylabel('Tensile Strength [MPa]', fontsize=11)
ax_main.grid(True, linestyle=':', linewidth=0.5, color='#BBBBBB')
ax_main.spines[['top', 'right', 'bottom', 'left']].set_color('black')
ax_main.tick_params(labelsize=9)
ax_main.legend(fontsize=8.5, loc='upper right', framealpha=0.9, edgecolor='black')
ax_main.text(0.02, 0.97, '(b)', transform=ax_main.transAxes,
             fontsize=10, fontweight='bold', va='top')

# Top marginal — CNT wt%
ax_top.hist(cnt, bins=18, color='#5BA4CF', alpha=0.70,
            edgecolor='white', linewidth=0.3)
ax_top.axis('off')

# Right marginal — TS
ax_right.hist(ts, bins=18, color='#2176AE', alpha=0.70,
              edgecolor='white', linewidth=0.3, orientation='horizontal')
ax_right.axis('off')

# ── Overall title ─────────────────────────────────────────────────────────────
fig.suptitle('Hiremath (2025) Dataset Distribution — 200 CNT-Polymer Nanocomposites\n'
             '(a) Tensile strength histogram (CV = 47.3%);  '
             '(b) CNT wt% vs Tensile Strength (r = 0.029) with marginals',
             fontsize=11, y=1.02)

fig.text(0.5, -0.02,
         'Data: Hiremath et al. (2025), DOI: 10.3390/jcs9070364. '
         'Script: writeup/figures/fig_dataset_dist.py',
         ha='center', va='top', fontsize=7, style='italic', color='#777777')

plt.tight_layout()
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
