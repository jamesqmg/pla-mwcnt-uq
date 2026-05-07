"""
F1 — Data Scarcity Scatter (Ch1 after "tensile strength values spread widely")
Panel (a): Full 200-row Hiremath pool — CNT wt% vs Tensile Strength [MPa].
Panel (b): Same axes, PLA subset (n=7) highlighted.
Live data from Hiremath XLSX.
Output: fig_data_scatter.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_PATH = '../../data/raw/hiremath/File_A1_A2_Encoded_Variables_and_Descriptions.xlsx'

df = pd.read_excel(DATA_PATH, sheet_name=0)
x_all  = df['CNT Weight Fraction (%)'].values
y_all  = df['Tensile Strength Output (MPa)'].values

mask_pla = df['Polymer'] == 'PLA'
x_pla    = df.loc[mask_pla, 'CNT Weight Fraction (%)'].values
y_pla    = df.loc[mask_pla, 'Tensile Strength Output (MPa)'].values

r_all = np.corrcoef(x_all, y_all)[0, 1]
cv    = df['Tensile Strength Output (MPa)'].std() / df['Tensile Strength Output (MPa)'].mean() * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5),
                                 gridspec_kw={'wspace': 0.30})

# ── Panel (a): Full pool ──────────────────────────────────────────────────────
ax1.scatter(x_all, y_all, s=22, color='#888888', alpha=0.45,
            edgecolors='none', zorder=2, label='All polymers (n = 200)')
ax1.set_xlabel('CNT Weight Fraction [wt%]', fontsize=11)
ax1.set_ylabel('Tensile Strength [MPa]', fontsize=11)
ax1.set_xlim(-0.3, 10.5)
ax1.set_ylim(10, 215)
ax1.grid(True, linestyle=':', linewidth=0.5, color='#BBBBBB')
ax1.spines[['top', 'right', 'bottom', 'left']].set_color('black')
ax1.tick_params(labelsize=9)

# Stats annotation box
stats_text = f'n = 200\nr = {r_all:.3f}\nCV = {cv:.1f}%'
ax1.text(0.97, 0.96, stats_text, transform=ax1.transAxes,
         ha='right', va='top', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#AAAAAA', alpha=0.9))

ax1.annotate('Near-zero\nlinear correlation\nacross full pool',
             xy=(5.5, 105), xytext=(7.0, 55),
             fontsize=8, color='#555555',
             arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8))

ax1.legend(fontsize=9, loc='upper left', framealpha=0.9, edgecolor='black')
ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes,
         fontsize=10, fontweight='bold', va='top')

# ── Panel (b): PLA highlighted ────────────────────────────────────────────────
ax2.scatter(x_all, y_all, s=18, color='#CCCCCC', alpha=0.30,
            edgecolors='none', zorder=2, label='Other polymers (n = 193)')
ax2.scatter(x_pla, y_pla, s=60, color='#2176AE', alpha=0.90,
            edgecolors='black', linewidths=0.5, zorder=4, label='PLA (n = 7)')

ax2.set_xlabel('CNT Weight Fraction [wt%]', fontsize=11)
ax2.set_ylabel('Tensile Strength [MPa]', fontsize=11)
ax2.set_xlim(-0.3, 10.5)
ax2.set_ylim(10, 215)
ax2.grid(True, linestyle=':', linewidth=0.5, color='#BBBBBB')
ax2.spines[['top', 'right', 'bottom', 'left']].set_color('black')
ax2.tick_params(labelsize=9)

# Annotate PLA spread
if len(y_pla) > 0:
    ax2.annotate(f'PLA: {y_pla.min():.0f}–{y_pla.max():.0f} MPa\nacross {x_pla.min():.2f}–{x_pla.max():.2f} wt%\n(n = {len(y_pla)})',
                 xy=(x_pla.mean(), y_pla.mean()),
                 xytext=(5.5, 170),
                 fontsize=8, color='#2176AE',
                 arrowprops=dict(arrowstyle='->', color='#2176AE', lw=0.9))

ax2.legend(fontsize=9, loc='upper left', framealpha=0.9, edgecolor='black')
ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes,
         fontsize=10, fontweight='bold', va='top')

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle('Hiremath (2025) Dataset: CNT Weight Fraction vs Tensile Strength\n'
             '(a) Full 200-row pool — r ≈ 0.03, CV = 47.3%; '
             '(b) PLA subset (n = 7) highlighted',
             fontsize=11, y=1.01)

fig.text(0.5, -0.02,
         'Data: Hiremath et al. (2025), DOI: 10.3390/jcs9070364. '
         'Script: writeup/figures/fig_data_scatter.py',
         ha='center', va='top', fontsize=7, style='italic', color='#777777')

plt.tight_layout()
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
