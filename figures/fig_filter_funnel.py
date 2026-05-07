"""
F4 — Data Filter Funnel (Ch3 §3.1 after sub-objective 1.1)
Horizontal waterfall chart showing row attrition through three filters.
Placeholders used until EDA notebook confirms exact row counts.
Output: fig_filter_funnel.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data (placeholders) ───────────────────────────────────────────────────────
stages = [
    'Full Hiremath\npool',
    'After Polymer = PLA\nfilter',
    'After Processing =\n3D Printing filter',
    'After ASTM D638\n/ ISO 527 filter',
]
n_vals = [200, 7, 1, '[n_PLA_FILTERED]']

# Bar widths scaled to actual row counts; last bar is placeholder
bar_widths = [200, 7, 1, 1]   # n=1 after 3D Printing; ASTM filter TBD

COLORS = ['#888888', '#5BA4CF', '#3A7ABF', '#1A5FA8']

fig, ax = plt.subplots(figsize=(8.5, 3.8))

bar_height = 0.55
y_positions = list(range(len(stages)))[::-1]   # top-to-bottom

for i, (stage, n, w, col, y) in enumerate(
        zip(stages, n_vals, bar_widths, COLORS, y_positions)):

    ax.barh(y, w, height=bar_height, color=col, alpha=0.88,
            edgecolor='black', linewidth=0.8, zorder=3)

    # n label inside bar (right-aligned)
    label_str = f'n = {n}'
    ax.text(w - 2, y, label_str, va='center', ha='right',
            fontsize=10, color='white', fontweight='bold', zorder=4)

    # % retained label outside bar
    if i > 0:
        pct_val = round(bar_widths[i] / bar_widths[0] * 100, 1)
        suffix  = ' — ASTM filter TBD' if isinstance(n, str) else ''
        pct = f'({pct_val}% retained{suffix})'
        ax.text(w + 3, y, pct, va='center', ha='left',
                fontsize=8, color='#555555', zorder=4)

# Drop arrows between bars
for i in range(len(stages) - 1):
    x_tip = bar_widths[i + 1]
    y_lo  = y_positions[i] - bar_height / 2
    y_hi  = y_positions[i + 1] + bar_height / 2
    ax.annotate('', xy=(x_tip, y_hi + 0.02),
                xytext=(x_tip, y_lo - 0.02),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0))

ax.set_yticks(y_positions)
ax.set_yticklabels(stages, fontsize=10)
ax.set_xlabel('Row count (schematic — placeholder values)', fontsize=10)
ax.set_xlim(0, 260)
ax.set_ylim(-0.55, len(stages) - 0.45)
ax.grid(True, axis='x', linestyle=':', linewidth=0.5, color='#BBBBBB', zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_color('black')
ax.tick_params(labelsize=9)

ax.set_title('Dataset Filter Attrition: Full Hiremath Pool → PLA/MWCNT Case-Study Dataset\n'
             'Three-Axis Filter: Polymer = PLA; Processing = FDM; Standard = ASTM D638 / ISO 527',
             fontsize=11, pad=8)

fig.text(0.5, 0.01,
         'n = 200 → 7 → 1 are live counts from Hiremath et al. (2025). '
         '[n_PLA_FILTERED] resolves once the ASTM D638/ISO 527 sub-filter is applied (Phase 1 sub-objective 1.1).\n'
         'Extremely sparse PLA + 3D-Printing subset (n = 1) explains why the full 200-row pool drives the n-ladder. '
         'Script: writeup/figures/fig_filter_funnel.py',
         ha='center', va='bottom', fontsize=7, style='italic', color='#777777')

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
