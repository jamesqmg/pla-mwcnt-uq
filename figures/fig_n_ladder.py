"""
Subsampling n-ladder diagram.
Actual n-ladder: [5, 8, 10, 15, 20, 26, 30, 35, 44]
Parent: master_dataset.csv (n=44, all PLA, mixed processing)
n=26: PLA+FDM+D638 case-study reference (shown as distinct marker)
Bayesian evaluated at key n-values only: 5, 10, 20, 26, 44
Output: fig_n_ladder.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

N_LADDER   = [5, 8, 10, 15, 20, 26, 30, 35, 44]
BAY_N      = {5, 10, 20, 26, 44}   # Bayesian evaluated at these only
CASE_N     = 26                     # PLA+FDM+D638 case-study reference

COL_BS     = '#2166ac'   # Bootstrap — blue
COL_BAY    = '#d6604d'   # Bayesian  — orange-red
COL_CASE   = '#5C1F6B'   # Case-study marker — purple
COL_NODE   = '#3A3A3A'
COL_SPINE  = '#AAAAAA'

fig, ax = plt.subplots(figsize=(12, 4.2))
ax.set_xlim(-0.7, len(N_LADDER) - 0.3)
ax.set_ylim(-1.2, 2.0)
ax.axis('off')

x_pos = np.arange(len(N_LADDER))

# Spine
ax.plot(x_pos, np.zeros(len(N_LADDER)), color=COL_SPINE, lw=1.5,
        solid_capstyle='round', zorder=1)

# Arrows between nodes
for i in range(len(N_LADDER) - 1):
    ax.annotate('', xy=(x_pos[i + 1] - 0.23, 0),
                xytext=(x_pos[i] + 0.23, 0),
                arrowprops=dict(arrowstyle='->', color=COL_SPINE, lw=1.1))

for i, (xi, n) in enumerate(zip(x_pos, N_LADDER)):
    is_case = (n == CASE_N)
    is_bay  = (n in BAY_N)

    # Node circle
    node_color = COL_CASE if is_case else COL_NODE
    circle = plt.Circle((xi, 0), 0.19, color=node_color, zorder=3)
    ax.add_patch(circle)
    ax.text(xi, 0, str(n), ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold', zorder=4)

    # Bootstrap indicator (all nodes)
    ax.annotate('', xy=(xi + 0.21, 0.62), xytext=(xi - 0.21, 0.62),
                arrowprops=dict(arrowstyle='<->', color=COL_BS, lw=1.4))
    ax.text(xi, 0.75, 'Boot', ha='center', fontsize=7, color=COL_BS)

    # Bayesian indicator (key n-values only)
    if is_bay:
        ax.annotate('', xy=(xi + 0.21, 1.15), xytext=(xi - 0.21, 1.15),
                    arrowprops=dict(arrowstyle='<->', color=COL_BAY, lw=1.4))
        ax.text(xi, 1.28, 'Bayes', ha='center', fontsize=7, color=COL_BAY)

    # Label below: case-study gets annotation
    label = f'n = {n}'
    label_color = COL_CASE if is_case else '#444444'
    ax.text(xi, -0.40, label, ha='center', va='top',
            fontsize=8, color=label_color,
            fontweight='bold' if is_case else 'normal')

    if is_case:
        ax.text(xi, -0.70, 'case-study\nreference', ha='center', va='top',
                fontsize=7, color=COL_CASE, style='italic')

# Direction label
ax.text(-0.6, 0, '← decreasing n', va='center', ha='left',
        fontsize=8, color='#777777', style='italic')

# Legend
legend_handles = [
    mpatches.Patch(color=COL_BS,  label='Bootstrap (B=500, all 9 n-values, 20 repeats each)'),
    mpatches.Patch(color=COL_BAY, label='Bayesian NUTS (draws=500, key n-values: 5, 10, 20, 26, 44)'),
    mpatches.Patch(color=COL_CASE, label='n=26: PLA+FDM+D638 filtered case-study dataset'),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc='lower center',
          bbox_to_anchor=(0.5, -1.05), ncol=1, framealpha=0.9, edgecolor='#CCCCCC')

ax.set_title(
    'Subsampling Experiment — n-Ladder Design\n'
    'Parent: master_dataset.csv (n=44, all PLA, mixed processing)',
    fontsize=11, pad=10)

plt.tight_layout()
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
