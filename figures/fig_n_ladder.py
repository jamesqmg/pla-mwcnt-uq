"""
F7 — n-Ladder Subsampling Design (Ch3 §3.3 after sub-objective 3.1)
Horizontal step diagram: n = 200 → 100 → 50 → 25 → 10.
Both UQ methods applied at each rung; [n_PLA_FILTERED] shown as reference point.
Output: fig_n_ladder.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

n_ladder   = [200, 100, 50, 25, 10]
n_labels   = ['n = 200\n(Full Hiremath\npool)', 'n = 100', 'n = 50',
               'n = 25', 'n = 10']
x_pos      = np.arange(len(n_ladder))

COL_BS     = '#E07B39'   # Bootstrap — orange
COL_BAY    = '#2176AE'   # Bayesian   — blue
COL_NODE   = '#1A1A2E'

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.set_xlim(-0.6, 4.6)
ax.set_ylim(-0.8, 1.6)
ax.axis('off')

# ── Horizontal spine ──────────────────────────────────────────────────────────
ax.plot(x_pos, np.zeros(len(n_ladder)), color='#AAAAAA', lw=1.5, zorder=1,
        solid_capstyle='round')

# ── Rung nodes and labels ─────────────────────────────────────────────────────
for i, (xi, label, n) in enumerate(zip(x_pos, n_labels, n_ladder)):
    # Main circle
    circle = plt.Circle((xi, 0), 0.18, color=COL_NODE, zorder=3)
    ax.add_patch(circle)
    ax.text(xi, 0, str(n), ha='center', va='center',
            fontsize=9, color='white', fontweight='bold', zorder=4)

    # Label below
    ax.text(xi, -0.38, label, ha='center', va='top',
            fontsize=8.5, color='#333333')

    # Bootstrap bracket (above spine)
    bracket_h = 0.52
    ax.annotate('', xy=(xi + 0.22, bracket_h),
                xytext=(xi - 0.22, bracket_h),
                arrowprops=dict(arrowstyle='<->', color=COL_BS, lw=1.5))
    ax.text(xi, bracket_h + 0.09, 'Bootstrap', ha='center', va='bottom',
            fontsize=7.5, color=COL_BS)

    # Bayesian bracket (higher)
    bracket_h2 = 0.95
    ax.annotate('', xy=(xi + 0.22, bracket_h2),
                xytext=(xi - 0.22, bracket_h2),
                arrowprops=dict(arrowstyle='<->', color=COL_BAY, lw=1.5))
    ax.text(xi, bracket_h2 + 0.09, 'Bayesian MCMC', ha='center', va='bottom',
            fontsize=7.5, color=COL_BAY)

# ── Arrows between rungs ──────────────────────────────────────────────────────
for i in range(len(n_ladder) - 1):
    ax.annotate('', xy=(x_pos[i + 1] - 0.21, 0),
                xytext=(x_pos[i] + 0.21, 0),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2))

# ── [n_PLA_FILTERED] reference marker ────────────────────────────────────────
# Shown as a distinct vertical tick between n=50 and n=25 (schematic position)
ref_x = 2.5
ax.plot([ref_x, ref_x], [-0.25, 0.28], color='#C03A8A',
        lw=1.2, ls='--', zorder=5)
ax.text(ref_x, 0.31, '[n_PLA_FILTERED]\nwithin-ladder\nreference',
        ha='center', va='bottom', fontsize=7.5, color='#C03A8A')
ax.plot(ref_x, 0, marker='D', markersize=7,
        color='#C03A8A', zorder=6, markeredgecolor='white', markeredgewidth=0.6)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=COL_BS,  label='Bootstrap resampling (Phase 1)'),
    mpatches.Patch(color=COL_BAY, label='Bayesian MCMC (Phase 2)'),
    plt.Line2D([0], [0], color='#C03A8A', lw=1.2, ls='--', marker='D',
               markersize=6, label='[n_PLA_FILTERED] — PLA case-study reference'),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc='lower right',
          bbox_to_anchor=(1.01, -0.55), framealpha=0.9, edgecolor='black')

ax.set_title(
    'Subsampling Experiment — n-Ladder Design (n = 200 → 100 → 50 → 25 → 10)\n'
    'Both UQ methods applied at each rung; identical subsamples ensure fair comparison',
    fontsize=11, pad=10)

fig.text(0.5, -0.01,
         'Parent pool: full 200-row Hiremath dataset. '
         '[n_PLA_FILTERED] is the PLA-filtered case-study size — within-ladder reference, not the parent.\n'
         'Fixed random seeds at each n-level ensure reproducibility. '
         'Script: writeup/figures/fig_n_ladder.py',
         ha='center', va='top', fontsize=7, style='italic', color='#777777')

plt.tight_layout()
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
