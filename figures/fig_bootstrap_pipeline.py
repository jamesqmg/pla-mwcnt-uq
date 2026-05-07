"""
F5 — Bootstrap Pipeline Detail (Ch3 §3.1 after sub-objective 1.3)
Schematic step diagram: dataset → resample → refit → stack → interval band.
No data dependencies — fully illustrative.
Output: fig_bootstrap_pipeline.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COL_DATA  = '#2176AE'
COL_RESAMP= '#5BA4CF'
COL_FIT   = '#E07B39'
COL_BAND  = '#888888'

fig, axes = plt.subplots(1, 4, figsize=(13, 5.2),
                          gridspec_kw={'wspace': 0.14})

for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

# ── Helper: draw arrow between axes ─────────────────────────────────────────
def inter_arrow(fig, ax_left, ax_right, label='', color='#555555'):
    """Draw annotation arrow from right edge of ax_left to left edge of ax_right."""
    x_l = ax_left.get_position().x1
    x_r = ax_right.get_position().x0
    y_c = (ax_left.get_position().y0 + ax_left.get_position().y1) / 2
    ax_fig = fig.add_axes([0, 0, 1, 1], zorder=10)
    ax_fig.set_xlim(0, 1); ax_fig.set_ylim(0, 1); ax_fig.axis('off')
    ax_fig.annotate('', xy=(x_r + 0.005, y_c),
                    xytext=(x_l - 0.005, y_c),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.4))
    if label:
        ax_fig.text((x_l + x_r) / 2, y_c + 0.04, label,
                    ha='center', va='bottom', fontsize=8, color=color)

rng = np.random.default_rng(7)

# ── Step 1: Dataset dots ──────────────────────────────────────────────────────
ax = axes[0]
xs = rng.uniform(0.15, 0.85, 14)
ys = 0.25 + 0.5 * xs + rng.normal(0, 0.08, 14)
ys = np.clip(ys, 0.12, 0.88)
ax.scatter(xs, ys, s=28, color=COL_DATA, zorder=3, edgecolors='black', linewidths=0.4)
ax.set_title('Step 1\nOriginal dataset\n(n observations)', fontsize=9, pad=4)
ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.82,
             boxstyle='round,pad=0.02', edgecolor='#AAAAAA',
             facecolor='#F8F8F8', linewidth=0.8, zorder=0))
ax.text(0.5, 0.02, 'n data points', ha='center', va='bottom',
        fontsize=8, color='#666666')

# ── Step 2: One bootstrap resample ───────────────────────────────────────────
ax = axes[1]
idx = rng.choice(len(xs), size=len(xs), replace=True)
xs2, ys2 = xs[idx], ys[idx]
# Jitter duplicates slightly for visibility
jitter_x = rng.normal(0, 0.018, len(xs2))
jitter_y = rng.normal(0, 0.018, len(ys2))
ax.scatter(xs2 + jitter_x, ys2 + jitter_y, s=28,
           color=COL_RESAMP, zorder=3, edgecolors='black', linewidths=0.4)
# Show a couple of "duplicate" circles
for i in range(3):
    ax.scatter(xs2[i] + 0.03, ys2[i], s=28, color=COL_RESAMP,
               zorder=2, edgecolors='black', linewidths=0.4, alpha=0.45)
ax.set_title('Step 2\nBootstrap resample\n(with replacement)', fontsize=9, pad=4)
ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.82,
             boxstyle='round,pad=0.02', edgecolor='#AAAAAA',
             facecolor='#F0F7FF', linewidth=0.8, zorder=0))
ax.text(0.5, 0.02, 'B = 1000 iterations', ha='center', va='bottom',
        fontsize=8, color='#555555')

# ── Step 3: Fitted line on resample ──────────────────────────────────────────
ax = axes[2]
# Simple linear fit to resample
m = np.polyfit(xs2, ys2, 1)
xp = np.array([0.1, 0.9])
ax.plot(xp, np.polyval(m, xp), color=COL_FIT, lw=1.8, zorder=3)
ax.scatter(xs2 + jitter_x, ys2 + jitter_y, s=18,
           color=COL_RESAMP, alpha=0.55, zorder=2, edgecolors='none')
ax.set_title('Step 3\nRefit model\non resample', fontsize=9, pad=4)
ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.82,
             boxstyle='round,pad=0.02', edgecolor='#AAAAAA',
             facecolor='#FFF8F2', linewidth=0.8, zorder=0))
ax.text(0.5, 0.02, 'Record ŷ at each x', ha='center', va='bottom',
        fontsize=8, color='#555555')

# ── Step 4: Stacked lines → interval band ────────────────────────────────────
ax = axes[3]
xp4 = np.linspace(0.1, 0.9, 80)
# Draw ~20 faint bootstrap lines
for seed in range(22):
    r = np.random.default_rng(seed + 100)
    idx_b = r.choice(len(xs), size=len(xs), replace=True)
    mb = np.polyfit(xs[idx_b], ys[idx_b], 1)
    ax.plot(xp4, np.polyval(mb, xp4), color=COL_FIT, lw=0.5, alpha=0.18, zorder=2)
# Show resulting 90% and 95% bands
center = np.polyval(np.array([0.5, 0.25]), xp4)
ax.fill_between(xp4, center - 0.16, center + 0.16,
                alpha=0.18, color=COL_DATA, zorder=1)
ax.fill_between(xp4, center - 0.11, center + 0.11,
                alpha=0.32, color=COL_DATA, zorder=1)
ax.plot(xp4, center, color='#1A1A2E', lw=1.4, zorder=3)
ax.text(0.78, center[-1] + 0.17, '95%', fontsize=7.5, color=COL_DATA, va='bottom')
ax.text(0.78, center[-1] + 0.12, '90%', fontsize=7.5, color=COL_DATA, va='bottom')
ax.set_title('Step 4\nStack B = 1000 fits\n→ empirical CI band', fontsize=9, pad=4)
ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.82,
             boxstyle='round,pad=0.02', edgecolor='#AAAAAA',
             facecolor='#F4F8FF', linewidth=0.8, zorder=0))
ax.text(0.5, 0.02, 'Percentile CI at each x', ha='center', va='bottom',
        fontsize=8, color='#555555')

# ── Inter-step arrows ─────────────────────────────────────────────────────────
for i in range(3):
    inter_arrow(fig, axes[i], axes[i + 1])

fig.suptitle('Bootstrap Resampling Pipeline — Phase 1 Procedure (B = 1000 iterations)',
             fontsize=11, y=1.01)

fig.text(0.5, -0.03,
         'Schematic — illustrative only. Data points and fitted lines are synthetic. '
         'Final plot produced by the bootstrap pipeline (Phase 1 sub-objective 1.3).\n'
         'Script: writeup/figures/fig_bootstrap_pipeline.py',
         ha='center', va='top', fontsize=7, style='italic', color='#777777')

plt.tight_layout(rect=[0, 0.05, 1, 0.93])
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
