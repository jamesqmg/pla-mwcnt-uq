"""
F2 — UQ Method Workflow Diagram (Ch1 after "Two UQ methods are applicable")
Two-column flow: Bootstrap (left) vs Bayesian MCMC (right).
No data dependencies — fully schematic.
Output: fig_uq_workflow.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COL_BS   = '#E07B39'
COL_BAY  = '#2176AE'
COL_BOX  = '#F5F5F5'
COL_TEXT = '#1A1A2E'

fig, ax = plt.subplots(figsize=(10, 7.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# ── Box drawing helper ────────────────────────────────────────────────────────
def draw_box(ax, cx, cy, w, h, text, color, fontsize=9.5, text_color='white'):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle='round,pad=0.15',
                          facecolor=color, edgecolor='black',
                          linewidth=0.9, zorder=3)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, color=text_color,
            fontweight='bold', zorder=4, wrap=True,
            multialignment='center')

def draw_arrow(ax, x, y_top, y_bot, color):
    ax.annotate('', xy=(x, y_bot + 0.02),
                xytext=(x, y_top - 0.02),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3),
                zorder=5)

# ── Column centers ────────────────────────────────────────────────────────────
cx_bs  = 2.5
cx_bay = 7.5

# ── Column headers ────────────────────────────────────────────────────────────
ax.text(cx_bs,  9.65, 'Bootstrap Resampling\n(Phase 1)', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COL_BS)
ax.text(cx_bay, 9.65, 'Bayesian MCMC (NUTS)\n(Phase 2)', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COL_BAY)

# ── Shared top node ───────────────────────────────────────────────────────────
draw_box(ax, 5.0, 8.6, 3.8, 0.72,
         'Observed data\n(PLA/MWCNT tensile strength, n observations)',
         '#444444', fontsize=9)

# Split arrows
for cx, col in [(cx_bs, COL_BS), (cx_bay, COL_BAY)]:
    ax.annotate('', xy=(cx, 8.10),
                xytext=(5.0, 8.24),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.1), zorder=5)

# ── Bootstrap steps ───────────────────────────────────────────────────────────
bs_steps = [
    (7.9, 'Resample with replacement\nB = 1000 iterations'),
    (6.6, 'Refit regression model\non each resample'),
    (5.3, 'Empirical distribution\nof predictions'),
    (4.0, 'Percentile confidence interval\n(90% / 95%)'),
    (2.7, 'PICP, MPIW, CRPS\n(Uncertainty Toolbox)'),
]
for y, txt in bs_steps:
    draw_box(ax, cx_bs, y, 3.6, 0.82, txt, COL_BS)

for i in range(len(bs_steps) - 1):
    draw_arrow(ax, cx_bs, bs_steps[i][0] - 0.41, bs_steps[i+1][0] + 0.41, COL_BS)

# ── Bayesian steps ────────────────────────────────────────────────────────────
bay_steps = [
    (7.9, 'Prior specification\nβ ~ N(0,1), σ ~ HalfN(1)'),
    (6.6, 'Prior predictive check\n(plausible TS range?)'),
    (5.3, 'NUTS sampling\n4 chains × 2000 draws'),
    (4.0, 'Posterior predictive interval\n(90% / 95% credible)'),
    (2.7, 'PICP, MPIW, CRPS\n(Uncertainty Toolbox)'),
]
for y, txt in bay_steps:
    draw_box(ax, cx_bay, y, 3.6, 0.82, txt, COL_BAY)

for i in range(len(bay_steps) - 1):
    draw_arrow(ax, cx_bay, bay_steps[i][0] - 0.41, bay_steps[i+1][0] + 0.41, COL_BAY)

# ── Convergence check note on Bayesian ───────────────────────────────────────
ax.text(cx_bay + 2.0, 5.3, 'R̂ < 1.01\nESS > 400',
        ha='center', va='center', fontsize=8, color='#217A3A',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFFEA',
                  edgecolor='#217A3A', lw=0.8))

# ── Bottom shared output ──────────────────────────────────────────────────────
for cx, col in [(cx_bs, COL_BS), (cx_bay, COL_BAY)]:
    ax.annotate('', xy=(5.0, 1.72),
                xytext=(cx, 2.29),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.1), zorder=5)

draw_box(ax, 5.0, 1.3, 4.8, 0.72,
         'Divergence curve: PICP(n) & MPIW(n)\nvs. sample size n — Phase 3 deliverable',
         '#1A1A2E', fontsize=9)

# ── Central contrast label ────────────────────────────────────────────────────
ax.text(5.0, 6.25,
        '← No prior, requires large n │ Prior regularizes, stable at small n →',
        ha='center', va='center', fontsize=8.5, color='#555555', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFF0',
                  edgecolor='#CCCCAA', lw=0.7))

# ── Vertical divider ──────────────────────────────────────────────────────────
ax.plot([5.0, 5.0], [1.0, 9.2], color='#DDDDDD', lw=1.0, ls='--', zorder=1)

ax.set_title('Bootstrap Resampling vs. Bayesian MCMC — Parallel Pipeline Overview',
             fontsize=11, pad=10, y=1.01)

fig.text(0.5, -0.01,
         'Script: writeup/figures/fig_uq_workflow.py',
         ha='center', va='top', fontsize=7, style='italic', color='#777777')

plt.tight_layout()
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
