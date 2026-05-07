"""
F6 — Bayesian MCMC Pipeline Schematic (Ch3 §3.2 after sub-objective 2.3)
Three horizontal panels: prior predictive check / NUTS trace / posterior predictive interval.
All data are synthetic — purely illustrative.
Output: fig_bayesian_pipeline.png
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rng = np.random.default_rng(42)

COL_PRIOR  = '#5BA4CF'
COL_POST   = '#2176AE'
COL_TRACE  = ['#E07B39', '#2176AE', '#3ABF7A', '#C03A8A']   # 4 chains
COL_DATA   = '#333333'

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8),
                          gridspec_kw={'wspace': 0.32})

# ── Panel (a): Prior predictive check ────────────────────────────────────────
ax = axes[0]
# Prior predictive: sample β ~ N(0,1), σ ~ HalfN(1) → y_prior
beta0_s = rng.normal(52, 10, 400)       # intercept
beta1_s = rng.normal(4.5, 3, 400)
sigma_s = np.abs(rng.normal(0, 10, 400))
x_ref   = 2.5                           # representative CNT wt%
y_prior = beta0_s + beta1_s * x_ref + rng.normal(0, sigma_s)

# KDE-style histogram
ax.hist(y_prior, bins=35, color=COL_PRIOR, alpha=0.65,
        edgecolor='white', linewidth=0.3, density=True, zorder=2)
ax.axvline(25,   color='#C03A3A', lw=1.0, ls='--', zorder=3)
ax.axvline(200,  color='#C03A3A', lw=1.0, ls='--', zorder=3)
ax.text(25,  ax.get_ylim()[1] * 0.5, '25 MPa', rotation=90,
        va='center', ha='right', fontsize=7.5, color='#C03A3A')
ax.text(200, ax.get_ylim()[1] * 0.5, '200 MPa', rotation=90,
        va='center', ha='right', fontsize=7.5, color='#C03A3A')

ax.set_xlabel('Tensile Strength [MPa]', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('(a) Prior predictive check\nConforms to plausible PLA/MWCNT range',
             fontsize=9, pad=5)
ax.grid(True, linestyle=':', linewidth=0.5, color='#BBBBBB')
ax.spines[['top', 'right', 'bottom', 'left']].set_color('black')
ax.tick_params(labelsize=8)
ax.annotate('Plausible\nrange', xy=(110, ax.get_ylim()[1] * 0.75),
            fontsize=7.5, color='#555555', ha='center')

# ── Panel (b): NUTS trace (schematic) ────────────────────────────────────────
ax = axes[1]
n_draw  = 200
n_tune  = 40    # show a short tuning section
draws   = np.arange(n_draw)

# β₁ traces — 4 chains, stationary Gaussian noise around true value
for c_i, col in enumerate(COL_TRACE):
    noise = rng.normal(0, 0.18, n_draw)
    # Small warm-up trend in the first n_tune draws
    warmup = np.linspace(0.6 if c_i % 2 == 0 else -0.6, 0, n_tune)
    trace = np.concatenate([warmup + noise[:n_tune], noise[n_tune:]])
    trace += 4.5   # center on β₁ ≈ 4.5
    ax.plot(draws, trace, color=col, lw=0.7, alpha=0.85, zorder=2)

ax.axvline(n_tune, color='#888888', lw=0.9, ls='--', zorder=3)
ax.text(n_tune + 2, 4.5 + 0.5, 'Post-tuning\n(sampling)', fontsize=7.5, color='#555555')
ax.text(2, 4.5 + 0.5, 'Tuning', fontsize=7.5, color='#888888')

# Convergence annotations
ax.text(0.97, 0.96, 'R̂ < 1.01 ✓', transform=ax.transAxes,
        ha='right', va='top', fontsize=8, color='#217A3A',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#EAFFEA', edgecolor='#217A3A', lw=0.8))
ax.text(0.97, 0.82, 'ESS > 400 ✓', transform=ax.transAxes,
        ha='right', va='top', fontsize=8, color='#217A3A',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#EAFFEA', edgecolor='#217A3A', lw=0.8))

ax.set_xlabel('Draw index', fontsize=10)
ax.set_ylabel('β₁ (slope) value', fontsize=10)
ax.set_title('(b) NUTS trace — 4 chains, 2000 draws\nGood mixing ≈ stationary, overlapping chains',
             fontsize=9, pad=5)
ax.grid(True, linestyle=':', linewidth=0.5, color='#BBBBBB')
ax.spines[['top', 'right', 'bottom', 'left']].set_color('black')
ax.tick_params(labelsize=8)

chain_handles = [Line2D([0], [0], color=c, lw=1.2, label=f'Chain {i+1}')
                 for i, c in enumerate(COL_TRACE)]
ax.legend(handles=chain_handles, fontsize=7.5, loc='lower right',
          framealpha=0.85, edgecolor='black')

# ── Panel (c): Posterior predictive interval ─────────────────────────────────
ax = axes[2]
x_data = rng.uniform(0.5, 5.0, 15)
y_data = 52 + 4.5 * x_data + rng.normal(0, 9, 15)
x_line = np.linspace(0.2, 5.5, 200)
y_fit  = 52 + 4.5 * x_line

# 90% and 95% posterior predictive bands
sigma_post = 9.0
ax.fill_between(x_line, y_fit - 1.960 * sigma_post, y_fit + 1.960 * sigma_post,
                alpha=0.18, color=COL_POST, label='95% Credible Interval')
ax.fill_between(x_line, y_fit - 1.645 * sigma_post, y_fit + 1.645 * sigma_post,
                alpha=0.32, color=COL_POST, label='90% Credible Interval')
ax.plot(x_line, y_fit, color='#1A1A2E', lw=1.5, label='Posterior mean')
ax.scatter(x_data, y_data, s=24, color=COL_DATA, zorder=5,
           edgecolors='none', label='Schematic data')

ax.set_xlabel('CNT Weight Fraction [wt%]', fontsize=10)
ax.set_ylabel('Tensile Strength [MPa]', fontsize=10)
ax.set_title('(c) Posterior predictive interval\n95% credible band on held-out subset',
             fontsize=9, pad=5)
ax.grid(True, linestyle=':', linewidth=0.5, color='#BBBBBB')
ax.spines[['top', 'right', 'bottom', 'left']].set_color('black')
ax.tick_params(labelsize=8)
ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9, edgecolor='black')

# ── Overall title ─────────────────────────────────────────────────────────────
fig.suptitle('Bayesian MCMC Pipeline — Phase 2 Procedure (PyMC v5+, NUTS Sampler)',
             fontsize=11, y=1.02)

fig.text(0.5, -0.04,
         'All data and traces are synthetic — schematic illustration only. '
         'Final convergence diagnostics and posterior are produced by the Phase 2 MCMC pipeline.\n'
         'Script: writeup/figures/fig_bayesian_pipeline.py',
         ha='center', va='top', fontsize=7, style='italic', color='#777777')

plt.tight_layout()
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
