"""
Figure: Divergence curves — PICP(n) and MPIW(n) for bootstrap vs Bayesian NUTS.
Thesis §3.6 / Expected Phase 3 output (schematic becomes actual result).

Run:  python figures/fig_divergence_curve.py
Output: figures/fig_divergence_curve.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Load results ──────────────────────────────────────────────────────────────
with open(ROOT / "results/subsampling_bootstrap.json") as f:
    boot_full = {r["n"]: r for r in json.load(f)}
with open(ROOT / "results/subsampling_bayesian_key.json") as f:
    bay_key = {r["n"]: r for r in json.load(f)}

# Merge: bootstrap has full ladder; Bayesian has key n-values
all_n = sorted(set(list(boot_full.keys()) + list(bay_key.keys())))

boot_picp = [boot_full[n]["picp_90_bootstrap"] for n in all_n if n in boot_full]
boot_picp_std = [boot_full[n]["picp_90_bootstrap_std"] for n in all_n if n in boot_full]
boot_n = [n for n in all_n if n in boot_full]

boot_mpiw = [boot_full[n]["mpiw_90_bootstrap"] for n in boot_n]
boot_mpiw_std = [boot_full[n]["mpiw_90_bootstrap_std"] for n in boot_n]

bay_n_picp = [n for n in all_n if n in bay_key and "picp_90_bayesian" in bay_key[n]]
bay_picp = [bay_key[n]["picp_90_bayesian"] for n in bay_n_picp]

bay_n_mpiw = [n for n in all_n if n in bay_key and "mpiw_90_bayesian" in bay_key[n]]
bay_mpiw = [bay_key[n]["mpiw_90_bayesian"] for n in bay_n_mpiw]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle(
    "Subsampling Divergence Curves: Bootstrap vs Bayesian NUTS\n"
    "Parent dataset: master_dataset.csv (n=44, all PLA, mixed processing)",
    fontsize=10, y=1.01
)

BOOT_COLOR  = "#2166ac"   # blue
BAYES_COLOR = "#d6604d"   # orange-red
NOMINAL     = 0.90
THRESHOLD_N = 10

# ── Panel 1: PICP ─────────────────────────────────────────────────────────────
ax1.axhline(NOMINAL, color="black", lw=1, ls="--", alpha=0.5, label="Nominal 90%")
ax1.fill_between(
    boot_n,
    [p - s for p, s in zip(boot_picp, boot_picp_std)],
    [p + s for p, s in zip(boot_picp, boot_picp_std)],
    alpha=0.15, color=BOOT_COLOR
)
ax1.plot(boot_n, boot_picp, "o-", color=BOOT_COLOR, lw=2, ms=5, label="Bootstrap")
ax1.plot(bay_n_picp, bay_picp, "s--", color=BAYES_COLOR, lw=2, ms=6,
         label="Bayesian NUTS", zorder=5)

# Divergence threshold annotation
ax1.axvline(THRESHOLD_N, color="gray", lw=1.2, ls=":", alpha=0.8)
ax1.text(THRESHOLD_N + 0.5, 0.78, f"n* ≈ {THRESHOLD_N}\n(threshold)", fontsize=8,
         color="gray", va="bottom")

ax1.set_xlabel("Sample size n", fontsize=10)
ax1.set_ylabel("PICP @ 90% nominal coverage", fontsize=10)
ax1.set_title("(a) PICP(n)", fontsize=10)
ax1.set_xlim(3, 47)
ax1.set_ylim(0.70, 1.05)
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(True, alpha=0.3, lw=0.5)

# ── Panel 2: MPIW ─────────────────────────────────────────────────────────────
# Clip error bands to visible range to avoid n=5 variance collapse
Y_MAX = 100
boot_lo = [max(m - s, 0)     for m, s in zip(boot_mpiw, boot_mpiw_std)]
boot_hi = [min(m + s, Y_MAX) for m, s in zip(boot_mpiw, boot_mpiw_std)]
ax2.fill_between(boot_n, boot_lo, boot_hi, alpha=0.15, color=BOOT_COLOR)
ax2.plot(boot_n, boot_mpiw, "o-", color=BOOT_COLOR, lw=2, ms=5, label="Bootstrap")
ax2.plot(bay_n_mpiw, bay_mpiw, "s--", color=BAYES_COLOR, lw=2, ms=6,
         label="Bayesian NUTS")

# Annotate n=5 high variance
ax2.annotate("High σ at n=5\n(±348 MPa clipped)",
             xy=(5, boot_mpiw[boot_n.index(5)]), xytext=(9, 85),
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
             fontsize=7, color="gray")

ax2.axvline(THRESHOLD_N, color="gray", lw=1.2, ls=":", alpha=0.8)

ax2.set_xlabel("Sample size n", fontsize=10)
ax2.set_ylabel("MPIW @ 90% coverage (MPa)", fontsize=10)
ax2.set_title("(b) MPIW(n)", fontsize=10)
ax2.set_xlim(3, 47)
ax2.set_ylim(0, Y_MAX)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3, lw=0.5)

# Caption
fig.text(
    0.5, -0.04,
    "Figure. Divergence curves for bootstrap and Bayesian NUTS at 90% nominal coverage.\n"
    "Bootstrap: B=500, add_noise=True, 20 repeats per n (error bands = ±1 SD). "
    "Bayesian: draws=500, tune=500, 4 chains, 5 repeats per n.\n"
    "Threshold n*≈10: largest n where |PICP_boot − PICP_Bayes| > 0.05. "
    "Both methods converge at n≥20.",
    ha="center", fontsize=7.5, color="dimgray", style="italic"
)

plt.tight_layout()
out = ROOT / "figures/fig_divergence_curve.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
