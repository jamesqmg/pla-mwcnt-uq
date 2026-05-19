"""
src/uq/run_elevation_metrics.py — Compute ECE + NLPD and patch phase1_phase2_loo.json

Adds 'nlpd' and 'ece' keys to both method entries in results/phase1_phase2_loo.json.

Bootstrap:  LOO NLPD (Gaussian approximation) + LOO ECE across 6 coverage levels.
Bayesian:   PSIS-LOO NLPD via az.loo() + in-sample ECE from posterior predictive.

Run:  python src/uq/run_elevation_metrics.py
      (from the Thesis 1 root directory)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent

# ── data ──────────────────────────────────────────────────────────────────────

from src.uq.data_loader import load_dataset, filter_pla_fdm  # noqa: E402

csv_path = ROOT / "data" / "extraction" / "master_dataset_strict.csv"
df = filter_pla_fdm(load_dataset(csv_path))
x = df["cnt_wt_pct"].values.astype(float)
y = df["tensile_mpa"].values.astype(float)
n = len(x)
print(f"Dataset loaded: n={n} (PLA+FDM+D638 filtered)")

# ── bootstrap LOO with NLPD + ECE ────────────────────────────────────────────

from src.uq.bootstrap import run_bootstrap_pipeline  # noqa: E402

print("\nRunning Bootstrap LOO (B=1000) …", flush=True)
t0 = time.perf_counter()
boot_loo = run_bootstrap_pipeline(x, y, B=1000, alpha=0.10, seed=42, eval_mode="loo")
t1 = time.perf_counter()
print(f"  Done in {t1-t0:.1f}s")
print(f"  PICP@90% = {boot_loo['picp_90']:.4f}")
print(f"  MPIW@90% = {boot_loo['mpiw_90']:.3f} MPa")
print(f"  CRPS     = {boot_loo['crps']:.4f}")
print(f"  NLPD     = {boot_loo['nlpd']:.4f}")
print(f"  ECE      = {boot_loo['ece']:.4f}")
print(f"  ECE by level: {boot_loo['ece_by_coverage']}")

# ── bayesian full run with NLPD + ECE ─────────────────────────────────────────

from src.uq.bayesian import run_bayesian_pipeline  # noqa: E402

print("\nRunning Bayesian NUTS (4 chains × 1000 draws) …", flush=True)
t0 = time.perf_counter()
bayes = run_bayesian_pipeline(x, y, draws=1000, tune=1000, chains=4, seed=42)
t1 = time.perf_counter()
print(f"  Done in {t1-t0:.1f}s")
conv = bayes["convergence"]
print(f"  R-hat max = {conv['rhat_max']:.4f}  (gate < 1.01: {'PASS' if conv['rhat_max'] < 1.01 else 'FAIL'})")
print(f"  ESS min   = {conv['ess_min']:.0f}  (gate > 400: {'PASS' if conv['ess_min'] > 400 else 'FAIL'})")
print(f"  PICP@90%  = {bayes['picp_90']:.4f}")
print(f"  MPIW@90%  = {bayes['mpiw_90']:.3f} MPa")
print(f"  CRPS      = {bayes['crps']:.4f}")
print(f"  NLPD      = {bayes['nlpd']:.4f}")
print(f"  ECE       = {bayes['ece']:.4f}  [{bayes['ece_note']}]")

# ── patch results/phase1_phase2_loo.json ──────────────────────────────────────

loo_json_path = ROOT / "results" / "phase1_phase2_loo.json"
with open(loo_json_path) as f:
    loo_data = json.load(f)

loo_data["phase1_bootstrap_loo"]["nlpd"]             = boot_loo["nlpd"]
loo_data["phase1_bootstrap_loo"]["ece"]              = boot_loo["ece"]
loo_data["phase1_bootstrap_loo"]["ece_by_coverage"]  = boot_loo["ece_by_coverage"]

loo_data["phase2_bayesian_loo"]["nlpd"]              = bayes["nlpd"]
loo_data["phase2_bayesian_loo"]["ece"]               = bayes["ece"]
loo_data["phase2_bayesian_loo"]["ece_note"]          = bayes["ece_note"]

with open(loo_json_path, "w") as f:
    json.dump(loo_data, f, indent=2)

print(f"\nPatched {loo_json_path}")
print("\n=== Summary ===")
print(f"{'Metric':<25} {'Bootstrap':>12} {'Bayesian':>12}")
print("-" * 50)
print(f"{'PICP@90% (LOO)':<25} {boot_loo['picp_90']:>12.3f} {loo_data['phase2_bayesian_loo']['picp_90']:>12.3f}")
print(f"{'MPIW@90% (MPa)':<25} {boot_loo['mpiw_90']:>12.2f} {loo_data['phase2_bayesian_loo']['mpiw_90']:>12.2f}")
print(f"{'CRPS':<25} {boot_loo['crps']:>12.4f} {loo_data['phase2_bayesian_loo'].get('crps', float('nan')):>12.4f}")
print(f"{'NLPD (LOO)':<25} {boot_loo['nlpd']:>12.4f} {bayes['nlpd']:>12.4f}")
print(f"{'ECE':<25} {boot_loo['ece']:>12.4f} {bayes['ece']:>12.4f}")
print()
