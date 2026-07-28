"""Tier-3 FE-RVE Monte Carlo sweep across the wt% levels used in Fig 2/3/4
(fe_rve_figure_spec.md).

Runs the full geometry -> mesh -> solve -> ensemble pipeline (rve_geometry.py
-> rve_mesh.py -> rve_solve.py -> rve_ensemble.py) at each MWCNT loading
level (0.5, 1, 2, 3, 4, 5 wt%), converting wt% -> V_f via the same density
conversion Tier 2 uses (micromechanics.weight_to_volume_fraction, default
densities), and writes the resulting effective-modulus ensembles to
results/rve_sweep.json.

Material inputs, aspect ratio, and all caveats: see
Thesis2/fe-rve-asset-inventory.md sec 6. This is still demo/provisional
scale (EFFECTIVE_ASPECT_RATIO=20, not the real 2609), not yet a defensible
publication result -- that doc tracks what's still needed (finer mesh
resolution, more realizations, Halpin-Tsai cross-check, real waviness
calibration).

Run:  python -m src.uq.run_rve_sweep
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.uq.micromechanics import weight_to_volume_fraction
from src.uq.rve_ensemble import RVEEnsembleConfig, run_mc_ensemble

ROOT = Path(__file__).parents[2]
OUT_JSON = ROOT / "results" / "rve_sweep.json"

WT_PCTS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]  # fe_rve_figure_spec.md Fig 2 loading levels
ASPECT_RATIO = 20.0                        # provisional EFFECTIVE_ASPECT_RATIO -- see inventory doc
DOMAIN_SIZE = 20.0                         # bumped from 15.0 -- 15.0 gave n_target=1 for BOTH
                                            # 0.5% and 1% wt (collision, see inventory doc); 20.0
                                            # gives distinct integer inclusion counts (2/3/6/9/15)
                                            # across all 5 wt% levels
N_ELEMENTS_PER_SIDE = 18                    # measured: 21/side gave 2.26x the elements of the
                                            # original 16/side run but ~7.5x the solve time (sparse
                                            # direct solve scales worse than linearly) -- 18/side is
                                            # a deliberate balance, not full publication resolution
E_MATRIX, NU_MATRIX = 3.023, 0.35          # da Silva et al. 2025 (same as Tier 2)
E_INCLUSION, NU_INCLUSION = 1000.0, 0.2    # 1.0 TPa CNT; nu is a placeholder, not a dataset value
MACRO_STRAIN = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
N_REALIZATIONS = 40                         # bumped from 15 (2026-07-09) to narrow the 0.5%/1% wt%
                                            # statistical-power gap (see fe-rve-asset-inventory.md
                                            # sec 6); measured ~20s/realization -> ~80 min full sweep
                                            # (6 levels, 2026-07-09 wt%=4 addition).
                                            # Note: full resolution (means >=2 combined-SE apart)
                                            # would need ~450 realizations/level given the true effect
                                            # size (Cohen's d~0.13) -- not attempted, impractical at
                                            # demo scale; this bump narrows, does not resolve, the gap.
BASE_SEED = 2026


def main() -> dict:
    ensemble_config = RVEEnsembleConfig(
        aspect_ratio=ASPECT_RATIO, domain_size=DOMAIN_SIZE,
        n_elements_per_side=N_ELEMENTS_PER_SIDE,
        matrix_props=(E_MATRIX, NU_MATRIX), inclusion_props=(E_INCLUSION, NU_INCLUSION),
        macro_strain=MACRO_STRAIN,
    )
    results = {}
    for wt in WT_PCTS:
        vf = float(weight_to_volume_fraction(wt))
        t0 = time.time()
        moduli = run_mc_ensemble(
            target_vf=vf, config=ensemble_config, n_realizations=N_REALIZATIONS,
            base_seed=BASE_SEED + int(wt * 100),
        )
        elapsed = time.time() - t0
        results[str(wt)] = {
            "wt_pct": wt,
            "vf": vf,
            "e_eff_mean": float(moduli.mean()),
            "e_eff_std": float(moduli.std()),
            "e_eff_min": float(moduli.min()),
            "e_eff_max": float(moduli.max()),
            "e_eff_all": moduli.tolist(),
            "n_realizations": N_REALIZATIONS,
            "elapsed_s": elapsed,
        }
        print(f"wt%={wt}: Vf={vf:.4f}, E_eff={moduli.mean():.3f}+/-{moduli.std():.3f} GPa "
              f"({elapsed:.1f}s for {N_REALIZATIONS} realizations)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "meta": {
            "aspect_ratio_effective": ASPECT_RATIO,
            "domain_size": DOMAIN_SIZE,
            "n_elements_per_side": N_ELEMENTS_PER_SIDE,
            "matrix_props_E_nu": [E_MATRIX, NU_MATRIX],
            "inclusion_props_E_nu": [E_INCLUSION, NU_INCLUSION],
            "note": "DEMO/PROVISIONAL SCALE -- see Thesis2/fe-rve-asset-inventory.md sec 6",
        },
        "results": results,
    }, indent=2))
    print(f"Wrote {OUT_JSON}")
    return results


if __name__ == "__main__":
    main()
