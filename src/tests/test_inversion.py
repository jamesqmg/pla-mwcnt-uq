"""Tests for src/uq/inversion.py — TDD-first.

The inversion module converts divergence curves (PICP(n), MPIW(n)) into a
specimen-budget table: "minimum n to achieve PICP ≥ target" or "MPIW ≤ target".

Run: python -m pytest src/tests/test_inversion.py -v
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from src.uq.inversion import compute_inversion_table, _find_stable_crossing


# ── Fixtures ──────────────────────────────────────────────────────────────────

REAL_DATA = Path(__file__).parent.parent.parent / "results" / "subsampling_combined.json"


def _make_json(records: list[dict], tmp_path: Path) -> Path:
    """Write a minimal subsampling JSON file and return its path."""
    p = tmp_path / "subsampling_combined.json"
    p.write_text(json.dumps(records))
    return p


def _synthetic_records() -> list[dict]:
    """Synthetic n-ladder: PICP increases with n, MPIW decreases with n.

    Bootstrap PICP: 0.70, 0.78, 0.85, 0.90, 0.92 at n=5,10,15,20,25
    Bootstrap MPIW: 60,   50,   40,   30,   25   at n=5,10,15,20,25
    Bayesian only at n=5 and n=20.
    """
    ns = [5, 10, 15, 20, 25]
    picps = [0.70, 0.78, 0.85, 0.90, 0.92]
    mpiws = [60.0, 50.0, 40.0, 30.0, 25.0]
    records = []
    for i, n in enumerate(ns):
        r: dict = {
            "n": n,
            "picp_90_bootstrap": picps[i],
            "picp_90_bootstrap_std": 0.01,
            "mpiw_90_bootstrap": mpiws[i],
            "mpiw_90_bootstrap_std": 1.0,
        }
        if n in (5, 20):
            r["picp_90_bayesian"] = picps[i] + 0.02
            r["picp_90_bayesian_std"] = 0.01
            r["mpiw_90_bayesian"] = mpiws[i] + 2.0
            r["mpiw_90_bayesian_std"] = 1.0
        records.append(r)
    return records


# ── Unit tests: _find_stable_crossing ─────────────────────────────────────────

class TestFindStableCrossing:
    """Tests for the core interpolation helper."""

    def test_picp_crossing_ascending(self):
        """PICP crosses 0.90 between n=15 and n=20 (monotone ascending)."""
        ns = [5, 10, 15, 20, 25]
        vals = [0.70, 0.78, 0.85, 0.90, 0.92]
        n_star = _find_stable_crossing(ns, vals, target=0.90, direction="above")
        assert n_star is not None
        assert n_star == 20  # exact crossing at 20

    def test_mpiw_crossing_descending(self):
        """MPIW crosses 30 MPa between n=15 and n=20 (monotone descending)."""
        ns = [5, 10, 15, 20, 25]
        vals = [60.0, 50.0, 40.0, 30.0, 25.0]
        n_star = _find_stable_crossing(ns, vals, target=30.0, direction="below")
        assert n_star is not None
        assert n_star == 20  # exact crossing at 20

    def test_target_never_achieved_returns_none(self):
        """Target above every observed value returns None."""
        ns = [5, 10, 15]
        vals = [0.70, 0.75, 0.80]
        result = _find_stable_crossing(ns, vals, target=0.95, direction="above")
        assert result is None

    def test_always_satisfied_returns_min_n(self):
        """If target is already met at the smallest n, return that n."""
        ns = [10, 20, 30]
        vals = [0.92, 0.93, 0.94]
        result = _find_stable_crossing(ns, vals, target=0.90, direction="above")
        assert result is not None
        assert result <= 10  # already satisfied from the start

    def test_interpolated_crossing_between_points(self):
        """Crossing between n=10 and n=20 should give n between 10 and 20."""
        ns = [10, 20]
        vals = [0.80, 0.95]
        result = _find_stable_crossing(ns, vals, target=0.90, direction="above")
        assert result is not None
        assert 10 < result <= 20


# ── Integration tests: compute_inversion_table ────────────────────────────────

class TestComputeInversionTable:
    """Tests for the public inversion API."""

    def test_output_has_required_keys(self, tmp_path):
        """Output dict must have 'bootstrap' and 'bayesian' top-level keys."""
        path = _make_json(_synthetic_records(), tmp_path)
        result = compute_inversion_table(path)
        assert "bootstrap" in result
        assert "bayesian" in result

    def test_bootstrap_picp_keys_present(self, tmp_path):
        """Bootstrap sub-dict has picp_90 key with n_min and target entries."""
        path = _make_json(_synthetic_records(), tmp_path)
        result = compute_inversion_table(path)
        bs = result["bootstrap"]
        assert "picp_90" in bs
        for entry in bs["picp_90"]:
            assert "target" in entry
            assert "n_min" in entry

    def test_bootstrap_mpiw_keys_present(self, tmp_path):
        """Bootstrap sub-dict has mpiw_90 key."""
        path = _make_json(_synthetic_records(), tmp_path)
        result = compute_inversion_table(path)
        assert "mpiw_90" in result["bootstrap"]

    def test_n_min_positive_integer_or_none(self, tmp_path):
        """Every n_min value is either a positive integer or None."""
        path = _make_json(_synthetic_records(), tmp_path)
        result = compute_inversion_table(path)
        for method in ("bootstrap", "bayesian"):
            for metric_key in result[method]:
                for entry in result[method][metric_key]:
                    n = entry["n_min"]
                    assert n is None or (isinstance(n, (int, float)) and n > 0)

    def test_picp_n_min_increases_with_target(self, tmp_path):
        """Higher PICP targets require at least as large an n."""
        path = _make_json(_synthetic_records(), tmp_path)
        result = compute_inversion_table(path)
        bs_picp = result["bootstrap"]["picp_90"]
        # Sort entries by target; n_min should be non-decreasing
        sorted_entries = sorted(
            [e for e in bs_picp if e["n_min"] is not None],
            key=lambda e: e["target"]
        )
        for i in range(len(sorted_entries) - 1):
            assert sorted_entries[i]["n_min"] <= sorted_entries[i + 1]["n_min"]

    def test_mpiw_n_min_increases_with_tighter_target(self, tmp_path):
        """Tighter MPIW targets (lower threshold) require at least as large an n."""
        path = _make_json(_synthetic_records(), tmp_path)
        result = compute_inversion_table(path)
        bs_mpiw = result["bootstrap"]["mpiw_90"]
        sorted_entries = sorted(
            [e for e in bs_mpiw if e["n_min"] is not None],
            key=lambda e: e["target"],   # lower target = tighter
        )
        for i in range(len(sorted_entries) - 1):
            assert sorted_entries[i]["n_min"] >= sorted_entries[i + 1]["n_min"]

    def test_real_data_runs_without_error(self):
        """compute_inversion_table runs on real subsampling_combined.json."""
        pytest.importorskip("scipy")
        result = compute_inversion_table(REAL_DATA)
        assert "bootstrap" in result
        assert "bayesian" in result

    def test_real_data_bootstrap_picp90_n_min_positive(self):
        """Real Bootstrap PICP 90% crossings are positive n values or None."""
        pytest.importorskip("scipy")
        result = compute_inversion_table(REAL_DATA)
        for entry in result["bootstrap"]["picp_90"]:
            n = entry["n_min"]
            assert n is None or n > 0

    def test_bayesian_has_fewer_entries_than_bootstrap(self, tmp_path):
        """Bayesian uses fewer data points; its table still has the same metric keys."""
        path = _make_json(_synthetic_records(), tmp_path)
        result = compute_inversion_table(path)
        # Both methods have the same metric keys even if Bayesian has fewer n points
        assert set(result["bootstrap"].keys()) == set(result["bayesian"].keys())
