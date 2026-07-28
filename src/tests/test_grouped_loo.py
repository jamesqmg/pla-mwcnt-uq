"""
Tests for leave-one-GROUP-out (paper-level) LOO in bootstrap.py and bayesian.py.

Row-LOO on a corpus drawn from few papers overstates generalization because
same-paper twins leak into the training fold. Passing ``groups=`` switches the
LOO loop to leave-one-group-out. Byte-stability contract: ``groups=None``
(default) preserves the exact row-LOO behavior, and singleton groups
(one row per group) reproduce it numerically.

Run:  python -m pytest src/tests/test_grouped_loo.py -v
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.bootstrap import run_bootstrap_pipeline, _make_loo_folds
from src.uq.bayesian import run_bayesian_loo


class TestMakeLooFolds:
    def test_row_folds_default(self):
        folds = _make_loo_folds(4)
        assert [f.tolist() for f in folds] == [[0], [1], [2], [3]]

    def test_group_folds_partition(self):
        groups = np.array([0, 0, 1, 1, 1, 2])
        folds = _make_loo_folds(6, groups)
        assert [f.tolist() for f in folds] == [[0, 1], [2, 3, 4], [5]]

    def test_group_folds_cover_each_index_once(self):
        groups = np.array(["a", "a", "b", "c", "c"])
        folds = _make_loo_folds(5, groups)
        allidx = sorted(int(i) for f in folds for i in f)
        assert allidx == [0, 1, 2, 3, 4]

    def test_string_groups_supported(self):
        groups = np.array(["p1", "p2", "p1"])
        folds = _make_loo_folds(3, groups)
        assert sorted(len(f) for f in folds) == [1, 2]


class TestBootstrapGroupedLoo:
    def _data(self, n=10):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 5, n)
        y = 3 * x + 10 + rng.normal(0, 2, size=n)
        return x, y

    def test_singleton_groups_equal_row_loo(self):
        """One row per group == leave-one-row-out, numerically identical."""
        x, y = self._data(10)
        row = run_bootstrap_pipeline(x, y, B=300, seed=42, eval_mode="loo")
        grp = run_bootstrap_pipeline(
            x, y, B=300, seed=42, eval_mode="loo", groups=np.arange(10)
        )
        assert grp["picp_90"] == row["picp_90"]
        assert grp["mpiw_90"] == row["mpiw_90"]
        assert grp["nlpd"] == row["nlpd"]
        assert grp["ece"] == row["ece"]

    def test_grouped_loo_arrays_and_keys(self):
        x, y = self._data(10)
        groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
        res = run_bootstrap_pipeline(
            x, y, B=300, seed=42, eval_mode="loo", groups=groups
        )
        assert res["n_groups"] == 3
        assert res["eval_scheme"] == "leave-one-group-out"
        assert len(res["loo_y_true"]) == 10
        assert np.isfinite(res["nlpd"])
        assert 0.0 <= res["picp_90"] <= 1.0

    def test_row_scheme_label_when_no_groups(self):
        x, y = self._data(8)
        res = run_bootstrap_pipeline(x, y, B=200, seed=1, eval_mode="loo")
        assert res["eval_scheme"] == "leave-one-row-out"

    def test_groups_length_mismatch_raises(self):
        x, y = self._data(10)
        with np.testing.assert_raises(ValueError):
            run_bootstrap_pipeline(
                x, y, B=100, seed=1, eval_mode="loo", groups=np.arange(9)
            )

    def test_single_group_raises_clear_error(self):
        """One group => empty training fold; must fail with a helpful message."""
        x, y = self._data(6)
        with np.testing.assert_raises(ValueError):
            run_bootstrap_pipeline(
                x, y, B=100, seed=1, eval_mode="loo", groups=np.array(["p1"] * 6)
            )


class TestBayesianGroupedLoo:
    KW = dict(draws=150, tune=150, chains=2)

    def test_grouped_loo_runs_and_reports_groups(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 5, 8)
        y = 3 * x + 10 + rng.normal(0, 2, size=8)
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        res = run_bayesian_loo(x, y, seed=42, groups=groups, **self.KW)
        assert res["n_groups"] == 2
        assert res["eval_scheme"] == "leave-one-group-out"
        assert len(res["loo_y_true"]) == 8
        assert 0.0 <= res["picp_90"] <= 1.0
        assert res["mpiw_90"] > 0
