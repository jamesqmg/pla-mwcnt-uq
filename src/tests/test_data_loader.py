"""
Tests for src/uq/data_loader.py  — RED phase (all should fail before implementation)

Run:  python -m pytest src/tests/test_data_loader.py -v
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Adjust import path so tests run from project root
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.data_loader import load_dataset, filter_pla_fdm, parse_numeric_column

DATA_DIR = Path(__file__).parents[2] / "data" / "extraction"
STRICT_CSV = DATA_DIR / "master_dataset_strict.csv"


class TestParseNumericColumn:
    """parse_numeric_column strips text flags and returns float or NaN."""

    def test_clean_float_passthrough(self):
        s = pd.Series(["44.2", "55.48", "62.48"])
        result = parse_numeric_column(s)
        np.testing.assert_allclose(result.values, [44.2, 55.48, 62.48])

    def test_graphical_suffix_extracted(self):
        s = pd.Series(["35.0 [GRAPHICAL]", "45.0 [GRAPHICAL]"])
        result = parse_numeric_column(s)
        np.testing.assert_allclose(result.values, [35.0, 45.0])

    def test_nr_graphical_only_becomes_nan(self):
        s = pd.Series(["NR [GRAPHICAL ONLY — estimated ~30 MPa]"])
        result = parse_numeric_column(s)
        assert pd.isna(result.iloc[0])

    def test_plain_nr_becomes_nan(self):
        s = pd.Series(["NR", "N/R"])
        result = parse_numeric_column(s)
        assert result.isna().all()

    def test_inferred_midpoint_extracted(self):
        s = pd.Series(["11.5 [INFERRED midpoint 8-15nm]"])
        result = parse_numeric_column(s)
        assert result.iloc[0] == pytest.approx(11.5)

    def test_mixed_series(self):
        s = pd.Series(["44.2", "35.0 [GRAPHICAL]", "NR", "62.48"])
        result = parse_numeric_column(s)
        assert result.iloc[0] == pytest.approx(44.2)
        assert result.iloc[1] == pytest.approx(35.0)
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == pytest.approx(62.48)


class TestLoadDataset:
    """load_dataset returns a DataFrame with standardised column names."""

    def test_returns_dataframe(self):
        df = load_dataset(STRICT_CSV)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self):
        df = load_dataset(STRICT_CSV)
        for col in ("polymer", "process", "cnt_wt_pct", "tensile_mpa",
                    "testing_std", "source_paper", "quality_tier"):
            assert col in df.columns, f"Missing column: {col}"

    def test_cnt_wt_pct_is_numeric(self):
        df = load_dataset(STRICT_CSV)
        assert pd.api.types.is_float_dtype(df["cnt_wt_pct"])

    def test_tensile_mpa_is_numeric_or_nan(self):
        df = load_dataset(STRICT_CSV)
        assert pd.api.types.is_float_dtype(df["tensile_mpa"])

    def test_row_count_matches_csv(self):
        raw = pd.read_csv(STRICT_CSV)
        df = load_dataset(STRICT_CSV)
        assert len(df) == len(raw)

    def test_matrix_tensile_mpa_is_numeric(self):
        """Matrix Tensile Strength annotations ([GRAPHICAL], [INFERRED]) stripped to float."""
        df = load_dataset(STRICT_CSV)
        assert "matrix_tensile_mpa" in df.columns, "matrix_tensile_mpa column missing"
        assert pd.api.types.is_float_dtype(df["matrix_tensile_mpa"]), (
            "matrix_tensile_mpa must be float, not mixed-text"
        )
        non_nan = df["matrix_tensile_mpa"].dropna()
        assert (non_nan > 0).all(), "All non-NaN matrix_tensile_mpa values must be positive"


class TestFilterPlaFdm:
    """filter_pla_fdm returns only PLA + FDM/FFF + D638/ISO527 rows with clean x,y."""

    def test_returns_dataframe(self):
        df = load_dataset(STRICT_CSV)
        result = filter_pla_fdm(df)
        assert isinstance(result, pd.DataFrame)

    def test_all_rows_are_pla(self):
        df = load_dataset(STRICT_CSV)
        result = filter_pla_fdm(df)
        assert result["polymer"].str.contains("PLA", case=False).all()

    def test_all_rows_are_fdm_process(self):
        df = load_dataset(STRICT_CSV)
        result = filter_pla_fdm(df)
        fdm_pattern = r"fdm|fused deposition|fused filament|fff"
        assert result["process"].str.lower().str.contains(fdm_pattern).all()

    def test_all_rows_have_d638_or_iso527_standard(self):
        df = load_dataset(STRICT_CSV)
        result = filter_pla_fdm(df)
        assert result["testing_std"].str.contains(r"D638|527", na=False).all()

    def test_no_nan_in_x_or_y(self):
        df = load_dataset(STRICT_CSV)
        result = filter_pla_fdm(df)
        assert not result["cnt_wt_pct"].isna().any()
        assert not result["tensile_mpa"].isna().any()

    def test_minimum_n_is_fifteen(self):
        """Phase 1 requires at least 15 clean x,y pairs for meaningful bootstrap."""
        df = load_dataset(STRICT_CSV)
        result = filter_pla_fdm(df)
        assert len(result) >= 15, f"Only {len(result)} clean pairs — need ≥ 15"

    def test_cnt_wt_pct_range_is_physical(self):
        df = load_dataset(STRICT_CSV)
        result = filter_pla_fdm(df)
        assert (result["cnt_wt_pct"] >= 0).all()
        assert (result["cnt_wt_pct"] <= 20).all()  # no physical composite above 20%

    def test_tensile_mpa_range_is_physical(self):
        df = load_dataset(STRICT_CSV)
        result = filter_pla_fdm(df)
        assert (result["tensile_mpa"] > 0).all()
        assert (result["tensile_mpa"] < 500).all()

    def test_negated_process_and_standard_are_rejected(self):
        """'NOT FDM' and 'NOT ASTM D638' contain the keywords but deny them (Velu 2025)."""
        df = pd.DataFrame({
            "polymer": ["PLA", "PLA", "PLA"],
            "process": ["FFF", "Filament (Melt Mixing + Single-Screw Extrusion — NOT FDM)", "FFF"],
            "testing_std": ["ASTM D638", "ASTM D638", "Non-standard (NOT ASTM D638)"],
            "cnt_wt_pct": [1.0, 1.0, 1.0],
            "tensile_mpa": [50.0, 50.0, 50.0],
        })
        result = filter_pla_fdm(df)
        assert len(result) == 1
        assert result["process"].iloc[0] == "FFF"

    def test_strict_set_is_hardani_and_vidakis(self):
        """After the negation fix the strict set is the two FDM + D638 studies, n = 21."""
        result = filter_pla_fdm(load_dataset(STRICT_CSV))
        assert len(result) == 21
        assert set(result["source_paper"]) == {"hardani_2024_pen", "vidakis_2021_taylorandfrancis"}
