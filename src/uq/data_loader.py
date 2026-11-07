"""Dataset loading and filtering for PLA/MWCNT/FDM tensile strength analysis."""
import re
import pandas as pd
import numpy as np
from pathlib import Path

_COL_MAP = {
    "Polymer":                        "polymer",
    "Processing Method":              "process",
    "CNT Weight Fraction (%)":        "cnt_wt_pct",
    "Tensile Strength Output (MPa)":  "tensile_mpa",
    "Matrix Tensile Strength (MPa)":  "matrix_tensile_mpa",
    "Testing Standard":               "testing_std",
    "source_paper":                   "source_paper",
    "quality_tier":                   "quality_tier",
}

_FDM_PATTERN  = r"fdm|fused.?deposition|fused.?filament|fff"
_NUMERIC_LEAD = re.compile(r"^-?\d+(?:\.\d+)?")


def parse_numeric_column(series: pd.Series) -> pd.Series:
    """Extract the leading float from a mixed-text series. Returns NaN for non-numeric entries."""
    def _parse(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        m = _NUMERIC_LEAD.match(s)
        if m:
            return float(m.group())
        return np.nan
    return series.map(_parse)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a master extraction CSV and return a DataFrame with standardised column names."""
    df = pd.read_csv(csv_path)
    rename = {k: v for k, v in _COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    for col in ("cnt_wt_pct", "tensile_mpa", "matrix_tensile_mpa"):
        if col in df.columns:
            df[col] = parse_numeric_column(df[col])
    if "tensile_mpa" in df.columns:
        df.loc[df["tensile_mpa"] < 0, "tensile_mpa"] = np.nan

    for col in ("polymer", "process", "testing_std", "source_paper", "quality_tier"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def _affirms(series: pd.Series, pattern: str) -> pd.Series:
    """True where the text names the pattern and does not negate it.

    Extraction notes write exclusions as 'NOT FDM' or 'NOT ASTM D638'; a plain substring
    match counts those as hits (that let Velu 2025's extruded filaments into the strict
    set until 2026-11-07).
    """
    hit = series.str.contains(pattern, flags=re.I, na=False)
    negated = series.str.contains(rf"\bnot\b[^;,()]*?(?:{pattern})", flags=re.I, na=False)
    return hit & ~negated


def filter_pla_fdm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the three-way thesis filter: PLA polymer + FDM/FFF process + ASTM D638 or ISO 527.
    Also drops rows with missing CNT wt% or tensile strength values.
    """
    mask_pla     = df["polymer"].str.contains("PLA", case=False, na=False)
    mask_fdm     = _affirms(df["process"], _FDM_PATTERN)
    mask_std     = _affirms(df["testing_std"], r"D638|527")
    mask_numeric = df["cnt_wt_pct"].notna() & df["tensile_mpa"].notna()

    return df[mask_pla & mask_fdm & mask_std & mask_numeric].copy().reset_index(drop=True)
