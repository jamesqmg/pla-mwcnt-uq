"""Dataset loading and filtering for PLA/MWCNT/FDM tensile strength analysis."""
import re
import pandas as pd
import numpy as np
from pathlib import Path

_COL_MAP = {
    "Polymer":                      "polymer",
    "Processing Method":            "process",
    "CNT Weight Fraction (%)":      "cnt_wt_pct",
    "Tensile Strength Output (MPa)":"tensile_mpa",
    "Testing Standard":             "testing_std",
    "source_paper":                 "source_paper",
    "quality_tier":                 "quality_tier",
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

    for col in ("cnt_wt_pct", "tensile_mpa"):
        if col in df.columns:
            df[col] = parse_numeric_column(df[col])
    if "tensile_mpa" in df.columns:
        df.loc[df["tensile_mpa"] < 0, "tensile_mpa"] = np.nan

    for col in ("polymer", "process", "testing_std", "source_paper", "quality_tier"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def filter_pla_fdm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the three-way thesis filter: PLA polymer + FDM/FFF process + ASTM D638 or ISO 527.
    Also drops rows with missing CNT wt% or tensile strength values.
    """
    mask_pla     = df["polymer"].str.contains("PLA", case=False, na=False)
    mask_fdm     = df["process"].str.contains(_FDM_PATTERN, flags=re.I, na=False)
    mask_std     = df["testing_std"].str.contains(r"D638|527", na=False)
    mask_numeric = df["cnt_wt_pct"].notna() & df["tensile_mpa"].notna()

    return df[mask_pla & mask_fdm & mask_std & mask_numeric].copy().reset_index(drop=True)
