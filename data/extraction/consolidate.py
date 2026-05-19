"""
Consolidates all per-paper extraction CSVs into master datasets.

Outputs:
  data/extraction/master_dataset.csv       — all rows with quality tier labels
  data/extraction/master_dataset_strict.csv — HIGH tier only (PLA + FDM/FFF + ASTM D638)

Quality tiers:
  HIGH    — PLA + FDM/FFF + ASTM D638
  MEDIUM  — PLA + FDM/FFF + other/unspecified standard
  LOW     — PLA + non-FDM processing
  FAIL    — not PLA, or no tensile data, or theoretical
"""

import pandas as pd
from pathlib import Path
import sys

PER_PAPER_DIR = Path(__file__).parent / "per_paper"
OUT_ALL       = Path(__file__).parent / "master_dataset.csv"
OUT_STRICT    = Path(__file__).parent / "master_dataset_strict.csv"

FDM_KEYWORDS  = {"fff", "fdm", "3d printing", "fused", "fused filament", "fused deposition"}

def classify_tier(row: pd.Series) -> str:
    polymer    = str(row.get("Polymer", "")).strip().upper()
    processing = str(row.get("Processing Method", "")).strip().lower()
    standard   = str(row.get("Testing Standard", "")).strip().upper()
    notes      = str(row.get("Extraction Notes", "")).upper()

    if "FAIL_NOT_PLA" in notes or polymer not in ("PLA",):
        return "FAIL"
    is_fdm = any(k in processing for k in FDM_KEYWORDS)
    if not is_fdm and "FAIL_NOT_FDM" in notes:
        return "LOW"
    if not is_fdm:
        return "LOW"
    if "ASTM D638" in standard:
        return "HIGH"
    return "MEDIUM"


def load_paper(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    df.insert(0, "source_paper", csv_path.stem)
    df["quality_tier"] = df.apply(classify_tier, axis=1)
    return df


def main():
    frames = []
    skipped = []

    for csv_file in sorted(PER_PAPER_DIR.glob("*.csv")):
        df = load_paper(csv_file)
        if df.empty:
            skipped.append(csv_file.stem)
            continue
        frames.append(df)

    if not frames:
        print("No data rows found in any per-paper CSV.")
        sys.exit(1)

    master = pd.concat(frames, ignore_index=True)
    master.to_csv(OUT_ALL, index=False)

    strict = master[master["quality_tier"] == "HIGH"].copy()
    strict.to_csv(OUT_STRICT, index=False)

    # ── summary report ──────────────────────────────────────────────────────
    print("\n=== CONSOLIDATION REPORT ===\n")
    print(f"Total rows (all tiers):  {len(master)}")
    print(f"Strict filter (HIGH):    {len(strict)}")
    print()

    tier_counts = master.groupby(["source_paper", "quality_tier"]).size().reset_index(name="rows")
    print(tier_counts.to_string(index=False))

    print("\n--- Tier breakdown ---")
    print(master["quality_tier"].value_counts().to_string())

    print("\n--- HIGH-tier papers ---")
    high_papers = master[master["quality_tier"] == "HIGH"].groupby("source_paper").size()
    print(high_papers.to_string())

    print(f"\nOutputs written:")
    print(f"  {OUT_ALL}")
    print(f"  {OUT_STRICT}")

    if skipped:
        print(f"\nSkipped (0 rows): {', '.join(skipped)}")

    print("\n--- n-ladder feasibility ---")
    n = len(strict)
    hiremath = 200
    print(f"  Hiremath primary:         {hiremath} rows  → ladder: 200/100/50/25/10 ✓")
    print(f"  Supplementary strict:     {n} rows  → ladder: ", end="")
    if n >= 50:
        print(f"{n}/35/20/10 ✓")
    elif n >= 30:
        print(f"{n}/20/15/10  (tight but viable)")
    else:
        print(f"<30 rows — use Hiremath primary only")


if __name__ == "__main__":
    main()
