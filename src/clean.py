"""
Step 1 - Clean the raw encounters.

Decisions (documented for the README):
  * Replace '?', 'Unknown/Invalid', etc. with proper NaN.
  * Remove encounters ending in death or hospice -- those patients cannot be
    readmitted, so leaving them in would bias the target.
  * Keep only the FIRST encounter per patient to avoid leaking future visits
    into the training signal (each patient counted once).
  * Build the binary target readmit_lt30 = 1 if readmitted == '<30'.
  * Drop near-constant / high-missing / identifier columns.

Run:
    python src/clean.py
Writes data/processed/clean.parquet (and .csv).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import (DATA_RAW, DATA_PROCESSED, MISSING_TOKENS, DROP_COLS,
                    DISCHARGE_EXPIRED_HOSPICE, TARGET_RAW, TARGET)


def load_raw() -> pd.DataFrame:
    if not DATA_RAW.exists():
        raise SystemExit(
            f"Raw data not found at {DATA_RAW}.\n"
            "Run `python src/download_data.py` (real data) or "
            "`python src/make_synthetic.py` (offline demo) first."
        )
    return pd.read_csv(DATA_RAW, dtype=str)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # 1. sentinels -> NaN
    df = df.replace(list(MISSING_TOKENS), np.nan)

    # 2. numeric coercion
    num_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency",
                "number_inpatient", "number_diagnoses", "number_diagnoses",
                "admission_type_id", "discharge_disposition_id",
                "admission_source_id", "patient_nbr"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3. remove death / hospice discharges (cannot be readmitted)
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(DISCHARGE_EXPIRED_HOSPICE)]
    removed_death = before - len(df)

    # 4. drop rows with invalid gender
    df = df[df["gender"].notna()]

    # 5. one row per patient (first encounter) to prevent target leakage
    before = len(df)
    df = df.sort_values("patient_nbr").drop_duplicates("patient_nbr", keep="first")
    removed_dupes = before - len(df)

    # 6. build binary target
    df[TARGET] = (df[TARGET_RAW] == "<30").astype(int)

    # 7. drop unhelpful columns (keep medical_specialty -> handled as category)
    drop_now = [c for c in DROP_COLS if c in df.columns and c != "medical_specialty"]
    df = df.drop(columns=drop_now)

    df = df.reset_index(drop=True)
    print(f"[clean] raw={n0:,} | removed_death/hospice={removed_death:,} | "
          f"removed_dup_patients={removed_dupes:,} | final={len(df):,}")
    print(f"[clean] 30-day readmission rate = {df[TARGET].mean():.1%}")
    return df


def main() -> None:
    df = clean(load_raw())
    out_pq = DATA_PROCESSED / "clean.parquet"
    out_csv = DATA_PROCESSED / "clean.csv"
    try:
        df.to_parquet(out_pq, index=False)
        print(f"[clean] wrote {out_pq}")
    except Exception as exc:   # parquet engine may be missing
        print(f"[clean] parquet skipped ({exc}); writing csv only")
    df.to_csv(out_csv, index=False)
    print(f"[clean] wrote {out_csv}")


if __name__ == "__main__":
    main()
