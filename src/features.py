"""
Step 2 - Feature engineering.

Turns cleaned encounters into a modeling table + human-readable segments used
by the dashboard. Key engineered features:
  * age_mid           : age bucket -> numeric midpoint (5,15,...,95)
  * prior_visits      : outpatient + emergency + inpatient in prior year
  * discharge_group   : Home / Home+Care / SNF-Rehab / Transfer / AMA-Other
  * n_meds_changed    : count of diabetes drugs with Up/Down
  * a1c_tested        : whether A1C was measured
  * service_use_tier  : Low / Medium / High utilization band (for dashboard)

Run:
    python src/features.py
Writes data/processed/model_table.parquet/.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, TARGET

AGE_MID = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
           "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
           "[80-90)": 85, "[90-100)": 95}

DRUG_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
             "glimepiride", "acetohexamide", "glipizide", "glyburide",
             "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
             "miglitol", "troglitazone", "tolazamide", "insulin",
             "glyburide-metformin", "glipizide-metformin",
             "glimepiride-pioglitazone", "metformin-rosiglitazone",
             "metformin-pioglitazone"]


def discharge_group(id_: float) -> str:
    if id_ == 1:
        return "Home"
    if id_ in (6, 8):
        return "Home + Home Health"
    if id_ in (3, 4, 5, 22, 23, 24):
        return "SNF / Rehab / LTC"
    if id_ in (2, 9, 10, 15, 16, 17, 27, 28, 29, 30):
        return "Transferred"
    return "AMA / Other"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["age_mid"] = df["age"].map(AGE_MID).fillna(55).astype(int)

    for c in ["number_outpatient", "number_emergency", "number_inpatient"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["prior_visits"] = (df["number_outpatient"] + df["number_emergency"]
                          + df["number_inpatient"])

    df["discharge_group"] = df["discharge_disposition_id"].apply(discharge_group)

    present = [c for c in DRUG_COLS if c in df.columns]
    df["n_meds_changed"] = (df[present].isin(["Up", "Down"])).sum(axis=1)

    df["a1c_tested"] = (df.get("A1Cresult").notna()
                        & (df.get("A1Cresult") != "None")).astype(int) \
        if "A1Cresult" in df.columns else 0
    df["a1c_high"] = df.get("A1Cresult").isin([">7", ">8"]).astype(int) \
        if "A1Cresult" in df.columns else 0

    df["med_changed"] = (df.get("change") == "Ch").astype(int)
    df["on_diabetes_med"] = (df.get("diabetesMed") == "Yes").astype(int)

    df["medical_specialty"] = df.get("medical_specialty").fillna("Missing") \
        if "medical_specialty" in df.columns else "Missing"

    # utilization tier for the dashboard narrative
    df["service_use_tier"] = pd.cut(
        df["prior_visits"], bins=[-1, 0, 2, np.inf],
        labels=["Low (0)", "Medium (1-2)", "High (3+)"])

    return df


MODEL_NUMERIC = ["age_mid", "time_in_hospital", "num_lab_procedures",
                 "num_procedures", "num_medications", "number_diagnoses",
                 "number_inpatient", "number_emergency", "number_outpatient",
                 "prior_visits", "n_meds_changed", "a1c_tested", "a1c_high",
                 "med_changed", "on_diabetes_med"]
MODEL_CATEG = ["race", "gender", "discharge_group", "admission_type_id",
               "insulin", "metformin"]


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in (MODEL_NUMERIC + MODEL_CATEG) if c in df.columns]
    keep = cols + [TARGET, "age", "service_use_tier", "medical_specialty"]
    return df[[c for c in keep if c in df.columns]].copy()


def main() -> None:
    src = DATA_PROCESSED / "clean.parquet"
    if not src.exists():
        src = DATA_PROCESSED / "clean.csv"
    df = (pd.read_parquet(src) if src.suffix == ".parquet"
          else pd.read_csv(src))
    df = add_features(df)
    out = build_model_table(df)
    try:
        out.to_parquet(DATA_PROCESSED / "model_table.parquet", index=False)
    except Exception:
        pass
    out.to_csv(DATA_PROCESSED / "model_table.csv", index=False)
    # also persist the enriched full table for the dashboard
    try:
        df.to_parquet(DATA_PROCESSED / "enriched.parquet", index=False)
    except Exception:
        pass
    df.to_csv(DATA_PROCESSED / "enriched.csv", index=False)
    print(f"[features] model_table={out.shape} enriched={df.shape}")
    print(f"[features] features: {list(out.columns)}")


if __name__ == "__main__":
    main()
