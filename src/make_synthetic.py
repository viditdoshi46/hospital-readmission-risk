"""
Generate a synthetic stand-in for the UCI 'Diabetes 130-US hospitals' dataset.

It reproduces the EXACT 50-column schema and value domains of the real data,
with plausible statistical relationships to 30-day readmission (prior inpatient
visits, number of diagnoses, discharge disposition, age, A1C testing, insulin
changes). This lets the full pipeline run offline / in CI. For real results,
run src/download_data.py to fetch the true dataset -- no code changes needed,
the column names match.

Usage:
    python src/make_synthetic.py            # 101,766 rows (matches real N)
    python src/make_synthetic.py --rows 20000
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT = RAW / "diabetic_data.csv"

RNG = np.random.default_rng(42)

AGE_BUCKETS = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
               "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"]
GENDERS = ["Female", "Male", "Unknown/Invalid"]
SPECIALTIES = ["InternalMedicine", "Family/GeneralPractice", "Cardiology",
               "Surgery-General", "Emergency/Trauma", "Nephrology",
               "Orthopedics", "?", "Pulmonology", "Psychiatry"]
PAYERS = ["MC", "HM", "SP", "BC", "MD", "CP", "UN", "?", "CM", "OG"]
# Oral diabetes drugs (subset carrying signal); rest set mostly to 'No'
DRUG_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
             "glimepiride", "acetohexamide", "glipizide", "glyburide",
             "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
             "miglitol", "troglitazone", "tolazamide", "examide",
             "citoglipton", "insulin", "glyburide-metformin",
             "glipizide-metformin", "glimepiride-pioglitazone",
             "metformin-rosiglitazone", "metformin-pioglitazone"]
DRUG_LEVELS = ["No", "Steady", "Up", "Down"]


def _diag_code() -> str:
    # ICD-9-like codes as strings, matching real column semantics
    r = RNG.random()
    if r < 0.5:
        return str(RNG.integers(250, 460))          # circulatory / diabetes-ish
    if r < 0.8:
        return str(RNG.integers(460, 620))
    return f"{RNG.integers(1, 999)}.{RNG.integers(0, 9)}"


def generate(n: int) -> pd.DataFrame:
    df = pd.DataFrame()
    df["encounter_id"] = np.arange(1, n + 1) + 100000
    # some patients appear multiple times (like the real data)
    df["patient_nbr"] = RNG.integers(1, int(n * 0.7), size=n)

    age_idx = RNG.choice(len(AGE_BUCKETS), size=n,
                         p=[0.002, 0.006, 0.02, 0.04, 0.09, 0.16,
                            0.22, 0.25, 0.15, 0.062])
    df["race"] = RNG.choice(RACES, size=n, p=[0.75, 0.19, 0.02, 0.01, 0.01, 0.02])
    df["gender"] = RNG.choice(GENDERS, size=n, p=[0.537, 0.462, 0.001])
    df["age"] = [AGE_BUCKETS[i] for i in age_idx]
    # weight almost entirely missing in the real data
    df["weight"] = np.where(RNG.random(n) < 0.968, "?", "[75-100)")

    df["admission_type_id"] = RNG.choice([1, 2, 3, 4, 5, 6, 7, 8], size=n,
                                         p=[0.38, 0.18, 0.27, 0.01, 0.07, 0.05, 0.02, 0.02])
    disch = RNG.choice([1, 2, 3, 4, 5, 6, 7, 11, 13, 18, 22, 25], size=n,
                       p=[0.59, 0.02, 0.13, 0.03, 0.02, 0.11, 0.01,
                          0.015, 0.005, 0.03, 0.02, 0.02])
    df["discharge_disposition_id"] = disch
    df["admission_source_id"] = RNG.choice([1, 2, 3, 4, 5, 6, 7, 9, 17, 20], size=n,
                                           p=[0.29, 0.03, 0.02, 0.03, 0.02, 0.04,
                                              0.55, 0.005, 0.005, 0.01])

    df["time_in_hospital"] = RNG.integers(1, 15, size=n)
    df["payer_code"] = RNG.choice(PAYERS, size=n)
    df["medical_specialty"] = RNG.choice(SPECIALTIES, size=n,
                                         p=[0.15, 0.08, 0.05, 0.06, 0.07,
                                            0.03, 0.03, 0.49, 0.02, 0.02])
    df["num_lab_procedures"] = np.clip(RNG.normal(43, 20, n).round(), 1, 132).astype(int)
    df["num_procedures"] = RNG.integers(0, 7, size=n)
    df["num_medications"] = np.clip(RNG.normal(16, 8, n).round(), 1, 81).astype(int)
    df["number_outpatient"] = RNG.poisson(0.37, n)
    df["number_emergency"] = RNG.poisson(0.2, n)
    df["number_inpatient"] = RNG.poisson(0.64, n)
    df["diag_1"] = [_diag_code() for _ in range(n)]
    df["diag_2"] = [_diag_code() for _ in range(n)]
    df["diag_3"] = [_diag_code() for _ in range(n)]
    df["number_diagnoses"] = np.clip(RNG.normal(7.4, 1.9, n).round(), 1, 16).astype(int)

    df["max_glu_serum"] = RNG.choice(["None", "Norm", ">200", ">300"], size=n,
                                     p=[0.945, 0.026, 0.017, 0.012])
    a1c = RNG.choice(["None", "Norm", ">7", ">8"], size=n,
                     p=[0.83, 0.05, 0.04, 0.08])
    df["A1Cresult"] = a1c

    for c in DRUG_COLS:
        if c in ("metformin", "insulin", "glipizide", "glyburide", "pioglitazone",
                 "rosiglitazone", "glimepiride"):
            df[c] = RNG.choice(DRUG_LEVELS, size=n, p=[0.6, 0.3, 0.06, 0.04])
        else:
            df[c] = RNG.choice(DRUG_LEVELS, size=n, p=[0.985, 0.01, 0.003, 0.002])

    med_changed = (df[DRUG_COLS].isin(["Up", "Down"])).any(axis=1)
    df["change"] = np.where(med_changed | (RNG.random(n) < 0.25), "Ch", "No")
    df["diabetesMed"] = np.where((df[DRUG_COLS] != "No").any(axis=1) |
                                 (RNG.random(n) < 0.3), "Yes", "No")

    # ---- Build readmission with realistic signal, then bucket to NO/>30/<30 ----
    z = (
        -2.30
        + 0.42 * df["number_inpatient"]
        + 0.24 * df["number_emergency"]
        + 0.12 * (df["number_diagnoses"] - 7)
        + 0.05 * (df["time_in_hospital"] - 4)
        + 0.90 * (df["discharge_disposition_id"].isin([3, 4, 5, 6, 22, 2]).astype(int))
        + 0.55 * (df["A1Cresult"].isin([">7", ">8"]).astype(int))
        + 0.30 * (df["change"] == "Ch").astype(int)
        + 0.06 * (age_idx - 6)
        + 0.015 * (df["num_medications"] - 16)
        + RNG.normal(0, 0.32, n)
    )
    p_readmit = 1 / (1 + np.exp(-z))
    readmit_any = RNG.random(n) < p_readmit
    # of those readmitted, ~40% are <30 days (real data: ~11% <30 overall)
    lt30 = readmit_any & (RNG.random(n) < 0.52)
    readmitted = np.where(lt30, "<30", np.where(readmit_any, ">30", "NO"))
    df["readmitted"] = readmitted

    # inject realistic missingness already covered by '?' sentinels above
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=101766)
    args = ap.parse_args()
    df = generate(args.rows)
    df.to_csv(OUT, index=False)
    rate = (df["readmitted"] == "<30").mean()
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols -> {OUT}")
    print(f"30-day readmission rate: {rate:.1%}  (real data ~11.2%)")


if __name__ == "__main__":
    main()
