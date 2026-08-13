"""Central paths and constants for the readmission project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "diabetic_data.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (DATA_PROCESSED, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# Missing-value sentinels used throughout the raw dataset
MISSING_TOKENS = {"?", "Unknown/Invalid", "None", "Not Available", "Not Mapped"}

# Columns dropped: near-constant, leakage, or >90% missing
DROP_COLS = [
    "weight",            # ~97% missing
    "payer_code",        # ~40% missing, low signal, admin field
    "medical_specialty", # ~49% missing (kept as 'Missing' category instead -> see features)
    "examide", "citoglipton",           # single-valued in real data
    "encounter_id",      # identifier
]

# Discharge dispositions that indicate death or hospice -> rows removed
# (patient cannot be readmitted). IDs per UCI IDS_mapping.
DISCHARGE_EXPIRED_HOSPICE = {11, 13, 14, 19, 20, 21}

TARGET_RAW = "readmitted"
TARGET = "readmit_lt30"   # 1 if readmitted within 30 days, else 0
