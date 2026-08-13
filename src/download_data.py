"""
Download the real UCI 'Diabetes 130-US hospitals for years 1999-2008' dataset
(~101,766 encounters, 50 features). Run this once on a machine with open
internet access; it writes data/raw/diabetic_data.csv.

Source: UCI ML Repository, dataset id 296.
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Usage:
    python src/download_data.py
"""
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT = RAW / "diabetic_data.csv"


def main() -> None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: pip install ucimlrepo"
        ) from exc

    import pandas as pd

    print("Downloading UCI dataset 296 (Diabetes 130-US hospitals)...")
    ds = fetch_ucirepo(id=296)
    # ucimlrepo splits the data into ids / features / targets. encounter_id and
    # patient_nbr live in `ids`, so we must include all three to get the full
    # 50-column schema (patient_nbr is needed for de-leaking in clean.py).
    frames = [getattr(ds.data, part, None)
              for part in ("ids", "features", "targets")]
    df = pd.concat([f for f in frames if f is not None], axis=1)
    # guard against any duplicated column names across the three frames
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv(OUT, index=False)
    print(f"Saved {df.shape[0]:,} rows x {df.shape[1]} cols -> {OUT}")


if __name__ == "__main__":
    main()
