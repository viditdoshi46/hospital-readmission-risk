# Hospital 30-Day Readmission Risk Analysis

**Business question:** Which patients are most likely to be readmitted within 30 days of discharge, *why*, and where should a hospital focus limited follow-up resources to reduce avoidable readmissions (and the CMS penalties tied to them)?

**Headline finding:** A patient's **discharge disposition** and **prior-year utilization** dominate readmission risk. High-utilization patients (3+ prior visits) discharged to a SNF/Rehab facility are readmitted at **16.2%** versus a **7.4%** baseline — roughly **2.2×**. An interpretable logistic-regression model reaches **ROC-AUC 0.677**, and its **top-decile risk band captures 2.4× more readmissions than random targeting** — meaning a care team that follows up with just the flagged 10% of patients reaches roughly two-and-a-half times as many future readmits per outreach call.

*(All figures below are measured on the real UCI dataset — run `python run_all.py --real` to reproduce.)*

---

## Recommendation (what a hospital should do)

1. **Prioritize follow-up on the top risk decile.** At the chosen operating threshold the model catches **~61% of 30-day readmissions** while flagging a minority of patients — a workable caseload for a care-coordination team.
2. **Target the highest-risk segment first:** high prior-utilization patients discharged to SNF/Rehab, transferred, or to home-health (14–19% readmission vs 7.4% baseline). Structured transition-of-care calls for this group offer the biggest absolute reduction.
3. **Use A1C testing + medication reconciliation as levers** — patients whose diabetes meds were changed during the stay and who had a high A1C show elevated risk, pointing to discharge-planning gaps.

---

## Architecture (end-to-end, cloud-ready)

```
UCI dataset ──► data/raw ──► clean.py ──► features.py ──► model.py ──► reports/ (+ model)
   │                │            │            │              │
(download_data)  (S3 landing)  SQL layer   engineered    logistic regression
                              (DuckDB now / Snowflake-ready)   + odds ratios
                                                          │
                                              app/streamlit_app.py (dashboard)
```

- **Python** — cleaning, feature engineering, modeling (scikit-learn).
- **SQL** — the same business questions in `sql/duckdb_analysis.sql` (runs locally) and `sql/snowflake_setup.sql` (S3 external stage → Snowflake table + views).
- **Streamlit + Plotly** — interactive dashboard: rate breakdowns, model risk drivers, and a live patient risk scorer.

---

## Dataset

[**UCI — Diabetes 130-US hospitals for years 1999-2008**](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008): ~101,766 diabetic inpatient encounters, 50 features, with a `readmitted` label of `NO` / `>30` / `<30`. Classic messy healthcare data — high-missing columns (`weight` ~97% missing), sentinel `?` values, and death/hospice discharges that must be excluded.

> **Reproducibility note:** the metrics in this README are measured on the **real** UCI dataset (`python run_all.py --real`, which calls `download_data.py` to pull dataset 296 via `ucimlrepo`). For a fully offline run, `src/make_synthetic.py` also generates a stand-in that reproduces the **exact 50-column schema and realistic risk relationships**, so the repo runs with zero downloads — the column names are identical, so no code changes are needed to switch between them.

---

## Key modeling decisions (visible, not hidden)

- **One row per patient (first encounter).** Prevents target leakage from a patient's own later visits.
- **Removed death/hospice discharges** — those patients cannot be readmitted, so keeping them biases the label.
- **`class_weight='balanced'`** for the ~11–14% positive rate; evaluated with **ROC-AUC + PR-AUC**, not accuracy (accuracy is misleading on imbalanced data).
- **Logistic regression on purpose.** Care teams need to see *why* a patient is flagged. The model reports **odds ratios per feature** (`reports/coefficients.csv`, `reports/figures/risk_drivers.png`) — a black box that can't explain itself is not clinically actionable.
- **Operating threshold chosen for 60% recall**, then reported precision/lift — an explicit business trade-off rather than an arbitrary 0.5 cutoff.

## Results (`reports/metrics.json`)

| Metric | Value |
|---|---|
| Encounters (after cleaning, one row/patient) | 69,987 |
| Baseline 30-day readmit rate | 7.4% |
| ROC-AUC | 0.677 |
| PR-AUC (avg precision) | 0.151 |
| Recall @ operating threshold | ~61% |
| Precision @ operating threshold | 12.3% |
| **Top-decile lift vs random** | **2.4×** |

Figures in `reports/figures/`: ROC curve, precision-recall curve, top risk drivers, readmission by discharge disposition.

---

## Run it

```bash
pip install -r requirements.txt

# option A: full pipeline on the offline demo data
python run_all.py

# option B: full pipeline on the REAL UCI dataset
python run_all.py --real

# explore the SQL business questions
python src/run_sql.py

# launch the interactive dashboard
streamlit run app/streamlit_app.py
```

## Repo structure

```
hospital-readmission-risk/
├── README.md
├── requirements.txt
├── run_all.py                 # one-command pipeline
├── src/
│   ├── config.py              # paths + constants
│   ├── download_data.py       # real UCI data
│   ├── make_synthetic.py      # offline schema-matched demo data
│   ├── clean.py               # missing values, leakage, target
│   ├── features.py            # engineered features + segments
│   ├── model.py               # logistic regression + metrics + figures
│   └── run_sql.py             # runs the DuckDB analysis
├── sql/
│   ├── duckdb_analysis.sql    # business questions, runs locally
│   └── snowflake_setup.sql    # S3 → Snowflake load + views
├── app/
│   └── streamlit_app.py       # interactive dashboard
└── reports/
    ├── metrics.json
    ├── coefficients.csv
    ├── segment_risk.csv
    └── figures/*.png
```

## Skills demonstrated

SQL (DuckDB + Snowflake) · Python (pandas, scikit-learn) · imbalanced-classification methodology · model interpretability · Streamlit/Plotly dashboarding · reproducible, cloud-ready pipeline design.

---
*Built by Vidit Doshi.*
