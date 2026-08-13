"""
Hospital 30-Day Readmission Risk - interactive dashboard (Streamlit + Plotly).

Run:
    streamlit run app/streamlit_app.py

Tabs:
  1. Overview      - KPIs + rate by discharge / utilization / age
  2. Risk Drivers  - model odds ratios (why patients get flagged)
  3. Risk Scorer   - score a hypothetical patient with the trained model
"""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

st.set_page_config(page_title="Readmission Risk | Vidit Doshi", layout="wide",
                   page_icon="🏥")

# ---------------- Branding / theme ----------------
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  /* header banner */
  .hero {
      background: linear-gradient(120deg, #0b3d5c 0%, #1c6b8c 100%);
      color: #ffffff; padding: 26px 30px; border-radius: 14px;
      margin-bottom: 8px;
  }
  .hero h1 {color:#fff; margin:0; font-size:1.9rem; font-weight:700;}
  .hero p  {color:#d7ecf5; margin:6px 0 0 0; font-size:1.02rem;}
  .hero .by {color:#9fd0e3; font-size:0.9rem; margin-top:10px;}
  /* KPI cards */
  /* subtle card only — no background/text overrides, so labels + values
     stay readable in BOTH light and dark themes */
  div[data-testid="stMetric"] {border:1px solid rgba(28,107,140,.30);
      border-radius:12px; padding:14px 16px;}
  .stTabs [data-baseweb="tab-list"] {gap: 6px;}
  .stTabs [data-baseweb="tab"] {font-weight:600;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    enr = PROC / "enriched.parquet"
    if not enr.exists():
        enr = PROC / "enriched.csv"
    df = (pd.read_parquet(enr) if enr.suffix == ".parquet"
          else pd.read_csv(enr))
    metrics = json.loads((REPORTS / "metrics.json").read_text()) \
        if (REPORTS / "metrics.json").exists() else {}
    coefs = pd.read_csv(REPORTS / "coefficients.csv") \
        if (REPORTS / "coefficients.csv").exists() else pd.DataFrame()
    return df, metrics, coefs


@st.cache_resource
def load_model():
    p = MODELS / "readmit_logreg.joblib"
    if p.exists():
        import joblib
        return joblib.load(p)
    return None


# Plain-English labels for model feature names (used in the Risk Drivers chart)
FEATURE_LABELS = {
    "age_mid": "Age",
    "time_in_hospital": "Length of stay",
    "num_lab_procedures": "# lab procedures",
    "num_procedures": "# procedures",
    "num_medications": "# medications",
    "number_diagnoses": "# diagnoses",
    "number_inpatient": "Prior inpatient visits",
    "number_emergency": "Prior ER visits",
    "number_outpatient": "Prior outpatient visits",
    "prior_visits": "Total prior visits",
    "n_meds_changed": "# diabetes meds changed",
    "a1c_tested": "A1C was tested",
    "a1c_high": "A1C high (>7)",
    "med_changed": "Meds changed during stay",
    "on_diabetes_med": "On diabetes medication",
    "discharge_group": "Discharged to",
    "race": "Race",
    "gender": "Gender",
    "admission_type_id": "Admission type",
    "insulin": "Insulin",
    "metformin": "Metformin",
}

# Nicer values for one-hot categories that are codes rather than words
ADMISSION_TYPE = {"1": "Emergency", "2": "Urgent", "3": "Elective",
                  "4": "Newborn", "5": "Not available", "6": "Trauma",
                  "7": "Trauma center", "8": "Not mapped"}


def humanize_feature(feat: str) -> str:
    """Turn e.g. 'cat__discharge_group_SNF / Rehab / LTC' into
    'Discharged to: SNF / Rehab / LTC', and 'num__number_inpatient' into
    'Prior inpatient visits'."""
    f = feat.split("__", 1)[-1]           # drop the num__/cat__ prefix
    if f in FEATURE_LABELS:               # numeric feature, exact match
        return FEATURE_LABELS[f]
    for base, label in FEATURE_LABELS.items():   # one-hot 'base_value'
        if f.startswith(base + "_"):
            val = f[len(base) + 1:]
            if val == "infrequent_sklearn":
                val = "(rare values)"
            elif base == "admission_type_id":
                val = ADMISSION_TYPE.get(val, val)
            return f"{label}: {val}"
    return f.replace("_", " ")


HOVER_STYLE = dict(bgcolor="white", bordercolor="#d0dbe4",
                   font=dict(color="#0b3d5c", size=13))


@st.cache_resource(show_spinner="First load: building the dataset & model (~1 min)…")
def ensure_pipeline():
    """On a fresh deploy the processed data/model aren't in the repo (gitignored).
    Build them once: try the real UCI data, fall back to the synthetic generator
    if the download is blocked. Training here also matches the runtime's
    scikit-learn version, avoiding pickle-version mismatches."""
    import subprocess, sys
    src = ROOT / "src"
    have_data = (PROC / "enriched.parquet").exists() or (PROC / "enriched.csv").exists()
    have_model = (MODELS / "readmit_logreg.joblib").exists()
    if have_data and have_model:
        return True
    try:
        subprocess.run([sys.executable, str(src / "download_data.py")],
                       check=True, timeout=300)
    except Exception:
        subprocess.run([sys.executable, str(src / "make_synthetic.py")], check=True)
    for step in ("clean.py", "features.py", "model.py"):
        subprocess.run([sys.executable, str(src / step)], check=True)
    return True


try:
    ensure_pipeline()
    df, metrics, coefs = load_data()
except Exception as exc:
    st.error("Could not build the dataset/model automatically. Run the pipeline "
             f"locally (`python run_all.py`).\n\n{exc}")
    st.stop()

st.markdown("""
<div class="hero">
  <h1>🏥 Hospital 30-Day Readmission Risk</h1>
  <p>Predicting which diabetic patients are most likely to be readmitted within
     30 days — and where care teams should focus follow-up to cut avoidable
     readmissions and CMS penalties.</p>
  <div class="by">Built by Vidit Doshi · SQL · Python · scikit-learn · Streamlit · Snowflake-ready pipeline</div>
</div>
""", unsafe_allow_html=True)

# ---------------- About this project ----------------
with st.expander("ℹ️  About this project — the business problem & approach", expanded=True):
    st.markdown("""
**The problem.** Medicare's *Hospital Readmissions Reduction Program (HRRP)*
financially penalizes hospitals with high 30-day readmission rates. Follow-up
resources (care-coordination calls, med reconciliation) are limited, so the
question is: **which patients should we prioritize?**

**The data.** UCI *Diabetes 130-US hospitals* — ~101k inpatient encounters, 50
features (demographics, admission/discharge logistics, prior-year utilization,
labs like A1C, and 23 medication columns). Label: readmitted `<30` days vs not.

**The approach.** Clean & de-leak the data (one row per patient, remove
death/hospice discharges) → engineer utilization and discharge features →
train an **interpretable logistic-regression** risk model → surface the drivers
and the highest-risk segments for action.

**The payoff.** The model's top-decile patients are readmitted at ~2× the
average rate, so targeting them roughly **doubles the efficiency** of a
follow-up program. Explore the tabs below.
""")

# ---------------- KPI row ----------------
base = df["readmit_lt30"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Encounters", f"{len(df):,}")
c2.metric("30-day readmit rate", f"{base*100:.1f}%")
c3.metric("Model ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}")
c4.metric("Top-decile lift", f"{metrics.get('top_decile_lift_vs_random','—')}×",
          help="Patients in the model's top 10% risk band are readmitted this "
               "many times more often than average.")

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Risk Drivers", "🧮 Risk Scorer"])

# ---------------- Tab 1: Overview ----------------
with tab1:
    dim = st.selectbox(
        "Break readmission rate down by:",
        {"discharge_group": "Discharge disposition",
         "service_use_tier": "Prior-utilization tier",
         "age": "Age band",
         "medical_specialty": "Admitting specialty"}.keys(),
        format_func=lambda k: {"discharge_group": "Discharge disposition",
                               "service_use_tier": "Prior-utilization tier",
                               "age": "Age band",
                               "medical_specialty": "Admitting specialty"}[k],
    )
    g = (df.groupby(dim, observed=True)["readmit_lt30"]
         .agg(rate="mean", n="count").reset_index())
    g["rate_pct"] = g["rate"] * 100
    g = g.sort_values("rate_pct", ascending=True)
    fig = px.bar(g, x="rate_pct", y=dim, orientation="h",
                 text=g["rate_pct"].round(1),
                 labels={"rate_pct": "30-day readmission rate (%)", dim: ""},
                 color="rate_pct", color_continuous_scale="Reds",
                 custom_data=["n"])
    fig.update_traces(
        texttemplate="%{text}%", textposition="outside",
        hovertemplate="<b>%{y}</b><br>Readmission rate: %{x:.1f}%"
                      "<br>Encounters: %{customdata[0]:,}<extra></extra>")
    fig.add_vline(x=base * 100, line_dash="dash", line_color="#333",
                  annotation_text=f"overall {base*100:.1f}%")
    fig.update_layout(height=430, coloraxis_showscale=False,
                      margin=dict(l=10, r=10, t=30, b=10),
                      hoverlabel=HOVER_STYLE)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Highest-risk actionable segment**")
    seg = (df[df["service_use_tier"] == "High (3+)"]
           .groupby("discharge_group", observed=True)["readmit_lt30"]
           .agg(rate="mean", n="count").reset_index()
           .sort_values("rate", ascending=False))
    seg["readmit_rate_%"] = (seg["rate"] * 100).round(1)
    st.dataframe(seg[["discharge_group", "n", "readmit_rate_%"]]
                 .rename(columns={"discharge_group": "Discharge",
                                  "n": "Encounters"}),
                 hide_index=True, use_container_width=True)

# ---------------- Tab 2: Risk Drivers ----------------
with tab2:
    if coefs.empty:
        st.info("Run `python src/model.py` to generate coefficients.")
    else:
        top = coefs.head(14).copy()
        top["label"] = top["feature"].apply(humanize_feature)
        top["direction"] = top["coef"].apply(
            lambda c: "Raises risk" if c > 0 else "Lowers risk")
        top = top.sort_values("coef")
        fig = px.bar(top, x="coef", y="label", orientation="h",
                     color="direction",
                     color_discrete_map={"Raises risk": "#c0392b",
                                         "Lowers risk": "#2471a3"},
                     labels={"coef": "Effect on readmission risk (log-odds)",
                             "label": ""},
                     custom_data=["direction", "odds_ratio"])
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}"
                          "<br>Odds multiplier: ×%{customdata[1]:.2f}<extra></extra>")
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                          legend_title_text="", hoverlabel=HOVER_STYLE)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("How each factor shifts a patient's 30-day readmission odds, "
                   "holding everything else fixed — red raises risk, blue lowers it. "
                   "Hover for the exact odds multiplier.")

# ---------------- Tab 3: Risk Scorer ----------------
with tab3:
    model = load_model()
    if model is None:
        st.info("Run `python src/model.py` to enable live scoring.")
    else:
        st.write("Enter a patient profile to estimate 30-day readmission risk:")
        a, b, c = st.columns(3)
        with a:
            age_mid = st.slider("Age (midpoint)", 5, 95, 65, step=10)
            number_inpatient = st.slider("Prior inpatient visits", 0, 10, 1)
            number_emergency = st.slider("Prior ER visits", 0, 10, 0)
            number_outpatient = st.slider("Prior outpatient visits", 0, 20, 0)
        with b:
            time_in_hospital = st.slider("Length of stay (days)", 1, 14, 4)
            number_diagnoses = st.slider("# diagnoses", 1, 16, 8)
            num_medications = st.slider("# medications", 1, 60, 16)
            num_procedures = st.slider("# procedures", 0, 6, 1)
        with c:
            discharge_group = st.selectbox(
                "Discharge disposition",
                ["Home", "Home + Home Health", "SNF / Rehab / LTC",
                 "Transferred", "AMA / Other"])
            a1c_high = 1 if st.checkbox("A1C > 7 (high)") else 0
            med_changed = 1 if st.checkbox("Diabetes meds changed") else 0
            insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])

        row = {
            "age_mid": age_mid, "time_in_hospital": time_in_hospital,
            "num_lab_procedures": 43, "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_diagnoses": number_diagnoses,
            "number_inpatient": number_inpatient,
            "number_emergency": number_emergency,
            "number_outpatient": number_outpatient,
            "prior_visits": number_inpatient + number_emergency + number_outpatient,
            "n_meds_changed": 1 if med_changed else 0,
            "a1c_tested": 1 if a1c_high else 0, "a1c_high": a1c_high,
            "med_changed": med_changed, "on_diabetes_med": 1,
            "race": "Caucasian", "gender": "Female",
            "discharge_group": discharge_group,
            "admission_type_id": 1, "insulin": insulin, "metformin": "No",
        }
        try:
            prob = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
        except Exception as exc:
            st.warning(
                "Could not score with the saved model — this usually means the "
                "model was trained with a different scikit-learn version than "
                "the one running now. Re-run `python src/model.py` to refresh "
                f"the model file.\n\nDetails: {exc}")
        else:
            thr = metrics.get("operating_threshold", 0.5)
            st.metric("Estimated 30-day readmission risk", f"{prob*100:.1f}%")
            if prob >= thr:
                st.error(f"⚠️ Above operating threshold ({thr:.0%}) — "
                         "flag for follow-up outreach.")
            else:
                st.success(f"Below operating threshold ({thr:.0%}).")

st.caption("Built by Vidit Doshi · SQL · Python · scikit-learn · Streamlit · "
           "Snowflake-ready pipeline")
