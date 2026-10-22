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
ECG = ("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='520' height='120' "
       "viewBox='0 0 520 120'%3E%3Cpath d='M0 62 H120 L136 62 L148 22 L166 100 L180 62 H250 L262 62 "
       "L274 36 L288 84 L300 62 H520' fill='none' stroke='%23ffffff' stroke-opacity='.22' "
       "stroke-width='2.5' stroke-linejoin='round'/%3E%3C/svg%3E")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600;700&display=swap');
  html, body, [class*="css"], .stMarkdown, .stSelectbox, .stSlider {font-family:'DM Sans', sans-serif;}
  .block-container {padding-top: 2.2rem; max-width: 1220px;}
  h1,h2,h3 {font-family:'DM Serif Display', Georgia, serif !important; font-weight:400 !important; letter-spacing:-.01em;}

  @keyframes rise {from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:none;}}

  /* hero */
  .hero {position:relative; overflow:hidden; color:#fff; padding:34px 38px 30px; border-radius:22px;
         background: radial-gradient(900px 300px at 85% -20%, rgba(94,234,212,.35), transparent 60%),
                     linear-gradient(120deg, #0a3550 0%, #0e6b80 55%, #138a8a 100%);
         box-shadow: 0 22px 44px -24px rgba(10,53,80,.7); animation: rise .6s both;}
  .hero::after {content:""; position:absolute; right:-20px; bottom:8px; width:520px; height:120px;
         background: url("__ECG__") no-repeat right bottom; pointer-events:none;}
  .hero .eyebrow {font-size:.76rem; letter-spacing:.16em; text-transform:uppercase; color:#8fe3d6; font-weight:700;}
  .hero h1 {color:#fff !important; margin:8px 0 10px; font-size:2.5rem; line-height:1.08;}
  .hero p  {color:#d6eef0; margin:0; font-size:1.05rem; max-width:62ch; line-height:1.6;}
  .hero .chips {margin-top:18px; display:flex; flex-wrap:wrap; gap:8px;}
  .hero .chips span {font-size:.78rem; font-weight:600; padding:5px 12px; border-radius:99px;
         background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.22); color:#eaffff;}

  /* kpi cards */
  .kpi {position:relative; background:#fff; border-radius:18px; padding:18px 20px 16px;
        border:1px solid #d9e6eb; box-shadow:0 10px 26px -18px rgba(10,53,80,.45);
        animation: rise .6s both; transition: transform .25s, box-shadow .25s; overflow:hidden;}
  .kpi:hover {transform: translateY(-4px); box-shadow:0 18px 34px -18px rgba(10,53,80,.55);}
  .kpi::before {content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--c,#0e7c86);}
  .kpi .ic {display:none;}
  .kpi {min-height:150px;}
  .kpi .v {font-family:'DM Serif Display', serif; font-size:2.15rem; line-height:1.1; color:#0a3550;}
  .kpi .l {font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#5d7f8c; margin-top:4px;}
  .kpi .n {font-size:.8rem; color:#7c98a3; margin-top:4px;}

  /* story cards */
  .story {background:#fff; border:1px solid #d9e6eb; border-radius:18px; padding:20px 22px; min-height:238px;
          box-shadow:0 10px 26px -20px rgba(10,53,80,.4);}
  .story .k {font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#0e7c86;}
  .story h4 {font-family:'DM Serif Display', serif; font-weight:400; font-size:1.25rem; margin:4px 0 8px; color:#0a3550;}
  .story p {font-size:.92rem; color:#42616d; margin:0; line-height:1.6;}

  .sec {margin:26px 0 4px;}
  .sec h3 {margin:0; font-size:1.55rem; color:#0a3550;}
  .sec p {margin:2px 0 0; color:#6a8794; font-size:.92rem;}

  .callout {background:linear-gradient(90deg, rgba(14,124,134,.12), rgba(14,124,134,.03)); border-left:4px solid #0e7c86;
            border-radius:10px; padding:11px 15px; font-size:.95rem; color:#12313f; margin:6px 0 10px;}
  .riskcard {border-radius:18px; padding:18px 20px; color:#fff; margin-top:-6px; animation: rise .5s both;}
  .riskcard b {font-family:'DM Serif Display', serif; font-weight:400; font-size:1.35rem;}
  .riskcard div {font-size:.93rem; opacity:.95; margin-top:4px; line-height:1.5;}

  /* tabs as pills */
  .stTabs [data-baseweb="tab-list"] {gap:8px; border-bottom:none; margin-top:8px;}
  .stTabs [data-baseweb="tab"] {background:#fff; border:1px solid #d9e6eb; border-radius:99px; padding:8px 20px; font-weight:600; height:auto;}
  .stTabs [aria-selected="true"] {background:#0a3550; color:#fff !important; border-color:#0a3550;}
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}

  /* bordered containers become soft panels */
  div[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-radius:18px; border-color:#d9e6eb !important;
          box-shadow:0 10px 26px -20px rgba(10,53,80,.4);}
  .stPlotlyChart {background:#fff; border:1px solid #d9e6eb; border-radius:18px; padding:8px 8px 0;
          box-shadow:0 10px 26px -20px rgba(10,53,80,.4);}
  .footer {text-align:center; color:#7c98a3; font-size:.82rem; margin:34px 0 6px;}
</style>
""".replace("__ECG__", ECG), unsafe_allow_html=True)


def kpi(col, icon, label, value, note, color="#0e7c86", delay=0):
    col.markdown(
        f'<div class="kpi" style="--c:{color}; animation-delay:{delay}ms">'
        f'<div class="ic">{icon}</div><div class="v">{value}</div>'
        f'<div class="l">{label}</div><div class="n">{note}</div></div>',
        unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec"><h3>{title}</h3>'
                + (f'<p>{sub}</p>' if sub else "") + '</div>',
                unsafe_allow_html=True)


# One consistent look for every chart
import plotly.graph_objects as go
import plotly.io as pio
pio.templates["clinic"] = go.layout.Template(layout=dict(
    font=dict(family="DM Sans, sans-serif", size=13, color="#12313f"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#0e7c86", "#e0533d", "#f2b24c", "#0a3550"],
    xaxis=dict(gridcolor="#e3edf1", zerolinecolor="#cfe0e6", linecolor="#cfe0e6"),
    yaxis=dict(gridcolor="#e3edf1", zerolinecolor="#cfe0e6", linecolor="#cfe0e6"),
))
pio.templates.default = "clinic"
RISK_SCALE = [[0, "#8fd3c8"], [0.5, "#f3c06b"], [1, "#e0533d"]]


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


HOVER_STYLE = dict(bgcolor="white", bordercolor="#cfe0e6",
                   font=dict(color="#0a3550", size=13, family="DM Sans"))


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
  <div class="eyebrow">Healthcare analytics · Predictive modeling</div>
  <h1>Hospital 30-Day Readmission Risk</h1>
  <p>Which diabetic patients are most likely to come back within 30 days, and where
     should care teams focus follow-up to cut avoidable readmissions and Medicare penalties?</p>
  <div class="chips"><span>SQL</span><span>Python</span><span>scikit-learn</span><span>Streamlit</span><span>Snowflake-ready</span></div>
</div>
""", unsafe_allow_html=True)

st.write("")
s1, s2, s3 = st.columns(3)
s1.markdown("""<div class="story"><div class="k">The problem</div><h4>Penalties for bounce-backs</h4>
<p>Medicare's Hospital Readmissions Reduction Program penalizes hospitals with high 30-day
readmission rates. Follow-up calls and medication reconciliation are limited, so the
question is who to prioritize.</p></div>""", unsafe_allow_html=True)
s2.markdown("""<div class="story"><div class="k">The data</div><h4>101K real encounters</h4>
<p>UCI <i>Diabetes 130-US hospitals</i>: about 101K inpatient encounters with demographics,
admission and discharge logistics, prior-year utilization, A1C, and 23 medication columns.
Label: readmitted in under 30 days.</p></div>""", unsafe_allow_html=True)
s3.markdown("""<div class="story"><div class="k">The approach</div><h4>Interpretable on purpose</h4>
<p>One row per patient, death and hospice discharges removed, then a logistic regression whose
odds ratios explain <i>why</i> a patient is flagged. The top risk decile is readmitted at about
2x the average rate.</p></div>""", unsafe_allow_html=True)

# ---------------- KPI row ----------------
base = df["readmit_lt30"].mean()
st.write("")
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "🏥", "Encounters", f"{len(df):,}", "one row per patient", "#0e7c86", 0)
kpi(c2, "🔁", "30-day readmit rate", f"{base*100:.1f}%", "the baseline to beat", "#e0533d", 80)
kpi(c3, "🎯", "Model ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}", "ranking quality", "#0a3550", 160)
kpi(c4, "📈", "Top-decile lift", f"{metrics.get('top_decile_lift_vs_random','n/a')}x",
    "readmits caught vs random outreach", "#f2b24c", 240)

st.write("")
tab1, tab2, tab3 = st.tabs(["Overview", "Risk Drivers", "Risk Scorer"])

# ---------------- Tab 1: Overview ----------------
with tab1:
    DIMS = {"discharge_group": "Where the patient went",
            "service_use_tier": "Prior-year visits",
            "age": "Age band",
            "medical_specialty": "Admitting specialty"}
    section("Who comes back?", "Pick a lens. The dashed line is the overall readmission rate.")
    dim = st.radio("Break readmission rate down by", list(DIMS), horizontal=True,
                   format_func=lambda k: DIMS[k], label_visibility="collapsed")
    g = (df.groupby(dim, observed=True)["readmit_lt30"]
         .agg(rate="mean", n="count").reset_index())
    g["rate_pct"] = g["rate"] * 100
    g = g.sort_values("rate_pct", ascending=True)
    hi, lo = g.iloc[-1], g.iloc[0]
    st.markdown(f'<div class="callout"><b>{hi[dim]}</b> readmits at <b>{hi["rate_pct"]:.1f}%</b>, '
                f'about {hi["rate_pct"]/max(lo["rate_pct"], 0.01):.1f}x the rate of <b>{lo[dim]}</b> '
                f'({lo["rate_pct"]:.1f}%).</div>', unsafe_allow_html=True)
    fig = px.bar(g, x="rate_pct", y=dim, orientation="h",
                 text=g["rate_pct"].round(1),
                 labels={"rate_pct": "30-day readmission rate (%)", dim: ""},
                 color="rate_pct", color_continuous_scale=RISK_SCALE,
                 custom_data=["n"])
    fig.update_traces(
        texttemplate="%{text}%", textposition="outside", marker_cornerradius=6,
        hovertemplate="<b>%{y}</b><br>Readmission rate: %{x:.1f}%"
                      "<br>Encounters: %{customdata[0]:,}<extra></extra>")
    fig.add_vline(x=base * 100, line_dash="dash", line_color="#0a3550",
                  annotation_text=f"overall {base*100:.1f}%", annotation_position="bottom right",
                  annotation_font_color="#0a3550")
    fig.update_layout(height=max(380, 34 * len(g) + 120), coloraxis_showscale=False,
                      margin=dict(l=10, r=30, t=30, b=10), hoverlabel=HOVER_STYLE)
    st.plotly_chart(fig, use_container_width=True)

    section("Highest-risk actionable segment",
            "Patients with 3+ prior visits, split by where they were discharged.")
    seg = (df[df["service_use_tier"] == "High (3+)"]
           .groupby("discharge_group", observed=True)["readmit_lt30"]
           .agg(rate="mean", n="count").reset_index()
           .sort_values("rate", ascending=False))
    seg["readmit_rate_%"] = (seg["rate"] * 100).round(1)
    st.dataframe(seg[["discharge_group", "n", "readmit_rate_%"]]
                 .rename(columns={"discharge_group": "Discharge",
                                  "n": "Encounters",
                                  "readmit_rate_%": "Readmit rate"}),
                 hide_index=True, use_container_width=True,
                 column_config={"Readmit rate": st.column_config.ProgressColumn(
                     "Readmit rate", format="%.1f%%", min_value=0,
                     max_value=float(max(seg["readmit_rate_%"].max(), 1) * 1.15)),
                     "Encounters": st.column_config.NumberColumn(format="%d")})

# ---------------- Tab 2: Risk Drivers ----------------
with tab2:
    if coefs.empty:
        st.info("Run `python src/model.py` to generate coefficients.")
    else:
        section("Why patients get flagged",
                "Each bar shows how a factor shifts a patient's odds of readmission, holding everything else fixed.")
        top = coefs.head(14).copy()
        top["label"] = top["feature"].apply(humanize_feature)
        top["direction"] = top["coef"].apply(
            lambda c: "Raises risk" if c > 0 else "Lowers risk")
        top = top.sort_values("coef")
        fig = px.bar(top, x="coef", y="label", orientation="h",
                     color="direction",
                     color_discrete_map={"Raises risk": "#e0533d",
                                         "Lowers risk": "#0e7c86"},
                     labels={"coef": "Effect on readmission risk (log-odds)",
                             "label": ""},
                     custom_data=["direction", "odds_ratio"])
        fig.update_traces(
            marker_cornerradius=6,
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}"
                          "<br>Odds multiplier: x%{customdata[1]:.2f}<extra></extra>")
        fig.add_vline(x=0, line_color="#0a3550", line_width=1)
        fig.update_layout(height=580, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.06, x=0, title_text=""),
                          hoverlabel=HOVER_STYLE)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red raises risk, teal lowers it. Hover any bar for the exact odds multiplier.")

# ---------------- Tab 3: Risk Scorer ----------------
with tab3:
    model = load_model()
    if model is None:
        st.info("Run `python src/model.py` to enable live scoring.")
    else:
        section("Score a patient", "Build a profile on the left and watch the estimated risk update.")
        left, right = st.columns([1.55, 1], gap="large")
        with left:
            with st.container(border=True):
                a, b = st.columns(2)
                with a:
                    age_mid = st.slider("Age (midpoint)", 5, 95, 65, step=10)
                    number_inpatient = st.slider("Prior inpatient visits", 0, 10, 1)
                    number_emergency = st.slider("Prior ER visits", 0, 10, 0)
                    number_outpatient = st.slider("Prior outpatient visits", 0, 20, 0)
                    time_in_hospital = st.slider("Length of stay (days)", 1, 14, 4)
                with b:
                    number_diagnoses = st.slider("# diagnoses", 1, 16, 8)
                    num_medications = st.slider("# medications", 1, 60, 16)
                    num_procedures = st.slider("# procedures", 0, 6, 1)
                    discharge_group = st.selectbox(
                        "Discharge disposition",
                        ["Home", "Home + Home Health", "SNF / Rehab / LTC",
                         "Transferred", "AMA / Other"])
                    insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
                k1, k2 = st.columns(2)
                a1c_high = 1 if k1.checkbox("A1C > 7 (high)") else 0
                med_changed = 1 if k2.checkbox("Diabetes meds changed") else 0

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
            right.warning(
                "Could not score with the saved model. This usually means the "
                "model was trained with a different scikit-learn version than "
                "the one running now. Re-run `python src/model.py` to refresh "
                f"the model file.\n\nDetails: {exc}")
        else:
            thr = metrics.get("operating_threshold", 0.5)
            hot = prob >= thr
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number=dict(suffix="%", valueformat=".1f",
                            font=dict(family="DM Serif Display", size=54, color="#0a3550")),
                gauge=dict(
                    axis=dict(range=[0, 80], tickcolor="#9fb6bf", ticksuffix="%"),
                    bar=dict(color="#e0533d" if hot else "#0e7c86", thickness=0.28),
                    bgcolor="white", borderwidth=0,
                    steps=[dict(range=[0, base * 100], color="#d9f0ec"),
                           dict(range=[base * 100, thr * 100], color="#fbe9c6"),
                           dict(range=[thr * 100, 80], color="#f8d0c9")],
                    threshold=dict(line=dict(color="#0a3550", width=3),
                                   thickness=0.85, value=thr * 100))))
            gauge.update_layout(height=290, margin=dict(l=24, r=24, t=30, b=0))
            with right:
                st.plotly_chart(gauge, use_container_width=True)
                mult = prob / base if base else 0
                if hot:
                    st.markdown(
                        f'<div class="riskcard" style="background:linear-gradient(120deg,#c4402c,#e0533d)">'
                        f'<b>Flag for follow-up</b><div>Above the operating threshold ({thr:.0%}). '
                        f'This patient is about {mult:.1f}x as likely to return as the average patient.</div></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="riskcard" style="background:linear-gradient(120deg,#0a6f78,#0e9a8f)">'
                        f'<b>Below the threshold</b><div>Under the operating threshold ({thr:.0%}). '
                        f'About {mult:.1f}x the average readmission risk of {base*100:.1f}%.</div></div>',
                        unsafe_allow_html=True)

st.markdown('<div class="footer">Built by Vidit Doshi · SQL · Python · scikit-learn · '
            'Streamlit · Snowflake-ready pipeline</div>', unsafe_allow_html=True)
