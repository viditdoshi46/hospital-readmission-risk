"""
Step 3 - Train an interpretable 30-day readmission risk model.

Model: L2-regularized logistic regression inside a scikit-learn Pipeline
(StandardScaler for numerics, OneHotEncoder for categoricals), with
class_weight='balanced' to handle the ~7-11% positive rate. Logistic
regression is chosen over a black box because care teams need to see WHY a
patient is flagged (odds ratios), and it is a defensible baseline.

Outputs:
  reports/metrics.json         -- AUC, precision/recall/F1, threshold, etc.
  reports/coefficients.csv     -- odds ratios per feature (interpretability)
  reports/segment_risk.csv     -- readmission rate by segment (dashboard)
  reports/figures/*.png        -- ROC, PR curve, confusion matrix, drivers
  models/readmit_logreg.joblib -- fitted pipeline

Run:
    python src/model.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support, roc_curve,
                             precision_recall_curve, confusion_matrix)
import joblib

from config import DATA_PROCESSED, REPORTS, FIGURES, TARGET
from features import MODEL_NUMERIC, MODEL_CATEG

MODELS = Path(__file__).resolve().parents[1] / "models"
MODELS.mkdir(exist_ok=True)


def load() -> pd.DataFrame:
    src = DATA_PROCESSED / "model_table.parquet"
    if not src.exists():
        src = DATA_PROCESSED / "model_table.csv"
    return pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)


def build_pipeline(num, cat) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), cat),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    return Pipeline([("pre", pre), ("clf", clf)])


def pick_threshold(y_true, y_prob, target_recall=0.60):
    """Choose the highest-precision threshold that still catches >= target_recall
    of true readmissions -- an operational trade-off care teams can act on."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # prec/rec have len = len(thr)+1; align
    best_t, best_p = 0.5, 0.0
    for p, r, t in zip(prec[:-1], rec[:-1], thr):
        if r >= target_recall and p > best_p:
            best_p, best_t = p, t
    return float(best_t)


def odds_ratios(pipe: Pipeline) -> pd.DataFrame:
    pre = pipe.named_steps["pre"]
    names = pre.get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": names, "coef": coefs})
    df["odds_ratio"] = np.exp(df["coef"])
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def segment_risk(enriched_path: Path) -> pd.DataFrame:
    df = (pd.read_parquet(enriched_path) if enriched_path.suffix == ".parquet"
          else pd.read_csv(enriched_path))
    rows = []
    for dim in ["discharge_group", "service_use_tier", "age", "a1c_high",
                "med_changed"]:
        g = df.groupby(dim, observed=True)[TARGET].agg(["mean", "count"])
        for idx, r in g.iterrows():
            rows.append({"dimension": dim, "segment": str(idx),
                         "readmit_rate": round(float(r["mean"]), 4),
                         "n": int(r["count"])})
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    num = [c for c in MODEL_NUMERIC if c in df.columns]
    cat = [c for c in MODEL_CATEG if c in df.columns]
    X, y = df[num + cat], df[TARGET].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = build_pipeline(num, cat)
    pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)[:, 1]

    auc = roc_auc_score(yte, prob)
    ap = average_precision_score(yte, prob)
    thr = pick_threshold(yte, prob, target_recall=0.60)
    pred = (prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        yte, pred, average="binary", zero_division=0)
    cm = confusion_matrix(yte, pred)

    # --- lift: how much better than random is the top-risk decile? ---
    order = np.argsort(-prob)
    top10 = order[: max(1, len(order) // 10)]
    base_rate = yte.mean()
    top_decile_rate = yte.values[top10].mean()
    lift = top_decile_rate / base_rate if base_rate else float("nan")

    metrics = {
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "base_readmit_rate": round(float(base_rate), 4),
        "roc_auc": round(float(auc), 4),
        "avg_precision": round(float(ap), 4),
        "operating_threshold": round(thr, 4),
        "precision_at_threshold": round(float(p), 4),
        "recall_at_threshold": round(float(r), 4),
        "f1_at_threshold": round(float(f1), 4),
        "top_decile_readmit_rate": round(float(top_decile_rate), 4),
        "top_decile_lift_vs_random": round(float(lift), 2),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    ors = odds_ratios(pipe)
    ors.to_csv(REPORTS / "coefficients.csv", index=False)

    enr = (DATA_PROCESSED / "enriched.parquet")
    if not enr.exists():
        enr = DATA_PROCESSED / "enriched.csv"
    seg = segment_risk(enr)
    seg.to_csv(REPORTS / "segment_risk.csv", index=False)

    joblib.dump(pipe, MODELS / "readmit_logreg.joblib")

    _make_figures(yte, prob, pred, ors, seg)
    print("[model] saved metrics, coefficients, segment_risk, figures, model")


def _make_figures(yte, prob, pred, ors, seg):
    # ROC
    fpr, tpr, _ = roc_curve(yte, prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(yte, prob):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC - 30-day readmission"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "roc_curve.png", dpi=120); plt.close()

    # PR curve
    prec, rec, _ = precision_recall_curve(yte, prob)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall"); plt.tight_layout()
    plt.savefig(FIGURES / "pr_curve.png", dpi=120); plt.close()

    # top drivers (odds ratios)
    top = ors.head(12).iloc[::-1]
    plt.figure(figsize=(6, 5))
    colors = ["#c0392b" if c > 0 else "#2471a3" for c in top["coef"]]
    plt.barh(top["feature"], top["odds_ratio"] - 1, color=colors)
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Odds ratio - 1  (red = raises risk, blue = lowers)")
    plt.title("Top readmission risk drivers")
    plt.tight_layout(); plt.savefig(FIGURES / "risk_drivers.png", dpi=120); plt.close()

    # segment risk: discharge group
    d = seg[seg["dimension"] == "discharge_group"].sort_values("readmit_rate")
    plt.figure(figsize=(6, 4))
    plt.barh(d["segment"], d["readmit_rate"] * 100, color="#2c7fb8")
    plt.xlabel("30-day readmission rate (%)")
    plt.title("Readmission rate by discharge disposition")
    plt.tight_layout(); plt.savefig(FIGURES / "risk_by_discharge.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
