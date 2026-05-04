# ============================================================
# app.py  —  Customer 360 Financial Health (XGBoost + SHAP)
# ============================================================
# Requirements (pip install before running):
#   streamlit pandas numpy xgboost shap matplotlib scikit-learn plotly
#
# Run:
#   streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, classification_report
)
import io
import os

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer 360 Financial Health",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Customer 360 Financial Health Score")
st.caption("Powered by XGBoost · SHAP Explainability · Give Me Some Credit Dataset")

# ──────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────
MODEL_PATH = "credit_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

bundle = load_model()

if bundle is None:
    st.error(
        f"❌ `{MODEL_PATH}` not found.  \n"
        "**Steps to fix:**  \n"
        "1. Run `train_model_colab.py` in Google Colab  \n"
        "2. Download `credit_model.pkl`  \n"
        "3. Place it in the **same folder** as `app.py`  \n"
        "4. Restart this app"
    )
    st.stop()

model        = bundle["model"]
FEATURES     = bundle["features"]
stored_auc   = bundle.get("auc", None)
stored_cm    = bundle.get("confusion_matrix", None)
shap_values  = bundle.get("shap_values", None)
shap_data    = bundle.get("shap_data", None)

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving Utilization",
    "age": "Age",
    "NumberOfTime30_59DaysPastDueNotWorse": "30-59 Days Late",
    "DebtRatio": "Debt Ratio",
    "MonthlyIncome": "Monthly Income (₹)",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines",
    "NumberOfTimes90DaysLate": "90+ Days Late",
    "NumberRealEstateLoansOrLines": "Real Estate Loans",
    "NumberOfTime60_89DaysPastDueNotWorse": "60-89 Days Late",
    "NumberOfDependents": "Dependents",
}

def prob_to_score(prob: float) -> int:
    """Convert default probability to a 300-850 style score."""
    return int(np.clip(850 - prob * 550, 300, 850))

def get_segment(score: int) -> tuple:
    if score >= 750:
        return "Excellent", "🟢", "#2ecc71"
    elif score >= 650:
        return "Good", "🟡", "#f1c40f"
    elif score >= 500:
        return "Risky", "🟠", "#e67e22"
    else:
        return "High Risk", "🔴", "#e74c3c"

def gauge_chart(score: int, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Credit Health Score", "font": {"size": 18}},
        gauge={
            "axis": {"range": [300, 850], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [300, 500], "color": "#fadbd8"},
                {"range": [500, 650], "color": "#fdebd0"},
                {"range": [650, 750], "color": "#fef9e7"},
                {"range": [750, 850], "color": "#eafaf1"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig

def shap_bar_chart(input_df: pd.DataFrame) -> plt.Figure:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(input_df)
    sv_arr = sv[0] if isinstance(sv, list) else sv[0]

    labels = [FEATURE_LABELS.get(f, f) for f in FEATURES]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in sv_arr]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = range(len(labels))
    ax.barh(y_pos, sv_arr, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on default probability)")
    ax.set_title("Feature Impact for This Customer")
    red_patch  = mpatches.Patch(color="#e74c3c", label="Increases default risk")
    green_patch = mpatches.Patch(color="#2ecc71", label="Decreases default risk")
    ax.legend(handles=[red_patch, green_patch], fontsize=8)
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Customer", "📂 Bulk Scoring (CSV)", "📈 Model Performance"])

# ══════════════════════════════════════════════════════════
# TAB 1 — SINGLE CUSTOMER
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        utilization    = st.slider("Revolving Credit Utilization (0–1)", 0.0, 1.0, 0.5, 0.01)
        age            = st.number_input("Age", min_value=18, max_value=100, value=40)
        late_30_59     = st.slider("Times 30-59 Days Late", 0, 20, 1)
        debt_ratio     = st.number_input("Debt Ratio", min_value=0.0, max_value=50.0, value=0.35, step=0.01)
        monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=1000)
    with col2:
        open_lines     = st.slider("Open Credit Lines & Loans", 0, 40, 8)
        late_90        = st.slider("Times 90+ Days Late", 0, 20, 0)
        real_estate    = st.slider("Real Estate Loans", 0, 20, 1)
        late_60_89     = st.slider("Times 60-89 Days Late", 0, 20, 0)
        dependents     = st.slider("Number of Dependents", 0, 20, 2)

    if st.button("🚀 Calculate Financial Health Score", use_container_width=True):
        input_data = pd.DataFrame([{
            "RevolvingUtilizationOfUnsecuredLines"  : utilization,
            "age"                                    : age,
            "NumberOfTime30_59DaysPastDueNotWorse"  : late_30_59,
            "DebtRatio"                              : debt_ratio,
            "MonthlyIncome"                          : monthly_income,
            "NumberOfOpenCreditLinesAndLoans"        : open_lines,
            "NumberOfTimes90DaysLate"                : late_90,
            "NumberRealEstateLoansOrLines"           : real_estate,
            "NumberOfTime60_89DaysPastDueNotWorse"  : late_60_89,
            "NumberOfDependents"                     : dependents,
        }])[FEATURES]

        prob  = model.predict_proba(input_data)[0][1]
        score = prob_to_score(prob)
        segment, emoji, color = get_segment(score)

        st.markdown("---")
        r1, r2, r3 = st.columns([2, 1, 1])
        with r1:
            st.plotly_chart(gauge_chart(score, color), use_container_width=True)
        with r2:
            st.metric("Default Probability", f"{prob*100:.1f}%")
            st.metric("Segment", f"{emoji} {segment}")
        with r3:
            st.metric("Credit Score", score)
            if segment == "Excellent":
                st.success("✅ Low Risk Customer")
            elif segment == "Good":
                st.info("ℹ️ Moderate Risk")
            elif segment == "Risky":
                st.warning("⚠️ Elevated Risk")
            else:
                st.error("❌ High Risk Customer")

        st.markdown("#### 🔬 SHAP Feature Importance (This Customer)")
        with st.spinner("Computing SHAP values…"):
            fig_shap = shap_bar_chart(input_data)
            st.pyplot(fig_shap)

        # Insights
        st.markdown("#### 📌 Flagged Insights")
        flags = []
        if utilization > 0.75:
            flags.append("⚠️ Very high revolving credit utilization (>75%)")
        if late_90 > 2:
            flags.append("⚠️ Multiple 90+ day late payments detected")
        if late_30_59 > 5:
            flags.append("⚠️ Frequent 30-59 day late payments")
        if debt_ratio > 0.5:
            flags.append("⚠️ Debt ratio above 50%")
        if monthly_income < 20000:
            flags.append("⚠️ Low monthly income")

        if flags:
            for f in flags:
                st.warning(f)
        else:
            st.success("✅ No major risk flags detected")

# ══════════════════════════════════════════════════════════
# TAB 2 — BULK SCORING
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("📂 Bulk Customer Scoring via CSV Upload")

    st.info(
        "Upload a CSV with **exactly** these columns (same as Kaggle dataset):  \n"
        "`RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30_59DaysPastDueNotWorse, "
        "DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate, "
        "NumberRealEstateLoansOrLines, NumberOfTime60_89DaysPastDueNotWorse, NumberOfDependents`"
    )

    # Download sample template
    sample = pd.DataFrame([{f: 0 for f in FEATURES}])
    sample_csv = sample.to_csv(index=False).encode()
    st.download_button("⬇️ Download Sample Template CSV", sample_csv, "template.csv", "text/csv")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        raw = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(raw):,}** rows")

        missing_cols = [c for c in FEATURES if c not in raw.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            X_bulk = raw[FEATURES].copy()

            # Fill common missing values
            if "MonthlyIncome" in X_bulk.columns:
                X_bulk["MonthlyIncome"].fillna(X_bulk["MonthlyIncome"].median(), inplace=True)
            if "NumberOfDependents" in X_bulk.columns:
                X_bulk["NumberOfDependents"].fillna(X_bulk["NumberOfDependents"].median(), inplace=True)

            with st.spinner("Scoring customers…"):
                probs  = model.predict_proba(X_bulk)[:, 1]
                scores = [prob_to_score(p) for p in probs]
                segs   = [get_segment(s)[0] for s in scores]

            result = raw.copy()
            result["Default_Probability_%"] = (probs * 100).round(2)
            result["Credit_Score"]          = scores
            result["Segment"]               = segs

            st.dataframe(result[["Default_Probability_%", "Credit_Score", "Segment"] + FEATURES].head(50))

            # Segment distribution
            seg_counts = pd.Series(segs).value_counts()
            fig_seg, ax_seg = plt.subplots(figsize=(5, 3))
            colors_map = {"Excellent": "#2ecc71", "Good": "#f1c40f",
                          "Risky": "#e67e22", "High Risk": "#e74c3c"}
            ax_seg.bar(seg_counts.index, seg_counts.values,
                       color=[colors_map.get(s, "grey") for s in seg_counts.index])
            ax_seg.set_title("Segment Distribution")
            ax_seg.set_ylabel("Count")
            st.pyplot(fig_seg)

            # Download results
            csv_out = result.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Scored Results CSV",
                csv_out,
                "scored_customers.csv",
                "text/csv",
                use_container_width=True
            )

# ══════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Model Performance Metrics")

    if stored_auc:
        st.metric("AUC-ROC (Test Set)", f"{stored_auc:.4f}")
    else:
        st.info("AUC not stored in bundle. Re-run training and resave.")

    # ── Confusion Matrix ──────────────────────────────────
    if stored_cm is not None:
        st.markdown("#### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=stored_cm,
            display_labels=["No Default", "Default"]
        )
        disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
        ax_cm.set_title("Confusion Matrix (Test Set)")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
    else:
        st.info("Confusion matrix not found in bundle.")

    # ── Global SHAP Summary ───────────────────────────────
    if shap_values is not None and shap_data is not None:
        st.markdown("#### Global SHAP Feature Importance")
        fig_gs, ax_gs = plt.subplots(figsize=(7, 4))
        sv = shap_values if not isinstance(shap_values, list) else shap_values[1]
        mean_abs = np.abs(sv).mean(axis=0)
        labels   = [FEATURE_LABELS.get(f, f) for f in FEATURES]
        sorted_idx = np.argsort(mean_abs)
        ax_gs.barh(
            [labels[i] for i in sorted_idx],
            mean_abs[sorted_idx],
            color="#3498db"
        )
        ax_gs.set_xlabel("Mean |SHAP value|")
        ax_gs.set_title("Global Feature Importance (SHAP)")
        fig_gs.tight_layout()
        st.pyplot(fig_gs)

        st.markdown("#### SHAP Beeswarm / Summary Plot")
        fig_bee, ax_bee = plt.subplots(figsize=(7, 4))
        shap.summary_plot(sv, shap_data, feature_names=FEATURES,
                          show=False, plot_size=None)
        ax_bee = plt.gca()
        ax_bee.set_yticklabels([FEATURE_LABELS.get(t.get_text(), t.get_text())
                                 for t in ax_bee.get_yticklabels()], fontsize=8)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.info("SHAP values not stored in bundle. Re-run training with SHAP cell.")

    # ── Feature Importances from model ───────────────────
    st.markdown("#### XGBoost Built-in Feature Importance")
    fi = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": [FEATURE_LABELS.get(f, f) for f in FEATURES],
                           "Importance": fi}).sort_values("Importance", ascending=False)
    fig_fi, ax_fi = plt.subplots(figsize=(7, 4))
    ax_fi.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color="#9b59b6")
    ax_fi.set_xlabel("Importance")
    ax_fi.set_title("XGBoost Feature Importances (gain)")
    fig_fi.tight_layout()
    st.pyplot(fig_fi)

    st.markdown("---")
    st.caption(
        "Model: XGBoost · Dataset: Give Me Some Credit (Kaggle) · "
        "Target: SeriousDlqin2yrs (default within 2 years)"
    )
    
