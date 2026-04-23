"""
05_ML_Predictions.py — Yield Prediction Analytics
───────────────────────────────────────────────────
Consumes gold_model_predictions (written by batch_inference.py → dbt)
and the /predict FastAPI endpoint for real-time single-wafer scoring.

Sections:
  1. Model health banner — current Production version, AUC, F1
  2. Prediction summary — confusion matrix counts, accuracy by line/shift
  3. Threshold explorer — slider to adjust decision boundary, live F1/prec/recall
  4. Recent wafer table — last 50 predictions with defect probability bar
  5. SHAP feature importance — from MLflow artifact (global) + per-wafer live call
  6. Live predictor — paste sensor readings, get instant prediction + SHAP waterfall
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# Adapted Imports
from common.utils import (
    apply_page_config, get_trino_engine, TEAL, AMBER, RED, BLUE, CORAL,
    PLOTLY_LAYOUT, GRAY, badge,
)

apply_page_config("ML Predictions", "🤖")
st.title("🤖 Yield Prediction Model")

# --- Trino Helper Wrapper ---
def query_trino(query: str, schema: str = "gold") -> pd.DataFrame:
    conn = get_trino_engine()
    cur = conn.cursor()
    if schema != "gold":
        cur.execute(f"USE {schema}")
    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    if schema != "gold":
        cur.execute("USE gold")
    return pd.DataFrame(rows, columns=columns)

SERVING_URL = "http://ml-serving:8001"

# ─── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_predictions(days: int = 7) -> pd.DataFrame:
    return query_trino(f"""
        SELECT
            observation_id, prediction_date, actual_status, predicted_status,
            defect_probability, yield_probability, is_correct, confusion_category,
            confidence_band, line_id, shift, model_version, last_updated_at
        FROM gold_model_predictions
        WHERE prediction_date >= CURRENT_DATE - INTERVAL '{days}' DAY
        ORDER BY last_updated_at DESC
        LIMIT 5000
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_model_info() -> dict:
    try:
        r = requests.get(f"{SERVING_URL}/model-info", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=300, show_spinner=False)
def load_mlflow_shap() -> pd.DataFrame | None:
    try:
        return query_trino("""
            SELECT sensor_id, abs_correlation AS mean_abs_shap, direction
            FROM gold_failure_pareto
            ORDER BY abs_correlation DESC
            LIMIT 20
        """, schema="gold")
    except Exception:
        return None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    day_window   = st.selectbox("Window", ["Last 1 day", "Last 7 days", "Last 30 days"], index=1)
    line_filter  = st.multiselect("Lines", ["LINE_A", "LINE_B", "LINE_C"],
                                   default=["LINE_A", "LINE_B", "LINE_C"])
    threshold    = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                              help="Probability above this threshold → predicted Fail")

days_map = {"Last 1 day": 1, "Last 7 days": 7, "Last 30 days": 30}
days     = days_map[day_window]

# ─── Model health banner ──────────────────────────────────────────────────────
model_info = load_model_info()

if "error" not in model_info:
    metrics = model_info.get("test_metrics", {})
    auc_val = metrics.get("test_auc", 0)
    f1_val  = metrics.get("test_f1_fail", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model version",   f"v{model_info.get('model_version', '?')}")
    col2.metric("Model type",      model_info.get("model_type", "?").upper())
    col3.metric("Test AUC",        f"{auc_val:.4f}" if auc_val else "—")
    col4.metric("Test F1 (Fail)",  f"{f1_val:.4f}" if f1_val else "—")
else:
    st.warning(f"ML serving offline: {model_info['error']}", icon="⚠️")

st.divider()

# ─── Load predictions ─────────────────────────────────────────────────────────
with st.spinner("Loading prediction data..."):
    pred_df = load_predictions(days)

if pred_df.empty:
    st.info(
        "No predictions available yet. "
        "Run the secom_ml_drift_monitor DAG to score recent silver data.",
        icon="ℹ️"
    )
    st.stop()

if line_filter:
    pred_df = pred_df[pred_df["line_id"].isin(line_filter)]

pred_df["adjusted_prediction"] = np.where(
    pred_df["defect_probability"] >= threshold, "Fail", "Pass"
)
pred_df["adjusted_correct"] = (pred_df["adjusted_prediction"] == pred_df["actual_status"]).astype(int)

# ─── Section 1: Confusion matrix KPIs ────────────────────────────────────────
st.subheader("Prediction performance")

tp = int(((pred_df["adjusted_prediction"] == "Fail") & (pred_df["actual_status"] == "Fail")).sum())
tn = int(((pred_df["adjusted_prediction"] == "Pass") & (pred_df["actual_status"] == "Pass")).sum())
fp = int(((pred_df["adjusted_prediction"] == "Fail") & (pred_df["actual_status"] == "Pass")).sum())
fn = int(((pred_df["adjusted_prediction"] == "Pass") & (pred_df["actual_status"] == "Fail")).sum())

total   = tp + tn + fp + fn
acc     = (tp + tn) / max(total, 1)
prec    = tp / max(tp + fp, 1)
rec     = tp / max(tp + fn, 1)
f1_live = 2 * prec * rec / max(prec + rec, 1e-9)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy",          f"{acc:.2%}")
c2.metric("Precision (Fail)",  f"{prec:.2%}")
c3.metric("Recall (Fail)",     f"{rec:.2%}")
c4.metric("F1 (Fail)",         f"{f1_live:.4f}")
c5.metric("Threshold",         f"{threshold:.2f}")

cm_data = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Actual Pass", "Actual Fail"],
    columns=["Pred Pass", "Pred Fail"],
)
left, right = st.columns([1, 2])

with left:
    fig_cm = go.Figure(go.Heatmap(
        z=cm_data.values,
        x=cm_data.columns.tolist(),
        y=cm_data.index.tolist(),
        colorscale=[[0, "rgba(29,158,117,0.15)"], [1, "#1D9E75"]],
        text=cm_data.values.astype(str),
        texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        showscale=False,
    ))
    fig_cm.update_layout(
        **PLOTLY_LAYOUT, height=250, title="Confusion matrix",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with right:
    fig_dist = go.Figure()
    for status, color in [("Pass", TEAL), ("Fail", RED)]:
        subset = pred_df[pred_df["actual_status"] == status]["defect_probability"]
        fig_dist.add_trace(go.Histogram(
            x=subset, name=f"Actual {status}",
            marker_color=color, opacity=0.65,
            nbinsx=30,
        ))
    fig_dist.add_vline(x=threshold, line_dash="dash", line_color=AMBER,
                        annotation_text=f"Threshold {threshold:.2f}")
    fig_dist.update_layout(
        **PLOTLY_LAYOUT, height=250,
        title="Defect probability distribution",
        xaxis_title="P(Fail)", yaxis_title="Count", barmode="overlay",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

st.divider()

# ─── Section 3: Accuracy by line and shift ────────────────────────────────────
st.subheader("Accuracy by line & shift")

grp = (
    pred_df
    .groupby(["line_id", "shift"])["adjusted_correct"]
    .mean()
    .reset_index()
    .rename(columns={"adjusted_correct": "accuracy"})
)
grp["accuracy_pct"] = grp["accuracy"] * 100

if not grp.empty:
    fig_bar = px.bar(
        grp, x="line_id", y="accuracy_pct", color="shift",
        barmode="group", text_auto=".1f",
        color_discrete_map={"Day": TEAL, "Swing": BLUE, "Night": AMBER},
    )
    fig_bar.add_hline(y=90, line_dash="dot", line_color=RED,
                       annotation_text="90% target")
    fig_bar.update_layout(**PLOTLY_LAYOUT, height=300,
                           yaxis_title="Accuracy %", yaxis_range=[0, 105])
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ─── Section 4: SHAP feature importance ─────────────────────────────────────
st.subheader("Feature importance (SHAP × correlation proxy)")

shap_df = load_mlflow_shap()
if shap_df is not None and not shap_df.empty:
    shap_df = shap_df.head(15).sort_values("mean_abs_shap", ascending=True)
    fig_shap = go.Figure(go.Bar(
        y=[f"Sensor {r['sensor_id']}" for _, r in shap_df.iterrows()],
        x=shap_df["mean_abs_shap"].values,
        orientation="h",
        marker_color=[RED if d == "Higher reading → more failures" else TEAL
                      for d in shap_df["direction"]],
        text=[f"{v:.4f}" for v in shap_df["mean_abs_shap"].values],
        textposition="outside",
    ))
    fig_shap.update_layout(
        **PLOTLY_LAYOUT, height=420,
        xaxis_title="Mean |SHAP| / |correlation|",
        title="Top 15 sensors by defect influence",
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# ─── Section 5: Recent predictions table ─────────────────────────────────────
st.subheader("Recent predictions")

display_cols = [
    "observation_id", "prediction_date", "line_id", "shift",
    "defect_probability", "predicted_status", "actual_status",
    "confidence_band",
]
disp = pred_df[display_cols].head(50).copy()
disp["defect_probability"] = disp["defect_probability"].round(4)

st.dataframe(
    disp.rename(columns={
        "observation_id":    "Wafer ID",
        "prediction_date":   "Date",
        "line_id":           "Line",
        "shift":             "Shift",
        "defect_probability":"P(Fail)",
        "predicted_status":  "Predicted",
        "actual_status":     "Actual",
        "confidence_band":   "Confidence",
    }),
    use_container_width=True,
    hide_index=True,
    column_config={
        "P(Fail)": st.column_config.ProgressColumn(
            "P(Fail)", min_value=0, max_value=1, format="%.4f"
        )
    }
)

st.divider()

# ─── Section 6: Live predictor ────────────────────────────────────────────────
st.subheader("Live wafer predictor")
st.caption("Enter raw sensor readings to get an instant prediction from the Production model.")

with st.form("live_predict"):
    cols = st.columns(5)
    sensor_inputs = {}
    top5 = ["59", "103", "511", "424", "158"]
    labels = ["Sensor 59", "Sensor 103", "Sensor 511", "Sensor 424", "Sensor 158"]

    for i, (s, label) in enumerate(zip(top5, labels)):
        with cols[i]:
            sensor_inputs[s] = st.number_input(label, value=0.0, format="%.4f")

    missing_rate = st.slider("Missing sensor rate", 0.0, 1.0, 0.05, 0.01)
    submitted    = st.form_submit_button("Predict")

if submitted:
    payload = {s: v for s, v in sensor_inputs.items()}
    payload["missing_sensor_rate"] = missing_rate

    try:
        r = requests.post(f"{SERVING_URL}/predict", json=payload, timeout=10)
        result = r.json()

        col_r1, col_r2, col_r3 = st.columns(3)
        prob = result.get("defect_probability", 0)
        pred = result.get("prediction", "?")
        conf = result.get("confidence_band", "?")

        color = RED if pred == "Fail" else TEAL
        col_r1.metric("Prediction", pred)
        col_r2.metric("P(Fail)",    f"{prob:.4f}")
        col_r3.metric("Confidence", conf)

        drivers = result.get("top_drivers", [])
        if drivers:
            st.markdown("**Top SHAP drivers:**")
            for d in drivers:
                direction_icon = "🔴" if "increases" in d["direction"] else "🟢"
                st.markdown(
                    f"- {direction_icon} **{d['feature']}**: "
                    f"SHAP = {d['shap_value']:+.4f} — {d['direction']}"
                )
    except Exception as e:
        st.error(f"Serving endpoint unavailable: {e}")