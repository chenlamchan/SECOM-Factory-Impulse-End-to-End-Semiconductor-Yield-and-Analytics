"""
07_Model_Cards.py — Model Registry & Cards
────────────────────────────────────────────
Read-only browser of the MLflow model registry with rich model cards.

Sections:
  1. Registry overview — all versions table, stage badges (Production/Staging/Archived)
  2. Champion model card — AUC, F1, confusion matrix, threshold analysis
  3. Version comparison — side-by-side metrics of any two selected versions
  4. Training lineage — timeline of training runs coloured by promotion outcome
  5. Feature manifest — feature list and imputation medians for the Production run
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import json

# Adapted Imports
from common.utils import (
    apply_page_config, TEAL, AMBER, RED, BLUE, CORAL, GRAY,
    PLOTLY_LAYOUT, badge,
)

apply_page_config("Model Cards", "📋")
st.title("📋 Model Registry & Cards")

MLFLOW_URI  = "http://mlflow:5000"
MODEL_NAME  = "secom_yield_predictor"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

STAGE_COLORS = {
    "Production": TEAL,
    "Staging":    BLUE,
    "Archived":   GRAY,
    "None":       AMBER,
}

STAGE_BADGE = {
    "Production": "ok",
    "Staging":    "warn",
    "Archived":   "idle",
    "None":       "idle",
}

@st.cache_data(ttl=60, show_spinner=False)
def load_all_versions() -> pd.DataFrame:
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        rows = []
        for v in versions:
            run = client.get_run(v.run_id) if v.run_id else None
            m   = run.data.metrics if run else {}
            rows.append({
                "version":       v.version,
                "stage":         v.current_stage,
                "run_id":        v.run_id,
                "creation_time": pd.to_datetime(v.creation_timestamp, unit="ms"),
                "model_type":    run.data.tags.get("model_type", "?") if run else "?",
                "auc":           round(m.get("test_auc", 0), 5),
                "f1_fail":       round(m.get("test_f1_fail", 0), 5),
                "precision":     round(m.get("test_prec_fail", 0), 5),
                "recall":        round(m.get("test_rec_fail", 0), 5),
                "promoted":      run.data.tags.get("promoted_to_production", "false") if run else "false",
            })
        return pd.DataFrame(rows).sort_values("version", ascending=False)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def load_run_details(run_id: str) -> dict:
    try:
        run = client.get_run(run_id)
        return {
            "metrics":    dict(run.data.metrics),
            "params":     dict(run.data.params),
            "tags":       dict(run.data.tags),
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
            "end_time":   pd.to_datetime(run.info.end_time, unit="ms") if run.info.end_time else None,
        }
    except Exception:
        return {}

@st.cache_data(ttl=120, show_spinner=False)
def load_training_runs() -> pd.DataFrame:
    try:
        experiment = mlflow.get_experiment_by_name("secom_yield_prediction")
        if not experiment:
            return pd.DataFrame()
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=50,
        )
        return runs
    except Exception:
        return pd.DataFrame()

# ─── Load data ────────────────────────────────────────────────────────────────
versions_df   = load_all_versions()
training_runs = load_training_runs()

# ─── Section 1: Registry overview ─────────────────────────────────────────────
st.subheader("Registry overview")

if versions_df.empty:
    st.info(
        "No registered models found. "
        "Complete the training pipeline (secom_ml_training_pipeline DAG) first.",
        icon="ℹ️"
    )
    st.stop()

# Stage summary counts
stage_counts = versions_df["stage"].value_counts().to_dict()
cs = st.columns(4)
for i, stage in enumerate(["Production", "Staging", "Archived", "None"]):
    if i < len(cs):
        cs[i].metric(stage, stage_counts.get(stage, 0))

# Versions table
display_df = versions_df.copy()
display_df["creation_time"] = display_df["creation_time"].dt.strftime("%Y-%m-%d %H:%M")

st.dataframe(
    display_df[[
        "version", "stage", "model_type", "auc", "f1_fail",
        "precision", "recall", "creation_time"
    ]].rename(columns={
        "version":       "Version",
        "stage":         "Stage",
        "model_type":    "Type",
        "auc":           "AUC",
        "f1_fail":       "F1 (Fail)",
        "precision":     "Precision",
        "recall":        "Recall",
        "creation_time": "Registered",
    }),
    use_container_width=True,
    hide_index=True,
    column_config={
        "AUC": st.column_config.ProgressColumn("AUC", min_value=0, max_value=1, format="%.5f"),
        "F1 (Fail)": st.column_config.ProgressColumn("F1 (Fail)", min_value=0, max_value=1, format="%.5f"),
    }
)

st.divider()

# ─── Section 2: Production model card ─────────────────────────────────────────
prod_row = versions_df[versions_df["stage"] == "Production"]

if not prod_row.empty:
    prod = prod_row.iloc[0]
    st.subheader(f"Production model — v{prod['version']} ({prod['model_type'].upper()})")

    details = load_run_details(prod["run_id"])
    m       = details.get("metrics", {})
    p       = details.get("params", {})

    # Metrics row
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("AUC",         f"{m.get('test_auc', 0):.5f}")
    mc2.metric("F1 (Fail)",   f"{m.get('test_f1_fail', 0):.5f}")
    mc3.metric("Precision",   f"{m.get('test_prec_fail', 0):.5f}")
    mc4.metric("Recall",      f"{m.get('test_rec_fail', 0):.5f}")
    mc5.metric("Train rows",  f"{p.get('train_rows', '?')}")
    mc6.metric("N features",  f"{p.get('n_features', '?')}")

    # Confusion matrix from stored metrics
    tp = int(m.get("tp", 0))
    tn = int(m.get("tn", 0))
    fp = int(m.get("fp", 0))
    fn = int(m.get("fn", 0))

    left_card, right_card = st.columns([1, 2])

    with left_card:
        if tp + tn + fp + fn > 0:
            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Pred Pass", "Pred Fail"],
                y=["Actual Pass", "Actual Fail"],
                colorscale=[[0, "rgba(29,158,117,0.15)"], [1, "#1D9E75"]],
                text=[[str(tn), str(fp)], [str(fn), str(tp)]],
                texttemplate="%{text}",
                textfont=dict(size=16, color="white"),
                showscale=False,
            ))
            fig_cm.update_layout(
                **PLOTLY_LAYOUT, height=240, title="Confusion matrix — test set",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    with right_card:
        try:
            with mlflow.start_run(run_id=prod["run_id"], nested=True):
                pass
            st.markdown(
                f"**Run ID:** `{prod['run_id'][:16]}...`  \n"
                f"**Registered:** {prod['creation_time']}  \n"
                f"**Training params:** "
                f"n_estimators={p.get('n_estimators', '?')}, "
                f"max_depth={p.get('max_depth', '?')}, "
                f"lr={p.get('learning_rate', '?')}  \n"
                f"**Scale pos weight:** {p.get('scale_pos_weight', '?')}  \n"
                f"**[View in MLflow]({MLFLOW_URI})**"
            )
        except Exception:
            pass

st.divider()

# ─── Section 3: Version comparison ────────────────────────────────────────────
st.subheader("Version comparison")

if len(versions_df) >= 2:
    v_options = versions_df["version"].tolist()
    col_a, col_b = st.columns(2)
    with col_a:
        v1 = st.selectbox("Version A", v_options, index=0)
    with col_b:
        v2 = st.selectbox("Version B", v_options, index=min(1, len(v_options) - 1))

    row_a = versions_df[versions_df["version"] == v1].iloc[0]
    row_b = versions_df[versions_df["version"] == v2].iloc[0]

    metrics = ["auc", "f1_fail", "precision", "recall"]
    labels  = ["AUC", "F1 (Fail)", "Precision", "Recall"]

    compare_df = pd.DataFrame({
        "Metric":    labels,
        f"v{v1}":   [row_a[m] for m in metrics],
        f"v{v2}":   [row_b[m] for m in metrics],
    })

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name=f"v{v1} ({row_a['stage']})",
        x=labels,
        y=[row_a[m] for m in metrics],
        marker_color=TEAL, text=[f"{row_a[m]:.4f}" for m in metrics],
        textposition="outside",
    ))
    fig_compare.add_trace(go.Bar(
        name=f"v{v2} ({row_b['stage']})",
        x=labels,
        y=[row_b[m] for m in metrics],
        marker_color=BLUE, text=[f"{row_b[m]:.4f}" for m in metrics],
        textposition="outside",
    ))
    fig_compare.update_layout(
        **PLOTLY_LAYOUT, height=340, barmode="group",
        yaxis_range=[0, 1.1], yaxis_title="Score",
        title=f"v{v1} vs v{v2}",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

st.divider()

# ─── Section 4: Training run lineage ─────────────────────────────────────────
st.subheader("Training run lineage")

if not training_runs.empty:
    runs_plot = training_runs.copy()

    auc_col  = "metrics.test_auc"    if "metrics.test_auc"    in runs_plot.columns else None
    type_col = "tags.model_type"     if "tags.model_type"     in runs_plot.columns else None
    prom_col = "tags.promoted_to_production" if "tags.promoted_to_production" in runs_plot.columns else None

    if auc_col:
        runs_plot = runs_plot[runs_plot[auc_col].notna()].copy()
        runs_plot["start_time_dt"] = pd.to_datetime(runs_plot["start_time"])
        runs_plot["promoted"]      = (
            runs_plot[prom_col] == "true" if prom_col else False
        )
        runs_plot["point_color"]   = np.where(runs_plot["promoted"], TEAL, GRAY)
        runs_plot["point_size"]    = np.where(runs_plot["promoted"], 12, 7)
        runs_plot["label"]         = runs_plot.apply(
            lambda r: f"{r.get(type_col, '?')} AUC={r[auc_col]:.4f}" +
                      (" ✓ Promoted" if r["promoted"] else ""),
            axis=1
        )

        fig_lineage = go.Figure()
        runs_sorted = runs_plot.sort_values("start_time_dt")
        fig_lineage.add_trace(go.Scatter(
            x=runs_sorted["start_time_dt"],
            y=runs_sorted[auc_col],
            mode="lines",
            line=dict(color=GRAY, width=1),
            showlegend=False,
        ))
        
        for promoted, color, name in [(True, TEAL, "Promoted"), (False, GRAY, "Not promoted")]:
            subset = runs_sorted[runs_sorted["promoted"] == promoted]
            if not subset.empty:
                fig_lineage.add_trace(go.Scatter(
                    x=subset["start_time_dt"],
                    y=subset[auc_col],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=10 if promoted else 7,
                                symbol="star" if promoted else "circle"),
                    text=subset["label"],
                    hovertemplate="%{text}<extra></extra>",
                ))

        fig_lineage.update_layout(
            **PLOTLY_LAYOUT, height=300,
            xaxis_title="Training date", yaxis_title="Test AUC",
            title="AUC across training runs — ★ = promoted to Production",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_lineage, use_container_width=True)
else:
    st.info("No training runs found in the MLflow experiment.")