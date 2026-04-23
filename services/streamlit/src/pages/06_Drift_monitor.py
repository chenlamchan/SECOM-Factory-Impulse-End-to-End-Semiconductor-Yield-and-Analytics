"""
06_Drift_Monitor.py — Data & Concept Drift Monitoring
───────────────────────────────────────────────────────
Surfaces the outputs of drift_monitor.py run by the daily Airflow DAG.

Sections:
  1. Drift status banner — last check timestamp, overall drift flag
  2. PSI heatmap — per-sensor PSI value with green/amber/red colouring
  3. Score distribution over time — defect probability trend (concept drift)
  4. Sensor comparison — current vs Phase I distribution for any selected sensor
  5. Evidently report embed — links to the latest full Evidently HTML report
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# Adapted Imports
from common.utils import (
    apply_page_config, get_trino_engine, TEAL, AMBER, RED, BLUE, GRAY,
    PLOTLY_LAYOUT, badge,
)

apply_page_config("Drift Monitor", "📡")
st.title("📡 Data & Concept Drift Monitor")

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

st.caption(
    "PSI < 0.10 = stable · PSI 0.10–0.20 = monitor · PSI > 0.20 = retrain triggered. "
    "Computed daily by the secom_ml_drift_monitor Airflow DAG."
)

SERVING_URL = "http://ml-serving:8001"
SENSOR_COLS = [
    "59", "103", "511", "424", "158",
    "4", "5", "6", "7", "8",
    "9", "10", "11", "12", "13",
    "57", "58", "60", "61", "62",
    "100", "101", "102", "104", "105",
    "200", "201", "202", "300", "400",
]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    psi_threshold = st.slider("PSI alert threshold", 0.05, 0.40, 0.20, 0.01)
    lookback_days = st.selectbox("Comparison window", [7, 14, 30], index=0)
    sensor_select = st.selectbox("Sensor detail view", SENSOR_COLS, index=0)


# ─── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_sensor_stats() -> pd.DataFrame:
    return query_trino(
        "SELECT sensor_id, line_id, mu, sigma, ucl, lcl, frozen_at FROM gold_sensor_stats"
    )

@st.cache_data(ttl=60, show_spinner=False)
def load_current_sensor_window(sensor_id: str, days: int) -> pd.DataFrame:
    return query_trino(
        f"""
        SELECT "{sensor_id}" AS val, process_timestamp, line_id
        FROM silver_secom_reporting
        WHERE "{sensor_id}" IS NOT NULL
          AND CAST(process_timestamp AS DATE) >= CURRENT_DATE - INTERVAL '{days}' DAY
        ORDER BY process_timestamp
        """,
        schema="silver",
    )

@st.cache_data(ttl=60, show_spinner=False)
def load_score_trend(days: int) -> pd.DataFrame:
    return query_trino(f"""
        SELECT
            prediction_date,
            AVG(defect_probability)            AS avg_defect_prob,
            STDDEV(defect_probability)         AS std_defect_prob,
            COUNT(*)                           AS n_predictions,
            SUM(CASE WHEN prediction=1 THEN 1 ELSE 0 END) AS n_predicted_fail
        FROM gold_model_predictions
        WHERE prediction_date >= CURRENT_DATE - INTERVAL '{days}' DAY
        GROUP BY prediction_date
        ORDER BY prediction_date
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_spc_violation_trend(days: int) -> pd.DataFrame:
    return query_trino(f"""
        SELECT
            CAST(process_timestamp AS DATE) AS viol_date,
            sensor_id,
            COUNT(*) AS violation_count
        FROM gold_spc_violations
        WHERE CAST(process_timestamp AS DATE) >= CURRENT_DATE - INTERVAL '{days}' DAY
        GROUP BY CAST(process_timestamp AS DATE), sensor_id
        ORDER BY viol_date
    """)

def _compute_psi(current: pd.Series, mu: float, sigma: float, bins: int = 10) -> float:
    from scipy.stats import norm
    cur = current.dropna().values
    if len(cur) == 0 or sigma == 0:
        return 0.0
    edges    = np.linspace(mu - 4 * sigma, mu + 4 * sigma, bins + 1)
    cur_cnt  = np.histogram(cur, bins=edges)[0]
    ref_cdf  = norm.cdf(edges, loc=mu, scale=sigma)
    ref_pct  = np.diff(ref_cdf) + 1e-6
    ref_pct /= ref_pct.sum()
    cur_pct  = (cur_cnt + 1e-6) / (len(cur) + 1e-6 * bins)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ─── Load data ────────────────────────────────────────────────────────────────
stats_df   = load_sensor_stats()
score_df   = load_score_trend(lookback_days)
spc_df     = load_spc_violation_trend(lookback_days)

# ─── Compute live PSI per sensor ─────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner="Computing PSI per sensor...")
def compute_all_psi(days: int, threshold: float) -> pd.DataFrame:
    rows = []
    sensor_list = ", ".join([f'"{s}"' for s in SENSOR_COLS])
    try:
        current_df = query_trino(
            f"""
            SELECT {sensor_list}
            FROM silver_secom_reporting
            WHERE CAST(process_timestamp AS DATE) >= CURRENT_DATE - INTERVAL '{days}' DAY
            """,
            schema="silver"
        )
    except Exception:
        return pd.DataFrame()

    ref_lookup = {
        row["sensor_id"]: (row["mu"], row["sigma"])
        for _, row in stats_df[stats_df["line_id"] == "LINE_A"].iterrows()
    }

    for sensor in SENSOR_COLS:
        if sensor not in current_df.columns:
            continue
        mu, sigma = ref_lookup.get(sensor, (0.0, 1.0))
        psi = _compute_psi(current_df[sensor], mu, sigma)
        level = "red" if psi > threshold else ("amber" if psi > 0.10 else "green")
        rows.append({"sensor": sensor, "psi": round(psi, 4), "level": level,
                     "mu": mu, "sigma": sigma})

    return pd.DataFrame(rows)


with st.spinner("Calculating PSI scores..."):
    psi_df = compute_all_psi(lookback_days, psi_threshold)

# ─── Banner ───────────────────────────────────────────────────────────────────
if not psi_df.empty:
    n_red   = (psi_df["level"] == "red").sum()
    n_amber = (psi_df["level"] == "amber").sum()
    n_green = (psi_df["level"] == "green").sum()

    if n_red > 0:
        st.error(
            f"**Drift detected** — {n_red} sensor(s) exceed PSI threshold {psi_threshold:.2f}. "
            "Retraining has been triggered via NATS.",
            icon="🔴"
        )
    elif n_amber > 0:
        st.warning(
            f"{n_amber} sensor(s) show moderate drift (PSI 0.10–{psi_threshold:.2f}). "
            "No retraining triggered — monitor closely.",
            icon="🟡"
        )
    else:
        st.success(
            f"All {n_green} sensors within stable range (PSI < 0.10).",
            icon="🟢"
        )

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Stable (PSI < 0.10)",     n_green)
    kc2.metric("Moderate (0.10–0.20)",    n_amber)
    kc3.metric(f"Alert (> {psi_threshold:.2f})", n_red, delta_color="inverse")

st.divider()

# ─── PSI heatmap ──────────────────────────────────────────────────────────────
st.subheader("PSI per sensor — all tracked sensors")

if not psi_df.empty:
    psi_sorted = psi_df.sort_values("psi", ascending=False)

    color_map = {"green": TEAL, "amber": AMBER, "red": RED}
    fig_psi = go.Figure(go.Bar(
        x=[f"S{r['sensor']}" for _, r in psi_sorted.iterrows()],
        y=psi_sorted["psi"].values,
        marker_color=[color_map[l] for l in psi_sorted["level"]],
        text=[f"{v:.3f}" for v in psi_sorted["psi"].values],
        textposition="outside",
    ))
    fig_psi.add_hline(y=psi_threshold, line_dash="dash", line_color=RED,
                       annotation_text=f"Alert threshold {psi_threshold:.2f}")
    fig_psi.add_hline(y=0.10, line_dash="dot", line_color=AMBER,
                       annotation_text="Moderate 0.10")
    fig_psi.update_layout(
        **PLOTLY_LAYOUT, height=340,
        xaxis_title="Sensor", yaxis_title="PSI",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_psi, use_container_width=True)

st.divider()

# ─── Score trend (concept drift) ──────────────────────────────────────────────
st.subheader("Prediction score drift — P(Fail) over time")

if not score_df.empty:
    score_df["prediction_date"] = pd.to_datetime(score_df["prediction_date"])
    score_df = score_df.sort_values("prediction_date")

    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(
        x=score_df["prediction_date"],
        y=score_df["avg_defect_prob"],
        mode="lines+markers",
        name="Avg P(Fail)",
        line=dict(color=TEAL, width=2),
        fill="tozeroy", fillcolor="rgba(29,158,117,0.08)",
    ))
    
    if "std_defect_prob" in score_df.columns:
        upper = score_df["avg_defect_prob"] + score_df["std_defect_prob"].fillna(0)
        lower = score_df["avg_defect_prob"] - score_df["std_defect_prob"].fillna(0)
        fig_score.add_trace(go.Scatter(
            x=pd.concat([score_df["prediction_date"], score_df["prediction_date"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(29,158,117,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±1σ band",
        ))
    fig_score.add_hline(y=0.07, line_dash="dot", line_color=GRAY,
                         annotation_text="Baseline 7%")
    fig_score.update_layout(
        **PLOTLY_LAYOUT, height=280,
        xaxis_title="Date", yaxis_title="Avg P(Fail)",
        yaxis_range=[0, min(0.5, score_df["avg_defect_prob"].max() * 2 + 0.05)],
    )
    st.plotly_chart(fig_score, use_container_width=True)
else:
    st.info("No prediction score data yet. Run batch inference first.")

st.divider()

# ─── Sensor detail comparison ─────────────────────────────────────────────────
st.subheader(f"Sensor {sensor_select} — current vs Phase I baseline")

sensor_current = load_current_sensor_window(sensor_select, lookback_days)
sensor_ref     = stats_df[
    (stats_df["sensor_id"] == sensor_select) & (stats_df["line_id"] == "LINE_A")
]

if not sensor_current.empty and not sensor_ref.empty:
    ref_mu    = float(sensor_ref.iloc[0]["mu"])
    ref_sigma = float(sensor_ref.iloc[0]["sigma"])
    ref_ucl   = float(sensor_ref.iloc[0]["ucl"])
    ref_lcl   = float(sensor_ref.iloc[0]["lcl"])

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Histogram(
        x=sensor_current["val"].dropna(),
        name=f"Current ({lookback_days}d)",
        marker_color=BLUE, opacity=0.65, nbinsx=40,
        histnorm="probability density",
    ))

    x_range = np.linspace(ref_lcl * 0.9, ref_ucl * 1.1, 200)
    from scipy.stats import norm
    y_ref = norm.pdf(x_range, ref_mu, ref_sigma)
    fig_comp.add_trace(go.Scatter(
        x=x_range, y=y_ref,
        name="Phase I baseline (N(μ,σ))",
        line=dict(color=TEAL, width=2, dash="dash"),
    ))
    fig_comp.add_vline(x=ref_ucl, line_dash="dot", line_color=RED,
                        annotation_text="UCL")
    fig_comp.add_vline(x=ref_lcl, line_dash="dot", line_color=RED,
                        annotation_text="LCL")

    psi_val = _compute_psi(sensor_current["val"], ref_mu, ref_sigma)
    psi_color = "red" if psi_val > psi_threshold else ("orange" if psi_val > 0.10 else "green")
    st.markdown(
        f"PSI = **{psi_val:.4f}** — "
        f":{psi_color}[{'Alert ⚠️' if psi_val > psi_threshold else 'Stable ✓'}]"
    )

    fig_comp.update_layout(
        **PLOTLY_LAYOUT, height=300, barmode="overlay",
        xaxis_title=f"Sensor {sensor_select} value",
        yaxis_title="Density",
    )
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info(f"No current data available for sensor {sensor_select}.")

st.divider()

# ─── Evidently report link ────────────────────────────────────────────────────
st.subheader("Evidently drift report")

try:
    r = requests.get(f"{SERVING_URL}/drift-report", timeout=5)
    info = r.json()
    if info.get("report_available"):
        st.info(
            f"Latest report: `{info.get('latest_report')}` "
            f"({info.get('report_count', 0)} total reports in MinIO). "
            "Download via MinIO console → evidently-reports bucket.",
            icon="📄"
        )
    else:
        st.warning("No Evidently reports found yet. Reports are generated by the daily monitoring DAG.")
except Exception:
    st.warning("ML serving offline — cannot fetch drift report URL.")