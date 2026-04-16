"""
06_Simulator.py — Multi-Line Simulator & Data Validation
──────────────────────────────────────────────────────────
Controls the three-line generator daemon (LINE_A, LINE_B, LINE_C).
Each line has its own run toggle, batch size, interval, jitter,
and independent drift injection for targeted anomaly simulation.
Also includes a data quality validator comparing generated batches
against the baseline SECOM dataset.
"""

import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import wasserstein_distance

from common.config import LineConfig, ServiceConfig, SimulationConfig, StateStore
from common.utils import (
    AMBER, BLUE, CORAL, GRAY, PLOTLY_LAYOUT, RED, TEAL, PURPLE,
    apply_page_config, get_s3_filesystem, get_latest_generated_batch
)

apply_page_config("Simulator", "🏗️")
st.title("🏗️ Multi-Line Synthetic Data Generator")


service_config = ServiceConfig()
# Constants
REL_WASSERSTEIN_THRES = 0.1
RAW_DATA_PATH = service_config.raw_dataset_file
DB_PATH = service_config.db_path
store = StateStore(DB_PATH)

@st.cache_data(ttl=3600)
def load_baseline():
    # Load your raw dataset for comparison
    return pd.read_csv(RAW_DATA_PATH, parse_dates=['Time'], date_format='%Y-%m-%d %H:%M:%S')

def stop_all_lines_callback():
    """Executes BEFORE the page redraws to safely update session state."""
    current_cfg = store.get_config()
    for lid, l_cfg in current_cfg.lines.items():
        st.session_state[f"run_{lid}"] = False
        l_cfg.is_running = False
        
    store.update_config(current_cfg)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_control, tab_validation, tab_history = st.tabs([
    "🎛️ Line Control Panel", "📊 Data Validation", "📜 Event History"
])

# ---------------------------------------------------------------------------
# tab_control
# ---------------------------------------------------------------------------
with tab_control:
    config: SimulationConfig = store.get_config()

    st.subheader("Production line controls")
    st.caption(
        "Each line runs independently. Changes are written to SQLite and "
        "picked up by the generator daemon within one polling cycle."
    )

    LINE_COLORS = {"LINE_A": TEAL, "LINE_B": BLUE, "LINE_C": AMBER}

    updated_lines = {}
    for line_id, lc in config.lines.items():
        color = LINE_COLORS.get(line_id, GRAY)
        st.markdown(
            f"<div style='border-left:3px solid {color};padding-left:1rem;margin-bottom:0.5rem'>"
            f"<b style='color:{color}'>{line_id}</b> — Tester: {lc.tester_id}</div>",
            unsafe_allow_html=True,
        )
    
        col1, col2, col3 = st.columns([1, 2, 2])

        with col1:
            is_running = st.toggle("Active", value=lc.is_running, key=f"run_{line_id}")
            fault_en = st.checkbox("Fault injection", value=lc.fault_injection_enabled, key=f"fault_{line_id}")

        with col2:
            batch_size = st.slider("Batch size (wafers)", 5, 100, lc.batch_size, 5, key=f"batch_{line_id}")
            interval = st.number_input("Interval (s)", 5, 300, lc.generation_interval_seconds, key=f"interval_{line_id}")

        with col3:
            jitter = st.slider("Jitter variance", 0.0, 0.15, lc.jitter_variance, 0.01, key=f"jitter_{line_id}")
            fault_prob = st.slider("Fault probability", 0.0, 0.3, lc.fault_probability, 0.01, key=f"fprob_{line_id}",disabled=not fault_en)

        # Drift config for this line
        with st.expander(f"🌡 Drift / anomaly injection — {line_id}"):
            st.caption("Inject sigma shifts to specific sensors to simulate degradation on this line.")
            current_drift = lc.drift_config.copy()
 
            d_col1, d_col2, d_col3 = st.columns([2, 2, 1])
            new_feat  = d_col1.text_input("Feature (e.g. '59')", key=f"feat_{line_id}")
            new_sigma = d_col2.slider("Sigma shift", -5.0, 5.0, 0.0, 0.5, key=f"sigma_{line_id}")
            if d_col3.button("Add", key=f"add_{line_id}") and new_feat:
                current_drift[new_feat] = new_sigma
 
            if st.button("Clear all drift", key=f"clear_{line_id}"):
                current_drift = {}
 
            if current_drift:
                for feat, sigma in list(current_drift.items()):
                    r1, r2 = st.columns([3, 1])
                    r1.markdown(f"`{feat}` → {sigma:+.1f}σ")
                    if r2.button("Remove", key=f"rm_{line_id}_{feat}"):
                        del current_drift[feat]
 
        updated_lines[line_id] = LineConfig(
            line_id=line_id,
            tester_id=lc.tester_id,
            is_running=is_running,
            batch_size=int(batch_size),
            generation_interval_seconds=int(interval),
            jitter_variance=round(jitter, 3),
            drift_config=current_drift,
            fault_injection_enabled=fault_en,
            fault_probability=round(fault_prob, 3),
            fault_duration_seconds=lc.fault_duration_seconds,
            date_ptr=lc.date_ptr,       
            lot_counter=lc.lot_counter, 
            year_offset=lc.year_offset,
        )

        st.markdown("---")

    # Global save
    col_save, col_stop = st.columns([1, 1])
    if col_save.button("💾 Save all configurations", type="primary", use_container_width=True):
        new_config = SimulationConfig(lines=updated_lines)
        store.update_config(new_config)
        st.toast("✅ Configuration saved — daemon will update within one cycle.")
        time.sleep(1.5)
        st.rerun()

    col_stop.button(
        "⛔ Stop all lines", 
        use_container_width=True, 
        on_click=stop_all_lines_callback
    )

    # Live status display
    st.markdown("#### Current status")

    col_stat1, col_stat2 = st.columns([4, 1])
    col_stat1.caption("Background daemon stats. Click refresh to poll the latest database updates.")
    if col_stat2.button("🔄 Refresh Status", use_container_width=True):
        st.rerun() # Forces Streamlit to pull the newest DB data

    live_cfg = store.get_config()
    status_cols = st.columns(3)
    for col, (lid, lc) in zip(status_cols, live_cfg.lines.items()):
        color = LINE_COLORS.get(lid, GRAY)
        status = "🟢 RUNNING" if lc.is_running else "⭕ IDLE"
        col.markdown(
            f"<div style='background:#161B22;border:1px solid #30363D;border-radius:8px;"
            f"padding:0.75rem;border-left:3px solid {color}'>"
            f"<div style='font-size:12px;color:#8B949E'>{lid}</div>"
            f"<div style='font-size:16px;font-weight:500'>{status}</div>"
            f"<div style='font-size:12px;color:#8B949E'>Batch: {lc.batch_size} wafers "
            f"every {lc.generation_interval_seconds}s</div>"
            f"<div style='font-size:12px;color:#EF9F27'>"
            f"Drift: {json.dumps(lc.drift_config) if lc.drift_config else 'None'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        
# ---------------------------------------------------------------------------
# tab_validation
# ---------------------------------------------------------------------------
with tab_validation:
    st.subheader("Data quality validation — latest batch vs baseline")
    st.caption(
        f"Compares the distribution of the most recently generated batch against "
        f"the raw UCI SECOM baseline. Relative Wasserstein > {REL_WASSERSTEIN_THRES} = drift detected."
    )
    
    line_sel_v = st.selectbox("Validate line", ["All", "LINE_A", "LINE_B", "LINE_C"])
    refresh = st.button("🔄 Refresh validation")

    fs = get_s3_filesystem()
    line_arg = None if line_sel_v == "All" else line_sel_v
    latest_df = None

    try:
        latest_df = get_latest_generated_batch(fs, line_id=line_arg)
    except Exception as e:
        st.error(f"Could not load latest batch: {e}")
    
    if latest_df is None:
        st.warning("No generated data found. Start the generator from the Control Panel tab.")
    else:
        baseline_df = load_baseline()

        st.success(
            f"Loaded batch: {len(latest_df):,} wafers | "
            f"Generated at: {latest_df.get('generation_timestamp', pd.Series(['unknown'])).iloc[0]}"
        )

        drift_info = latest_df.get("applied_drift_features", pd.Series(["{}"])).iloc[0]
        st.markdown(f"**Active drift on this batch:** `{drift_info}`")

        batch_dates = latest_df['Time'].dt.date.unique()
        matched_baseline_df = baseline_df[baseline_df['Time'].dt.date.isin(batch_dates)]
        
        # Shared feature intersection
        numeric_base = matched_baseline_df.select_dtypes(include=["float64", "int64"]).columns
        numeric_latest = latest_df.select_dtypes(include=["float64", "int64"]).columns
        shared = [c for c in numeric_base if c in numeric_latest and len(c) <= 5][:20]

        if matched_baseline_df.empty:
            st.error("Could not match the generated batch dates to the baseline dataset.")
            st.stop()

        # Wasserstein drift scores
        results = []
        for feat in shared:
            base_data   = matched_baseline_df[feat].dropna()
            latest_data = latest_df[feat].dropna()
            if len(base_data) < 5 or len(latest_data) < 5:
                continue
            base_std = float(base_data.std())
            if base_std == 0:
                continue
            dist = wasserstein_distance(base_data.values, latest_data.values)
            rel  = dist / base_std
            results.append({
                "Feature": f"Sensor {feat}",
                "Drift Score": round(rel, 4),
                "Status": "🚨 Drifted" if rel > REL_WASSERSTEIN_THRES else "✅ Stable",
                "Drifted": rel > REL_WASSERSTEIN_THRES,
            })
        
        drift_result = pd.DataFrame(results).sort_values("Drift Score", ascending=False)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Feature drift detection")
            st.dataframe(
                drift_result[["Feature", "Drift Score", "Status"]],
                use_container_width=True, hide_index=True,
            )
 
        with col_b:
            st.markdown("### Missing value topology")
            base_nulls   = matched_baseline_df[shared].isnull().mean() * 100
            latest_nulls = latest_df[shared].isnull().mean() * 100
            null_df = pd.DataFrame({
                "Baseline %":  base_nulls,
                "Generated %": latest_nulls,
            }).reset_index().rename(columns={"index": "Feature"})
            fig = px.bar(
                null_df, x="Feature", y=["Baseline %", "Generated %"],
                barmode="group",
                color_discrete_map={"Baseline %": GRAY, "Generated %": TEAL},
                labels={"value": "% Missing"},
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig, use_container_width=True)
 
        # Distribution comparison for top-drifted sensor
        if len(drift_result) > 0:
            top_feat_raw = drift_result.iloc[0]["Feature"].replace("Sensor ", "")
            st.markdown(f"#### Distribution comparison — {drift_result.iloc[0]['Feature']}")
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=baseline_df[top_feat_raw].dropna(), name="Baseline",
                marker_color=GRAY, opacity=0.6, nbinsx=40,
            ))
            fig2.add_trace(go.Histogram(
                x=latest_df[top_feat_raw].dropna(), name="Generated",
                marker_color=TEAL, opacity=0.7, nbinsx=40,
            ))
            fig2.update_layout(
                **PLOTLY_LAYOUT, barmode="overlay", height=300,
                xaxis_title=drift_result.iloc[0]["Feature"], yaxis_title="Count",
            )
            st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# tab_history
# ---------------------------------------------------------------------------
with tab_history:
    st.subheader("Generator event log")
    n_hours = st.slider("Show last N hours", 1, 168, 24)

    try:
        events = store.get_events(hours=n_hours)

        if events:
            ev_df = pd.DataFrame(events)
            ev_df["ts"] = pd.to_datetime(ev_df["ts"])
 
            # Summary
            c1, c2, c3 = st.columns(3)
            batch_evts = ev_df[ev_df["event_type"] == "BATCH"]
            fault_evts = ev_df[ev_df["event_type"] == "FAULT"]
            c1.metric("Total batches generated", len(batch_evts))
            c2.metric("Fault injections", len(fault_evts))
            c3.metric("Active lines",ev_df[ev_df["event_type"]=="BATCH"]["line_id"].nunique())

            # Batch count by line
            if not batch_evts.empty:
                batch_by_line = batch_evts.groupby("line_id").size().reset_index(name="batches")
                fig = px.bar(
                    batch_by_line, x="line_id", y="batches",
                    color="line_id",
                    color_discrete_map={"LINE_A": TEAL, "LINE_B": BLUE, "LINE_C": AMBER},
                    title="Batches generated per line",
                    labels={"batches": "Batch count", "line_id": "Line"},
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
 
            st.dataframe(
                ev_df[["ts","line_id","event_type","payload"]]
                .sort_values("ts", ascending=False)
                .rename(columns={"ts":"Time","line_id":"Line",
                                 "event_type":"Event","payload":"Detail"})
                .head(200),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No events in the selected window. Start the generator to begin.")
    except Exception as e:
        st.error(f"Could not load event log: {e}")