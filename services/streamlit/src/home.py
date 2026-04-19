"""
home.py — Executive Overview
────────────────────────────
KPI headline metrics, pipeline health, yield sparkline,
active SPC alarms, and quarantine rate. Auto-refreshes every 30 s.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from common.utils import (
    apply_page_config, badge, query_trino, badge
    PLOTLY_LAYOUT, TEAL, AMBER, RED, CORAL, GRAY,
)

apply_page_config("Executive Overview", "🏭")
 
st.title("🏭 SECOM Manufacturing Command Center")
st.caption(f"Last render: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

@st.cache_data(ttl=30, show_spinner=False)
def load_executive_data():
    try:
        kpis = query_trino("""
            SELECT * FROM gold_daily_yield_metrics
            ORDER BY process_date DESC LIMIT 1
        """)
        trend = query_trino("""
            SELECT process_date, yield_percentage, ppm_defective, total_wafers_tested
            FROM gold_daily_yield_metrics
            ORDER BY process_date DESC LIMIT 30
        """)
        alarms = query_trino("""
            SELECT sensor_id, rule_name, COUNT(*) AS alarm_count, MAX(process_timestamp) AS last_seen
            FROM gold_spc_violations
            WHERE process_timestamp >= NOW() - INTERVAL '24' HOUR
            GROUP BY sensor_id, rule_name
            ORDER BY alarm_count DESC
            LIMIT 10
        """)
        quarantine = query_trino("""
            SELECT
                SUM(quarantined_wafers) AS total_quarantined,
                SUM(wafers_tested + quarantined_wafers) AS total_input
            FROM gold_shift_metrics
            WHERE process_date = CURRENT_DATE
        """)
        oee_today = query_trino("""
            SELECT line_id, oee_pct, availability_pct, performance_pct, quality_pct
            FROM gold_oee_metrics
            WHERE process_date = CURRENT_DATE
        """)
        return kpis, trend, alarms, quarantine, oee_today
    except Exception as e:
        st.error(f"Trino query failed: {e}")
        return (
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame()
        )
 
 
@st.fragment(run_every="30s")
def render_dashboard():
    kpis, trend, alarms, quarantine, oee_today = load_executive_data()
 
    # ------------------------------------------------------------------
    # Active alarm banner
    # ------------------------------------------------------------------
    if not alarms.empty:
        total_alarms = len(alarms)
        st.markdown(
            f'<div style="background:#3d0f0f;border:1px solid #A32D2D;border-radius:8px;'
            f'padding:0.75rem 1rem;margin-bottom:1rem">'
            f'⚠️ &nbsp;<b>{total_alarms}</b> active SPC alarm{"s" if total_alarms > 1 else ""} '
            f'in the last 24 hours — see SPC Monitor for details.</div>',
            unsafe_allow_html=True,
        )
 
    # ------------------------------------------------------------------
    # Top KPI row
    # ------------------------------------------------------------------
    col1, col2, col3, col4, col5, col6 = st.columns(6)
 
    if not kpis.empty:
        row = kpis.iloc[0]
        yield_pct  = float(row.get("yield_percentage", 0))
        dppm       = float(row.get("ppm_defective", 0))
        wafers     = int(row.get("total_wafers_tested", 0))
 
        # Delta vs previous day
        if len(trend) >= 2:
            prev = trend.iloc[1]
            yield_delta = round(yield_pct - float(prev["yield_percentage"]), 2)
            dppm_delta  = round(dppm  - float(prev["ppm_defective"]), 0)
        else:
            yield_delta = dppm_delta = None
 
        col1.metric("Yield %",        f"{yield_pct:.2f}%", f"{yield_delta:+.2f}%" if yield_delta is not None else None)
        col2.metric("DPPM",           f"{dppm:,.0f}",      f"{dppm_delta:+.0f}"   if dppm_delta  is not None else None)
        col3.metric("Wafers Tested",  f"{wafers:,}")
    else:
        col1.metric("Yield %",  "—")
        col2.metric("DPPM",     "—")
        col3.metric("Wafers",   "—")
 
    # Quarantine rate
    if not quarantine.empty and quarantine.iloc[0]["total_input"]:
        qr = quarantine.iloc[0]
        qrate = round(int(qr["total_quarantined"]) / int(qr["total_input"]) * 100, 2) if qr["total_input"] else 0
        col4.metric("Scrap / Quarantine", f"{qrate:.2f}%")
    else:
        col4.metric("Scrap / Quarantine", "—")
 
    col5.metric("SPC Alarms (24h)", len(alarms) if not alarms.empty else 0)
 
    # Data pipeline health — check if today has data
    if not trend.empty:
        latest_date = pd.to_datetime(trend.iloc[0]["process_date"])
        lag_days = (datetime.now().date() - latest_date.date()).days
        health_str = "Live ✓" if lag_days == 0 else f"{lag_days}d lag"
        col6.metric("Pipeline Health", health_str)
    else:
        col6.metric("Pipeline Health", "No data")
 
    st.divider()
 
    # ------------------------------------------------------------------
    # Yield sparkline + DPPM trend
    # ------------------------------------------------------------------
    left, right = st.columns([3, 2])
 
    with left:
        st.subheader("Yield trend — last 30 days")
        if not trend.empty:
            trend_sorted = trend.sort_values("process_date")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_sorted["process_date"],
                y=trend_sorted["yield_percentage"],
                mode="lines+markers",
                name="Yield %",
                line=dict(color=TEAL, width=2),
                fill="tozeroy",
                fillcolor=f"rgba(29,158,117,0.1)",
                marker=dict(size=5),
            ))
            # UCL/LCL reference bands
            mean_y = float(trend_sorted["yield_percentage"].mean())
            fig.add_hline(y=mean_y, line_dash="dot", line_color=GRAY,
                          annotation_text=f"Avg {mean_y:.1f}%", annotation_position="left")
            fig.update_layout(
                **PLOTLY_LAYOUT, height=280,
                xaxis_title="Date", yaxis_title="Yield %",
                yaxis_range=[max(0, trend_sorted["yield_percentage"].min() - 5), 100],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No yield data available yet.")
 
    with right:
        st.subheader("DPPM trend")
        if not trend.empty:
            trend_sorted = trend.sort_values("process_date")
            fig2 = go.Figure(go.Bar(
                x=trend_sorted["process_date"],
                y=trend_sorted["ppm_defective"],
                marker_color=[RED if v > 10000 else AMBER if v > 5000 else TEAL
                               for v in trend_sorted["ppm_defective"]],
                name="DPPM",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=280,
                               xaxis_title="Date", yaxis_title="DPPM")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No DPPM data available yet.")
 
    st.divider()
 
    # ------------------------------------------------------------------
    # Line OEE summary + Active alarms table
    # ------------------------------------------------------------------
    left2, right2 = st.columns([2, 3])
 
    with left2:
        st.subheader("OEE by line (today)")
        if not oee_today.empty:
            for _, row in oee_today.iterrows():
                oee_val = float(row["oee_pct"])
                color   = TEAL if oee_val >= 85 else (AMBER if oee_val >= 65 else RED)
                st.markdown(
                    f"**{row['line_id']}** &nbsp; "
                    f"<span style='font-size:20px;font-weight:500;color:{color}'>{oee_val:.1f}%</span>"
                    f" &nbsp; A:{row['availability_pct']:.0f}% "
                    f"P:{row['performance_pct']:.0f}% "
                    f"Q:{row['quality_pct']:.0f}%",
                    unsafe_allow_html=True,
                )
        else:
            st.info("OEE data not yet available. Run at least one full day of simulation.")
 
    with right2:
        st.subheader("Active SPC alarms")
        if not alarms.empty:
            alarms_disp = alarms.copy()
            alarms_disp["last_seen"] = pd.to_datetime(alarms_disp["last_seen"]).dt.strftime("%H:%M")
            st.dataframe(
                alarms_disp.rename(columns={
                    "sensor_id":   "Sensor",
                    "rule_name":   "Rule",
                    "alarm_count": "Count",
                    "last_seen":   "Last seen",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No SPC alarms in the last 24 hours.")
 
 
render_dashboard()

# ------------------------------------------------------------------
# Sidebar navigation hint
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("""
    | Page | Purpose |
    |------|---------|
    | Yield Analytics | Trends, Pareto, calendar heatmap |
    | SPC Monitor | Nelson rules, Cp/Cpk |
    | OEE Equipment | Line efficiency breakdown |
    | Failure Analysis | Pareto, correlation heatmap |
    | ML Insights | Predictive yield, drift |
    | Simulator | Multi-line generator control |
    """)