"""
home.py — Executive Overview
────────────────────────────
KPI headline metrics, pipeline health, yield sparkline,
active SPC alarms, and quarantine rate. Auto-refreshes every 30 s.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from common.utils import (
    apply_page_config, badge, query_trino, badge,
    PLOTLY_LAYOUT, TEAL, AMBER, RED, CORAL, GRAY,
)

apply_page_config("Executive Overview", "🏭")
 
st.title("🏭 SECOM Manufacturing Command Center")

@st.cache_data(ttl=30 )
def load_executive_data():
    try:
        with st.spinner("Waiting for Trino cluster & fetching data..."):
            kpis = query_trino("""
                WITH max_d AS (SELECT MAX(process_date) AS md FROM gold_daily_yield_metrics)
                SELECT 
                    process_date,
                    SUM(total_wafers_tested) AS total_wafers_tested,
                    SUM(passed_wafers) AS passed_wafers,
                    SUM(failed_wafers) AS failed_wafers,
                    (SUM(passed_wafers) / CAST(NULLIF(SUM(total_wafers_tested), 0) AS DOUBLE)) * 100 AS yield_percentage,
                    (SUM(failed_wafers) / CAST(NULLIF(SUM(total_wafers_tested), 0) AS DOUBLE)) * 1000000 AS ppm_defective
                FROM gold_daily_yield_metrics
                WHERE process_date >= (SELECT md - INTERVAL '1' DAY FROM max_d)
                GROUP BY process_date
                ORDER BY process_date DESC
            """)
            
            trend = query_trino("""
                SELECT 
                    process_date, 
                    (SUM(passed_wafers) / CAST(NULLIF(SUM(total_wafers_tested), 0) AS DOUBLE)) * 100 AS yield_percentage,
                    (SUM(failed_wafers) / CAST(NULLIF(SUM(total_wafers_tested), 0) AS DOUBLE)) * 1000000 AS ppm_defective,
                    SUM(total_wafers_tested) AS total_wafers_tested
                FROM gold_daily_yield_metrics
                GROUP BY process_date
                ORDER BY process_date DESC 
                LIMIT 30
            """)

            alarms = query_trino("""
                WITH max_t AS (SELECT MAX(process_timestamp) AS max_ts FROM gold_spc_violations)
                SELECT sensor_id, rule_name, COUNT(*) AS alarm_count, MAX(process_timestamp) AS last_seen
                FROM gold_spc_violations
                WHERE process_timestamp >= (SELECT max_ts FROM max_t) - INTERVAL '24' HOUR
                GROUP BY sensor_id, rule_name
                ORDER BY alarm_count DESC
                LIMIT 10
            """)

            quarantine = query_trino("""
                WITH max_d AS (SELECT MAX(process_date) AS md FROM gold_shift_metrics)
                SELECT
                    SUM(quarantined_wafers) AS total_quarantined,
                    SUM(wafers_tested + quarantined_wafers) AS total_input
                FROM gold_shift_metrics
                WHERE process_date = (SELECT md FROM max_d)
            """)

            oee_today = query_trino("""
                WITH max_d AS (SELECT MAX(process_date) AS md FROM gold_oee_metrics)
                SELECT line_id, oee_pct, availability_pct, performance_pct, quality_pct
                FROM gold_oee_metrics
                WHERE process_date = (SELECT md FROM max_d)
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

    if kpis is None or kpis.empty:
        st.error("⚠️ Trino Cluster is currently unavailable or returned no data.")
        st.info("The system is likely scaling up or experiencing high load. Please refresh in a few moments.")
        st.stop()

    max_ts = kpis.iloc[0]['process_date']

    if max_ts is not None and not pd.isna(max_ts):
        ts_str = pd.to_datetime(max_ts).strftime('%Y-%m-%d')
    else:
        ts_str = "No data available"
        
    st.caption(f"Latest Production Timestamp: **{ts_str}**")
 
    # ------------------------------------------------------------------
    # Active alarm banner
    # ------------------------------------------------------------------
    if not alarms.empty:
        total_alarms = len(alarms)
        st.error(
            f"**{total_alarms} active SPC alarm{'s' if total_alarms > 1 else ''}** in the last 24 hours — see SPC Monitor for details.",
            icon="🚨"
        )
 
    st.markdown("<br>", unsafe_allow_html=True) # Adds a little breathing room
 
    # ------------------------------------------------------------------
    # Top KPI row
    # ------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
 
    if not kpis.empty:
        row = kpis.iloc[0]
        yield_pct = float(row.get("yield_percentage", 0))
        dppm = float(row.get("ppm_defective", 0))
        wafers = int(row.get("total_wafers_tested", 0))
 
        # Delta vs previous day
        if len(kpis) > 1:
            yesterday_row = kpis.iloc[1]
            yield_delta = round(yield_pct - float(yesterday_row["yield_percentage"]), 2)
            
            yesterday_dppm = float(yesterday_row["ppm_defective"])
            dppm_delta  = round(((dppm - yesterday_dppm) * 100 / yesterday_dppm) if yesterday_dppm != 0 else 0, 2)
        else:
            yield_delta = dppm_delta = None
 
        col1.metric("Yield Today% (Overall)", f"{yield_pct:.2f}%", f"{yield_delta:+.2f}%" if yield_delta is not None else None, help="Passed wafers / Total tested")
        col2.metric("DPPM Today (Overall)", f"{dppm:,.0f}", f"{dppm_delta:+.2f}" if dppm_delta  is not None else None, delta_color="inverse", help="Defective Parts Per Million")
        col3.metric("Wafers Tested Today", f"{wafers:,}")
    else:
        col1.metric("Yield %", "—")
        col2.metric("DPPM", "—")
        col3.metric("Wafers", "—")
 
    # Quarantine rate
    if not quarantine.empty and quarantine.iloc[0]["total_input"]:
        qr = quarantine.iloc[0]
        qrate = round(int(qr["total_quarantined"]) / int(qr["total_input"]) * 100, 2) if qr["total_input"] else 0
        col4.metric("Scrap / Quarantine", f"{qrate:.2f}%")
    else:
        col4.metric("Scrap / Quarantine", "—")
 
    col5.metric("SPC Alarms (24h)", len(alarms) if not alarms.empty else 0)
 
    st.divider()
 
    # ------------------------------------------------------------------
    # Yield sparkline + DPPM trend
    # ------------------------------------------------------------------
    left, right = st.columns([3, 2])
 
    with left:
        st.subheader("Yield trend — last 30 days")
        if not trend.empty:
            trend_sorted = trend.sort_values("process_date")
            trend_sorted["yield_rolling_7d"] = trend_sorted["yield_percentage"].rolling(window=7, min_periods=1).mean()

            fig = px.scatter(
                trend_sorted, x="process_date", y="yield_percentage",
                color_discrete_sequence=[TEAL], opacity=0.4,
                labels={"yield_percentage": "Yield %", "process_date": "Date"}
            )
            fig.add_trace(go.Scatter(
                x=trend_sorted["process_date"], y=trend_sorted["yield_rolling_7d"],
                mode="lines", name="7-Day Trend",
                line=dict(color=TEAL, width=3)
            ))
            # UCL/LCL reference bands
            mean_y = float(trend_sorted["yield_percentage"].mean())
            fig.add_hline(y=mean_y, line_dash="dot", line_color=GRAY,
                          annotation_text=f"Avg {mean_y:.1f}%", annotation_position="bottom right")
            fig.update_layout(
                **PLOTLY_LAYOUT, height=280,
                xaxis_title="Date", yaxis_title="Yield %",
                yaxis_range=[max(0, trend_sorted["yield_percentage"].min() - 5), 105],
            )

            fig.update_traces(cliponaxis=False)

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
    left2, right2 = st.columns([1, 1])
 
    with left2:
        st.subheader("OEE by line (Latest Day)")
        if not oee_today.empty:
            for _, row in oee_today.iterrows():
                oee_val = float(row["oee_pct"])

                with st.container(border=True):
                    st.markdown(f"##### {row['line_id']}")
                    c1, c2, c3, c4 = st.columns(4)
                    
                    c1.metric("OEE", f"{oee_val:.1f}%")
                    c2.metric("Availability", f"{row['availability_pct']:.0f}%")
                    c3.metric("Performance", f"{row['performance_pct']:.0f}%")
                    c4.metric("Quality", f"{row['quality_pct']:.0f}%")
                
        else:
            st.info("OEE data not yet available. Run at least one full day of simulation.")
 
    with right2:
        st.subheader("Active SPC alarms")
        if not alarms.empty:
            alarms_disp = alarms.copy()
            alarms_disp["last_seen"] = pd.to_datetime(alarms_disp["last_seen"]).dt.strftime("%H:%M")
            
            max_count = int(alarms_disp['alarm_count'].max())
            st.dataframe(
                alarms_disp.rename(columns={
                    "sensor_id": "Sensor",
                    "rule_name": "Rule",
                    "alarm_count": "Count",
                    "last_seen": "Last seen",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Count": st.column_config.ProgressColumn(
                        "Count",
                        help="Number of times rule triggered",
                        format="%d",
                        min_value=0,
                        max_value=max_count + 1,
                    )
                }
            )
        else:
            st.success("No SPC alarms in the last 24 hours.", icon="✅")
 
 
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