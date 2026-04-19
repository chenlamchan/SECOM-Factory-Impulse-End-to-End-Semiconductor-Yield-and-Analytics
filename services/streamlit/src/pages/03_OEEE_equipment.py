"""
03_OEE_Equipment.py — OEE & Equipment Effectiveness
─────────────────────────────────────────────────────
Per-line OEE gauges (Availability × Performance × Quality),
shift-level throughput, and a timeline of line activity.
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from common.utils import (
    AMBER, BLUE, CORAL, GRAY, PLOTLY_LAYOUT, RED, TEAL, PURPLE,
    OEEEngine, apply_page_config, query_trino, badge,
)

apply_page_config("OEE / Equipment", "⚙️")
st.title("⚙️ Overall Equipment Effectiveness (OEE)")

st.caption(
    "OEE = Availability × Performance × Quality. "
    "World-class target ≥ 85%. Computed from multi-line shift data."
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    date_range = st.selectbox("Date window", ["Latest Day", "Last 7 days", "Last 30 days"], index=1)
    line_filter = st.multiselect("Lines", ["LINE_A", "LINE_B", "LINE_C"],
                                 default=["LINE_A", "LINE_B", "LINE_C"])

date_interval = {"Latest Day": "0", "Last 7 days": "7", "Last 30 days": "30"}[date_range]


@st.cache_data(ttl=60, show_spinner=False)
def load_oee(interval: str):

    oee = query_trino(f"""
        WITH LatestDates AS (
            SELECT line_id, MAX(process_date) as max_date
            FROM gold_oee_metrics
            GROUP BY line_id
        )

        SELECT m.process_date, m.line_id, m.tester_id, m.oee_pct,
               m.availability_pct, m.performance_pct, m.quality_pct,
               m.oee_classification, m.total_wafers_tested,
               m.total_passed, m.total_failed, m.total_lots
        FROM gold_oee_metrics m
        LEFT JOIN LatestDates ld 
            ON m.line_id = ld.line_id
        WHERE m.process_date >= ld.max_date - INTERVAL '{interval}' DAY
        ORDER BY m.process_date DESC, m.line_id
    """)
    
    shifts = query_trino(f"""
        WITH LatestDates AS (
            SELECT line_id, MAX(process_date) as max_date
            FROM gold_shift_metrics
            GROUP BY line_id
        )
        SELECT m.process_date, m.line_id, m.shift, m.shift_order,
               m.wafers_tested, m.passed_wafers, m.failed_wafers,
               m.quarantined_wafers, m.yield_pct, m.ppm_defective
        FROM gold_shift_metrics m
        LEFT JOIN LatestDates 
            ld ON m.line_id = ld.line_id
        WHERE m.process_date >= ld.max_date - INTERVAL '{interval}' DAY
        ORDER BY m.process_date, m.line_id, m.shift_order
    """)
    return oee, shifts


with st.spinner("Loading OEE data …"):
    oee_df, shifts_df = load_oee(date_interval)

if not line_filter:
    st.warning("Select at least one line.")
    st.stop()

if not oee_df.empty:
    oee_df = oee_df[oee_df["line_id"].isin(line_filter)]
if not shifts_df.empty:
    shifts_df = shifts_df[shifts_df["line_id"].isin(line_filter)]

# ---------------------------------------------------------------------------
# Latest OEE snapshot (most recent date per line)
# ---------------------------------------------------------------------------
st.subheader(f"OEE snapshot — {date_range.lower()}")

if not oee_df.empty:
    latest_oee = (
        oee_df.sort_values("process_date", ascending=False)
        .groupby("line_id").first().reset_index()
    )
    # OEE gauges
    gauge_cols = st.columns(len(latest_oee))
    for col, (_, row) in zip(gauge_cols, latest_oee.iterrows()):
        with col:
            st.plotly_chart(
                OEEEngine.oee_gauge(float(row["oee_pct"]), row["line_id"]),
                use_container_width=True,
            )
            st.markdown(
                f"<div style='text-align: center; margin-top: -10px;'>"
                f"{OEEEngine.classification_badge(float(row['oee_pct']))}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # A / P / Q breakdown
    st.markdown("#### Availability · Performance · Quality decomposition")
    apq_rows = []
    for _, row in latest_oee.iterrows():
        for comp, val in [
            ("Availability", row["availability_pct"]),
            ("Performance",  row["performance_pct"]),
            ("Quality",      row["quality_pct"]),
        ]:
            apq_rows.append({"Line": row["line_id"], "Component": comp, "Value": float(val)})

    apq_df = pd.DataFrame(apq_rows)
    fig_apq = px.bar(
        apq_df, x="Line", y="Value", color="Component", barmode="group",
        color_discrete_map={"Availability": TEAL, "Performance": BLUE, "Quality": AMBER},
        text_auto=".1f",
    )
    fig_apq.add_hline(y=85, line_dash="dot", line_color=RED,
                      annotation_text="World-class 85%", annotation_position="right")
    fig_apq.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_range=[0, 110])
    st.plotly_chart(fig_apq, use_container_width=True)

else:
    st.info("No OEE data found. Ensure the multi-line generator has run for at least one day "
            "and dbt models have been executed.")

st.divider()

# ---------------------------------------------------------------------------
# OEE trend over time
# ---------------------------------------------------------------------------
if not oee_df.empty and len(oee_df["process_date"].unique()) > 1:
    st.subheader("OEE trend over time")
    trend_fig = go.Figure()
    colors = {"LINE_A": TEAL, "LINE_B": BLUE, "LINE_C": AMBER}
    for line in oee_df["line_id"].unique():
        ld = oee_df[oee_df["line_id"] == line].sort_values("process_date")
        trend_fig.add_trace(go.Scatter(
            x=ld["process_date"], y=ld["oee_pct"],
            name=line, mode="lines+markers",
            line=dict(color=colors.get(line, GRAY), width=2),
        ))
    trend_fig.add_hline(y=85, line_dash="dot", line_color=RED,
                        annotation_text="Target 85%", annotation_position="right")
    trend_fig.update_layout(
        **PLOTLY_LAYOUT, height=320,
        yaxis_range=[0, 100], yaxis_title="OEE %", xaxis_title="Date",
        legend=dict(
                orientation="h",    
                yanchor="top",      
                y=-1.0,            
                xanchor="center",   
                x=0.5               
            )
    )
    st.plotly_chart(trend_fig, use_container_width=True)
    st.divider()