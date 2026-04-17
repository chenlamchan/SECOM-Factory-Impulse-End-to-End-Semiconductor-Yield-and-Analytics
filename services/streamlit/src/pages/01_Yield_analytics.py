"""
01_Yield_Analytics.py — Yield & Quality Analytics
──────────────────────────────────────────────────
Rolling yield trends, DPPM, yield by line/shift, Pareto of losses,
and a GitHub-style calendar heatmap.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common.utils import (
    AMBER, BLUE, CORAL, GRAY, PLOTLY_LAYOUT, RED, TEAL,
    apply_page_config, badge, query_trino,
)

apply_page_config("Yield Analytics", "📈")

st.title("📈 Yield & Quality Analytics")

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    window_days = st.selectbox("Trend window", [7, 14, 30, 90], index=2)
    lines_filter = st.multiselect("Production lines", ["LINE_A", "LINE_B", "LINE_C"],
                                  default=["LINE_A", "LINE_B", "LINE_C"])

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def load_yield_data(days: int):
    daily = query_trino(f"""
        SELECT process_date, yield_percentage, ppm_defective,
               total_wafers_tested, passed_wafers, failed_wafers
        FROM gold_daily_yield_metrics
        ORDER BY process_date DESC LIMIT {days}
    """)
    shift = query_trino(f"""
        SELECT process_date, line_id, shift, wafers_tested,
               passed_wafers, failed_wafers, quarantined_wafers,
               yield_pct, ppm_defective, scrap_rate_pct
        FROM gold_shift_metrics
        WHERE process_date >= (SELECT MAX(process_date) FROM gold_shift_metrics) - INTERVAL '{days}' DAY
        ORDER BY process_date DESC, shift_order
    """)
    pareto = query_trino("""
        SELECT sensor_id, abs_correlation, effect_strength, direction
        FROM gold_failure_pareto
        ORDER BY rank LIMIT 15
    """)
    # 90-day calendar data
    calendar = query_trino("""
        SELECT process_date, yield_percentage, total_wafers_tested
        FROM gold_daily_yield_metrics
        ORDER BY process_date DESC LIMIT 90
    """)
    return daily, shift, pareto, calendar


with st.spinner("Loading yield data …"):
    daily, shift_df, pareto, calendar = load_yield_data(window_days)

if not lines_filter:
    st.warning("Select at least one production line.")
    st.stop()

if shift_df is not None and not shift_df.empty:
    shift_df = shift_df[shift_df["line_id"].isin(lines_filter)]

# ---------------------------------------------------------------------------
# KPI headline row
# ---------------------------------------------------------------------------
if not daily.empty:
    latest = daily.iloc[0]
    prev   = daily.iloc[1] if len(daily) > 1 else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Yield",
              f"{latest['yield_percentage']:.2f}%",
              f"{latest['yield_percentage'] - float(prev['yield_percentage']):+.2f}%" if prev is not None else None)
    c2.metric("DPPM",
              f"{latest['ppm_defective']:,.0f}",
              f"{latest['ppm_defective'] - float(prev['ppm_defective']):+.0f}" if prev is not None else None)
    c3.metric("Wafers Tested Today", f"{int(latest['total_wafers_tested']):,}")
    roll_avg = round(float(daily["yield_percentage"].mean()), 2)
    c4.metric(f"{window_days}-Day Avg Yield", f"{roll_avg:.2f}%")
    st.divider()

# ---------------------------------------------------------------------------
# Yield Trend + DPPM dual-axis
# ---------------------------------------------------------------------------
tab_trend, tab_line, tab_pareto, tab_calendar, tab_waterfall = st.tabs([
    "📉 Yield Trend", "🏭 By Line & Shift", "📊 Failure Pareto",
    "📅 Calendar Heatmap", "🌊 Yield Waterfall",
])

with tab_trend:
    if not daily.empty:
        d = daily.sort_values("process_date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d["process_date"], y=d["yield_percentage"],
            name="Yield %", mode="lines+markers",
            line=dict(color=TEAL, width=2),
            fill="tozeroy", fillcolor="rgba(29,158,117,0.08)",
        ))
        # Rolling average
        d["roll7"] = d["yield_percentage"].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=d["process_date"], y=d["roll7"],
            name="7-day MA", mode="lines",
            line=dict(color=AMBER, dash="dot", width=1.5),
        ))
        # DPPM on secondary axis
        fig.add_trace(go.Bar(
            x=d["process_date"], y=d["ppm_defective"],
            name="DPPM", yaxis="y2", opacity=0.4,
            marker_color=RED,
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=400,
            title=f"Yield % & DPPM — last {window_days} days",
            yaxis=dict(title="Yield %", range=[0, 100]),
            yaxis2=dict(title="DPPM", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily yield data found.")

with tab_line:
    if not shift_df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Yield % by production line")
            line_agg = shift_df.groupby("line_id").agg(
                yield_pct=("yield_pct", "mean")
            ).reset_index()
            fig = px.bar(line_agg, x="line_id", y="yield_pct",
                         color="line_id", color_discrete_sequence=[TEAL, BLUE, AMBER],
                         labels={"yield_pct": "Avg Yield %", "line_id": "Line"},
                         text_auto=".1f")
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                              yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("#### Yield % by shift")
            shift_agg = shift_df.groupby("shift").agg(
                yield_pct=("yield_pct", "mean")
            ).reset_index()
            fig2 = px.bar(shift_agg, x="shift", y="yield_pct",
                          color="shift", color_discrete_sequence=[TEAL, AMBER, CORAL],
                          labels={"yield_pct": "Avg Yield %", "shift": "Shift"},
                          text_auto=".1f")
            fig2.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                               yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Yield by line × shift (heatmap)")
        pivot = shift_df.groupby(["line_id", "shift"])["yield_pct"].mean().reset_index()
        pivot_wide = pivot.pivot(index="line_id", columns="shift", values="yield_pct")
        fig3 = go.Figure(go.Heatmap(
            z=pivot_wide.values,
            x=pivot_wide.columns.tolist(),
            y=pivot_wide.index.tolist(),
            colorscale=[[0, "#A32D2D"], [0.5, "#854F0B"], [1, "#0F6E56"]],
            zmin=0, zmax=100,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "N/A" for v in row]
                  for row in pivot_wide.values],
            texttemplate="%{text}",
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No shift-level data found. Run the multi-line generator to populate.")

with tab_pareto:
    if not pareto.empty:
        st.markdown("#### Top sensors correlated with wafer failure")
        st.caption("Point-biserial |r| — higher = stronger association with fail label")
        fig = go.Figure(go.Bar(
            x=pareto["abs_correlation"],
            y=[f"Sensor {s}" for s in pareto["sensor_id"]],
            orientation="h",
            marker_color=[
                RED if e == "Strong" else AMBER if e == "Moderate" else GRAY
                for e in pareto["effect_strength"]
            ],
            text=[f"|r|={v:.3f}" for v in pareto["abs_correlation"]],
            textposition="outside",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=480,
            xaxis_title="|Correlation|",
            yaxis=dict(autorange="reversed"),
            title="Failure Pareto — sensor correlation with Pass/Fail label",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full table"):
            st.dataframe(pareto, use_container_width=True, hide_index=True)
    else:
        st.info("Failure pareto not yet computed. Run dbt models first.")

with tab_calendar:
    if not calendar.empty:
        cal = calendar.sort_values("process_date").copy()
        cal["process_date"] = pd.to_datetime(cal["process_date"])
        cal["week"]     = cal["process_date"].dt.isocalendar().week.astype(int)
        cal["dow"]      = cal["process_date"].dt.dayofweek
        cal["dow_name"] = cal["process_date"].dt.strftime("%a")

        fig = go.Figure(go.Heatmap(
            x=cal["week"],
            y=cal["dow"],
            z=cal["yield_percentage"],
            text=[f"{d.strftime('%b %d')}<br>Yield: {y:.1f}%"
                  for d, y in zip(cal["process_date"], cal["yield_percentage"])],
            hovertemplate="%{text}<extra></extra>",
            colorscale=[[0, "#A32D2D"], [0.5, "#854F0B"], [1, "#0F6E56"]],
            zmin=0, zmax=100,
            xgap=3, ygap=3,
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=280,
            title="Yield calendar — last 90 days",
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(7)),
                ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                autorange="reversed",
            ),
            xaxis_title="Week of year",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No calendar data yet.")

with tab_waterfall:
    st.markdown("#### Yield loss waterfall")
    st.caption("Breaks down where yield is lost across the pipeline")
    if not daily.empty:
        latest = daily.iloc[0]
        total  = int(latest["total_wafers_tested"]) + 0
        passed = int(latest["passed_wafers"])
        failed = int(latest["failed_wafers"])
        quarantined_est = max(0, total - passed - failed)

        labels  = ["Input wafers", "Quarantined (data quality)", "Failed (process)", "Final yield"]
        measure = ["absolute",     "relative",                  "relative",         "total"]
        values  = [total,          -quarantined_est,            -failed,             passed]
        colors  = [BLUE,           AMBER,                        RED,                TEAL]

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=measure,
            x=labels,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": TEAL}},
            decreasing={"marker": {"color": RED}},
            totals={"marker": {"color": BLUE}},
            text=[f"{abs(v):,}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          title="Wafer yield waterfall — most recent day",
                          yaxis_title="Wafer count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for waterfall chart.")