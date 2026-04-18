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
import time

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
    window_days = st.selectbox("Trend window", [7, 14, 30, 90, 180, 365], index=3)
    lines_filter = st.multiselect("Production lines", ["LINE_A", "LINE_B", "LINE_C"],
                                  default=["LINE_A", "LINE_B", "LINE_C"])

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def load_yield_data(days: int, lines:list):

    if len(lines) == 1:
        sql_lines = f"('{lines[0]}')"
    else:
        sql_lines = str(tuple(lines))

    # 1. Daily KPIs (Dynamic aggregation across selected lines)
    daily = query_trino(f"""
        WITH LineMaxDate AS (
            SELECT line_id, MAX(process_date) AS max_date 
            FROM gold_daily_yield_metrics 
            GROUP BY line_id
        )

        SELECT 
            d.process_date, 
            d.line_id,
            SUM(d.passed_wafers) / CAST(NULLIF(SUM(d.total_wafers_tested), 0) AS DOUBLE) * 100 AS yield_percentage, 
            SUM(d.failed_wafers) / CAST(NULLIF(SUM(d.total_wafers_tested), 0) AS DOUBLE) * 1000000 AS ppm_defective,
            SUM(d.total_wafers_tested) AS total_wafers_tested, 
            SUM(d.quarantined_wafers) AS quarantined_wafers,
            SUM(d.passed_wafers) AS passed_wafers, 
            SUM(d.failed_wafers) AS failed_wafers
        FROM gold_daily_yield_metrics d
        JOIN LineMaxDate m ON d.line_id = m.line_id
        WHERE d.process_date >= m.max_date - INTERVAL '{days}' DAY
        GROUP BY d.process_date, d.line_id
        ORDER BY d.process_date DESC, d.line_id
    """)

    # 2. Shift Data (Filtered by lines)
    shift = query_trino(f"""
        WITH LineMaxDate AS (
            SELECT line_id, MAX(process_date) AS max_date
            FROM gold_shift_metrics 
            WHERE line_id IN {sql_lines}
            GROUP BY line_id
        )

        SELECT s.process_date, s.line_id, s.shift, s.wafers_tested,
               s.passed_wafers, s.failed_wafers, s.quarantined_wafers,
               s.yield_pct, s.ppm_defective, s.scrap_rate_pct
        FROM gold_shift_metrics s
        JOIN LineMaxDate m ON s.line_id = m.line_id
        WHERE s.line_id IN {sql_lines}
          AND s.process_date >= m.max_date - INTERVAL '{days}' DAY
        ORDER BY s.process_date DESC, s.shift_order
    """)

    pareto = query_trino("""
        SELECT sensor_id, abs_correlation, effect_strength, direction
        FROM gold_failure_pareto
        ORDER BY rank LIMIT 15
    """)

    # (Dynamic aggregation for a N-day view)
    calendar = query_trino(f"""
        WITH LineMaxDate AS (
            SELECT line_id, MAX(process_date) AS max_date
            FROM gold_daily_yield_metrics
            WHERE line_id IN {sql_lines}
            GROUP BY line_id
        )

        SELECT 
            d.process_date, 
            SUM(d.passed_wafers) / CAST(NULLIF(SUM(d.total_wafers_tested), 0) AS DOUBLE) * 100 AS yield_percentage, 
            SUM(d.total_wafers_tested) AS total_wafers_tested
        FROM gold_daily_yield_metrics d
        JOIN LineMaxDate m ON d.line_id = m.line_id
        WHERE d.line_id IN {sql_lines}
          AND d.process_date >= m.max_date - INTERVAL '{days}' DAY
        GROUP BY d.process_date
        ORDER BY d.process_date DESC
    """)
    return daily, shift, pareto, calendar

if not lines_filter:
    st.warning("Select at least one production line.")
    st.stop()

with st.spinner("Loading yield data …"):
    daily, shift_df, pareto, calendar = load_yield_data(window_days, lines_filter)

# ---------------------------------------------------------------------------
# KPI headline row
# ---------------------------------------------------------------------------
if not daily.empty:
    global_max_date = daily["process_date"].max()
    global_min_date = global_max_date - pd.Timedelta(days=window_days)

    period_df = daily[
        (daily["process_date"] >= global_min_date) & 
        (daily["process_date"] <= global_max_date)
    ]

    # 1. Compute overall aggregate stats from the line-level daily dataframe
    period_agg = period_df.groupby("process_date").agg({
        "total_wafers_tested": "sum",
        "quarantined_wafers": "sum",
        "passed_wafers": "sum",
        "failed_wafers": "sum"
    }).reset_index()

    period_agg["yield_percentage"] = (period_agg["passed_wafers"] / period_agg["total_wafers_tested"].replace(0, np.nan)) * 100
    period_agg["ppm_defective"] = (period_agg["failed_wafers"] / period_agg["total_wafers_tested"].replace(0, np.nan)) * 1000000
    period_agg = period_agg.sort_values("process_date", ascending=False)

    period_total_tested = period_agg["total_wafers_tested"].sum()
    period_passed = period_agg["passed_wafers"].sum()
    period_failed = period_agg["failed_wafers"].sum()
    period_yield = (period_passed / period_total_tested) * 100
    period_dppm = (period_failed / period_total_tested) * 1000000

    start_str = global_min_date.strftime('%Y-%m-%d')
    end_str = global_max_date.strftime('%Y-%m-%d')

    st.subheader("🏭 Factory Overall KPIs (All Lines)")
    st.caption(f"📅 **Data Range:** {start_str} to {end_str}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Overall Yield",
              f"{period_yield:.2f}%")
    c2.metric("Overall DPPM ",
              f"{period_dppm:,.0f}")
    c3.metric("Total Wafers Tested Today", 
              f"{int(period_total_tested):,}")

    st.divider()

# ---------------------------------------------------------------------------
# Yield Trend + DPPM dual-axis
# ---------------------------------------------------------------------------
tab_trend, tab_line, tab_pareto, tab_calendar, tab_waterfall = st.tabs([
    "📉 Overall Yield Trend", "🏭 By Line & Shift", "📊 Failure Pareto",
    "📅 Calendar Heatmap", "🌊 Yield Waterfall",
])

with tab_trend:
    if not period_df.empty:
        line_agg = period_df.groupby(["process_date", "line_id"]).agg({
        "total_wafers_tested": "sum",
        "quarantined_wafers": "sum",
        "passed_wafers": "sum",
        "failed_wafers": "sum"
        }).reset_index()

        line_agg["yield_percentage"] = (line_agg["passed_wafers"] / line_agg["total_wafers_tested"].replace(0, np.nan)) * 100
        line_agg["ppm_defective"] = (line_agg["failed_wafers"] / line_agg["total_wafers_tested"].replace(0, np.nan)) * 1000000
        line_agg = line_agg.sort_values("process_date", ascending=False)

        # Plot individual lines
        fig = px.line(
            line_agg, x="process_date", y="yield_percentage", color="line_id",
            markers=True, color_discrete_sequence=[TEAL, BLUE, AMBER],
            labels={"yield_percentage": "Yield %", "process_date": "Date", "line_id": "Line"}
        )

        fig.add_trace(go.Scatter(
            x=period_agg["process_date"], y=period_agg["yield_percentage"],
            name="Overall", mode="lines",
            line=dict(color=AMBER, dash="dot", width=1.5),
        ))

        # Add Overall DPPM on secondary axis
        fig.add_trace(go.Bar(
            x=period_agg["process_date"], y=period_agg["ppm_defective"],
            name="Overall DPPM", yaxis="y2", opacity=0.3,
            marker_color=RED
        ))

        fig.update_layout(
            **PLOTLY_LAYOUT, height=450,
            title=f"Yield % by Line & Overall DPPM — last {window_days} days",
            yaxis=dict(title="Yield %", range=[0, 100]),
            yaxis2=dict(title="Overall DPPM", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="right", x=1)
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
        st.markdown("#### Top sensors correlated with wafer failure *(Global Factory Data)*")
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
        cal["week"] = cal["process_date"].dt.isocalendar().week.astype(int)
        cal["dow"] = cal["process_date"].dt.dayofweek
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
            title=f"Yield calendar — last {window_days} days",
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
    # Dynamically display which lines are making up this aggregate
    selected_lines_str = ", ".join(lines_filter)
    st.caption(f"Breaks down where yield is lost across the pipeline ({selected_lines_str})")

    wf_df = daily_agg.sort_values("process_date", ascending=False)
    
    if not daily.empty:

        latest = daily.iloc[0]
        process_tested = int(latest["total_wafers_tested"])
        quarantined = int(latest["quarantined_wafers"])
        passed = int(latest["passed_wafers"])
        failed = int(latest["failed_wafers"])

        gross_input = process_tested + quarantined

        labels  = ["Input wafers", "Quarantined (data quality)", "Failed (process)", "Final yield"]
        measure = ["absolute", "relative", "relative", "total"]
        values  = [gross_input, -quarantined, -failed, passed]
        colors  = [BLUE, AMBER, RED, TEAL]

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

        # Sum the data across all days currently in the 'daily' dataframe
        agg_tested = int(daily["total_wafers_tested"].sum())
        agg_quarantined = int(daily["quarantined_wafers"].sum())
        agg_passed = int(daily["passed_wafers"].sum())
        agg_failed = int(daily["failed_wafers"].sum())
        
        agg_gross_input = agg_tested + agg_quarantined

        labels_agg  = ["Input wafers", "Quarantined (data quality)", "Failed (process)", "Final yield"]
        values_agg  = [agg_gross_input, -agg_quarantined, -agg_failed, agg_passed]

        fig2 = go.Figure(go.Waterfall(
            orientation="v", measure=measure, x=labels_agg, y=values_agg,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": TEAL}},
            decreasing={"marker": {"color": RED}},
            totals={"marker": {"color": BLUE}},
            text=[f"{abs(v):,}" for v in values_agg], textposition="outside",
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=360,
                            title=f"Aggregate (Past {window_days} Days)", yaxis_title="Wafer count")
        st.plotly_chart(fig2, use_container_width=True)
                
    else:
        st.info("No data for waterfall chart.")