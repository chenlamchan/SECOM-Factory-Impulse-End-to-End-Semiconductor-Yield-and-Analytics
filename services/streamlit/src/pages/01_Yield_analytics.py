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

    global_period_df = daily[
        (daily["process_date"] >= global_min_date) & 
        (daily["process_date"] <= global_max_date)
    ]

    # 1. Compute overall aggregate stats from the line-level daily dataframe
    global_period_agg = global_period_df.groupby("process_date").agg({
        "total_wafers_tested": "sum",
        "quarantined_wafers": "sum",
        "passed_wafers": "sum",
        "failed_wafers": "sum"
    }).reset_index()

    global_period_agg["yield_percentage"] = (global_period_agg["passed_wafers"] / global_period_agg["total_wafers_tested"].replace(0, np.nan)) * 100
    global_period_agg["ppm_defective"] = (global_period_agg["failed_wafers"] / global_period_agg["total_wafers_tested"].replace(0, np.nan)) * 1000000
    global_period_agg = global_period_agg.sort_values("process_date", ascending=False)

    global_period_total_tested = global_period_agg["total_wafers_tested"].sum()
    global_period_passed = global_period_agg["passed_wafers"].sum()
    global_period_failed = global_period_agg["failed_wafers"].sum()
    global_period_yield = (global_period_passed / global_period_total_tested) * 100
    global_period_dppm = (global_period_failed / global_period_total_tested) * 1000000

    global_start_str = global_min_date.strftime('%Y-%m-%d')
    global_end_str = global_max_date.strftime('%Y-%m-%d')

    st.subheader("🏭 Factory Overall KPIs (All Lines)")
    st.caption(f"📅 **Data Range:** {global_start_str} to {global_end_str}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Overall Yield",
              f"{global_period_yield:.2f}%")
    c2.metric("Overall DPPM ",
              f"{global_period_dppm:,.0f}")
    c3.metric("Total Wafers Tested", 
              f"{int(global_period_total_tested):,}")

    with st.expander("View Overall Yield Calendar Heatmap", expanded=False):
        if not global_period_agg.empty:
            cal_global = global_period_agg.sort_values("process_date").copy()
            cal_global["process_date"] = pd.to_datetime(cal_global["process_date"])
            cal_global["week"] = cal_global["process_date"].dt.isocalendar().week.astype(int)
            cal_global["dow"] = cal_global["process_date"].dt.dayofweek
            
            fig_cal_global = go.Figure(go.Heatmap(
                x=cal_global["week"],
                y=cal_global["dow"],
                z=cal_global["yield_percentage"],
                text=[f"{d.strftime('%b %d')}<br>Yield: {y:.1f}%"
                      for d, y in zip(cal_global["process_date"], cal_global["yield_percentage"])],
                hovertemplate="%{text}<extra></extra>",
                colorscale=[[0, "#A32D2D"], [0.5, "#854F0B"], [1, "#0F6E56"]],
                zmin=0, zmax=100,
                xgap=3, ygap=3,
            ))
            fig_cal_global.update_layout(
                **PLOTLY_LAYOUT, height=280,
                title=f"Global Yield Calendar ({global_start_str} to {global_end_str})",
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(7)),
                    ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                    autorange="reversed",
                ),
                xaxis_title="Week of year",
            )
            st.plotly_chart(fig_cal_global, use_container_width=True)
        else:
            st.info("No data available to render the global calendar.")

    with st.expander("View Overall Yield Loss Waterfall", expanded=False):
 
        # Aggregate data for the global period
        agg_tested = int(global_period_df["total_wafers_tested"].sum())
        agg_quarantined = int(global_period_df["quarantined_wafers"].sum())
        agg_passed = int(global_period_df["passed_wafers"].sum())
        agg_failed = int(global_period_df["failed_wafers"].sum())
        
        agg_gross_input = agg_tested + agg_quarantined

        labels  = ["Input wafers", "Quarantined (data quality)", "Failed (process)", "Final yield"]
        measure = ["absolute", "relative", "relative", "total"]
        values_agg  = [agg_gross_input, -agg_quarantined, -agg_failed, agg_passed]

        fig_global = go.Figure(go.Waterfall(
            orientation="v", measure=measure, x=labels, y=values_agg,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": TEAL}},
            decreasing={"marker": {"color": RED}},
            totals={"marker": {"color": BLUE}},
            text=[f"{abs(v):,}" for v in values_agg], textposition="outside",
        ))
        fig_global.update_layout(**PLOTLY_LAYOUT, height=360,
                            title="Global Yield Waterfall", yaxis_title="Wafer count")
        st.plotly_chart(fig_global, use_container_width=True)

        st.divider()


    st.divider()

# ---------------------------------------------------------------------------
# Yield Trend + DPPM dual-axis
# ---------------------------------------------------------------------------
tab_trend, tab_line, tab_pareto, tab_calendar, tab_waterfall = st.tabs([
    "📉 Overall Yield Trend", "🏭 By Line & Shift", "📊 Failure Pareto",
    "📅 Calendar Heatmap", "🌊 Yield Waterfall",
])

with tab_trend:
    if not global_period_df.empty:
        global_line_agg = global_period_df.groupby(["process_date", "line_id"]).agg({
        "total_wafers_tested": "sum",
        "quarantined_wafers": "sum",
        "passed_wafers": "sum",
        "failed_wafers": "sum"
        }).reset_index()

        global_line_agg["yield_percentage"] = (global_line_agg["passed_wafers"] / global_line_agg["total_wafers_tested"].replace(0, np.nan)) * 100
        global_line_agg["ppm_defective"] = (global_line_agg["failed_wafers"] / global_line_agg["total_wafers_tested"].replace(0, np.nan)) * 1000000
        global_line_agg = global_line_agg.sort_values("process_date", ascending=False)

        # Plot individual lines
        fig = px.line(
            global_line_agg, x="process_date", y="yield_percentage", color="line_id",
            markers=True, color_discrete_sequence=[TEAL, BLUE, AMBER],
            labels={"yield_percentage": "Yield %", "process_date": "Date", "line_id": "Line"}
        )

        fig.add_trace(go.Scatter(
            x=global_period_agg["process_date"], y=global_period_agg["yield_percentage"],
            name="Overall", mode="lines",
            line=dict(color=AMBER, dash="dot", width=1.5),
        ))

        # Add Overall DPPM on secondary axis
        fig.add_trace(go.Bar(
            x=global_period_agg["process_date"], y=global_period_agg["ppm_defective"],
            name="Overall DPPM", yaxis="y2", opacity=0.3,
            marker_color=RED
        ))

        fig.update_layout(
            **PLOTLY_LAYOUT, height=450,
            title=f"Yield % by Line & Overall DPPM — last {window_days} days",
            yaxis=dict(title="Yield %", range=[0, 100]),
            yaxis2=dict(title="Overall DPPM", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="right", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True, key='global_yield_chart')
    else:
        st.info("No daily yield data found.")

with tab_line:
    
    if not shift_df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Yield % by production line")
            line_agg = shift_df.groupby("line_id").agg(
                passed=("passed_wafers", "sum"),
                tested=("wafers_tested", "sum")
            ).reset_index()
            line_agg["yield_pct"] = (line_agg["passed"] / line_agg["tested"].replace(0, np.nan)) * 100

            fig = px.bar(line_agg, x="line_id", y="yield_pct",
                         color="line_id", color_discrete_sequence=[TEAL, BLUE, AMBER],
                         labels={"yield_pct": "Avg Yield %", "line_id": "Line"},
                         text_auto=".1f")
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                              yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True, key="bar_chart_avg_yield_by_line")

        with col_b:
            st.markdown("#### Yield % by shift")
            shift_agg = shift_df.groupby("shift").agg(
                passed=("passed_wafers", "sum"),
                tested=("wafers_tested", "sum")
            ).reset_index()
            shift_agg["yield_pct"] = (shift_agg["passed"] / shift_agg["tested"].replace(0, np.nan)) * 100

            fig2 = px.bar(shift_agg, x="shift", y="yield_pct",
                          color="shift", color_discrete_sequence=[TEAL, AMBER, CORAL],
                          labels={"yield_pct": "Avg Yield %", "shift": "Shift"},
                          text_auto=".1f")
            fig2.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                               yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True, key="bar_chart_avg_yield_by_shift")

        st.markdown("#### Yield by line × shift (heatmap)")
        pivot = shift_df.groupby(["line_id", "shift"]).agg(
            passed=("passed_wafers", "sum"),
            tested=("wafers_tested", "sum")
        ).reset_index()
        pivot["yield_pct"] = (pivot["passed"] / pivot["tested"].replace(0, np.nan)) * 100
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
        st.plotly_chart(fig3, use_container_width=True, key="heatmap_avg_yield_by_line_shift")
    else:
        st.info("No shift-level data found. Run the multi-line generator to populate.")

    st.divider()
    
    with st.expander("View Yield Trend by Production Lines", expanded=False):
        st.markdown("#### Yield Trend & Metrics by Production Line")
        
        if not daily.empty:
            for line in lines_filter:
                line_df = daily[daily["line_id"] == line]
                line_max_date = line_df["process_date"].max()
                line_min_date = line_max_date - pd.Timedelta(days=window_days)
                line_start_str = line_min_date.strftime('%Y-%m-%d')
                line_end_str = line_max_date.strftime('%Y-%m-%d')

                line_period_df = line_df[
                    (line_df["process_date"] >= line_min_date) & 
                    (line_df["process_date"] <= line_max_date)
                ]

                line_agg = line_period_df.groupby("process_date").agg({
                    "total_wafers_tested": "sum",
                    "quarantined_wafers": "sum",
                    "passed_wafers": "sum",
                    "failed_wafers": "sum"
                }).reset_index()

                line_agg["yield_percentage"] = (line_agg["passed_wafers"] / line_agg["total_wafers_tested"].replace(0, np.nan)) * 100
                line_agg["ppm_defective"] = (line_agg["failed_wafers"] / line_agg["total_wafers_tested"].replace(0, np.nan)) * 1000000
                line_agg = line_agg.sort_values("process_date", ascending=False)

                line_total_tested = line_agg["total_wafers_tested"].sum()
                line_passed = line_agg["passed_wafers"].sum()
                line_failed = line_agg["failed_wafers"].sum()
                line_yield = (line_passed / line_total_tested) * 100
                line_dppm = (line_failed / line_total_tested) * 1000000

                st.markdown(f"### 🏭 {line}")
                st.caption(f"📅 **Data Range:** {line_start_str} to {line_end_str}")

                # Render the time plot for the individual line trendline
                fig = px.line(
                    line_agg, x="process_date", y="yield_percentage",
                    markers=True, color_discrete_sequence=[TEAL],
                    labels={"yield_percentage": "Yield %", "process_date": "Date"}
                )

                # Add Overall DPPM on secondary axis
                fig.add_trace(go.Bar(
                    x=line_agg["process_date"], y=line_agg["ppm_defective"],
                    name="Overall DPPM", yaxis="y2", opacity=0.3,
                    marker_color=RED
                ))

                fig.update_layout(
                    **PLOTLY_LAYOUT, 
                    height=350,
                    margin=dict(l=0, r=0, t=20, b=0),
                    yaxis=dict(title="Yield %", range=[0, 100]),
                    yaxis2=dict(title="Overall DPPM", overlaying="y", side="right",
                                gridcolor="rgba(0,0,0,0)"),
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True, key=f"yield_chart_{line}")

                c1, c2, c3 = st.columns(3)
                # Rendering metrics vertically beside the plot fits perfectly
                c1.metric(f"Overall Yield", f"{line_yield:.2f}%")
                c2.metric("Overall DPPM", f"{line_dppm:,.0f}")
                c3.metric("Total Wafers Tested", f"{int(line_total_tested):,}")
                
                st.divider()

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
    st.markdown("#### Yield Calendar by Production Line")
    
    if not daily.empty:
        cols = st.columns(len(lines_filter))

        for idx, line in enumerate(lines_filter):
            line_df = daily[daily["line_id"] == line]
            if line_df.empty:
                continue
            
            # Determine line-specific date context
            line_max_date = line_df["process_date"].max()
            line_min_date = line_max_date - pd.Timedelta(days=window_days)
            line_period_df = line_df[
                (line_df["process_date"] >= line_min_date) & 
                (line_df["process_date"] <= line_max_date)
            ]
            
            line_start_str = line_min_date.strftime('%Y-%m-%d')
            line_end_str = line_max_date.strftime('%Y-%m-%d')

            cal_line = line_period_df.sort_values("process_date").copy()
            cal_line["process_date"] = pd.to_datetime(cal_line["process_date"])
            cal_line["week"] = cal_line["process_date"].dt.isocalendar().week.astype(int)
            cal_line["dow"] = cal_line["process_date"].dt.dayofweek

            fig_cal_line = go.Figure(go.Heatmap(
                x=cal_line["week"],
                y=cal_line["dow"],
                z=cal_line["yield_percentage"],
                text=[f"{d.strftime('%b %d')}<br>Yield: {y:.1f}%"
                      for d, y in zip(cal_line["process_date"], cal_line["yield_percentage"])],
                hovertemplate="%{text}<extra></extra>",
                colorscale=[[0, "#A32D2D"], [0.5, "#854F0B"], [1, "#0F6E56"]],
                zmin=0, zmax=100,
                xgap=3, ygap=3,
            ))
            
            # Add a unique key so Streamlit renders multiple charts correctly
            fig_cal_line.update_layout(
                **PLOTLY_LAYOUT, height=280,
                title=f"🏭 {line} Yield Calendar ({line_start_str} to {line_end_str})",
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(7)),
                    ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                    autorange="reversed",
                ),
                xaxis_title="Week of year",
            )

            with cols[idx % len(cols)]:
                st.plotly_chart(fig_cal_line, use_container_width=True, key=f"cal_chart_{line}")
    else:
        st.info("No daily data available for line-specific calendars.")

with tab_waterfall:
    st.markdown("#### Yield Loss Waterfall")
    
    if not daily.empty and not global_period_df.empty:

        st.markdown("### 🏭 By Production Line")
        
        # Create columns dynamically based on how many lines are selected
        # (Limit to 2 or 3 per row so they don't get too squished)
        cols = st.columns(len(lines_filter)) 
        
        for idx, line in enumerate(lines_filter):
            line_df = daily[daily["line_id"] == line]
            if line_df.empty:
                continue
                
            line_max_date = line_df["process_date"].max()
            line_min_date = line_max_date - pd.Timedelta(days=window_days)
            
            line_period_df = line_df[
                (line_df["process_date"] >= line_min_date) & 
                (line_df["process_date"] <= line_max_date)
            ]
            
            line_start_str = line_min_date.strftime('%Y-%m-%d')
            line_end_str = line_max_date.strftime('%Y-%m-%d')
            
            l_tested = int(line_period_df["total_wafers_tested"].sum())
            l_quarantined = int(line_period_df["quarantined_wafers"].sum())
            l_passed = int(line_period_df["passed_wafers"].sum())
            l_failed = int(line_period_df["failed_wafers"].sum())
            
            l_gross_input = l_tested + l_quarantined
            l_values = [l_gross_input, -l_quarantined, -l_failed, l_passed]
            
            fig_line = go.Figure(go.Waterfall(
                orientation="v", measure=measure, x=labels, y=l_values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": TEAL}},
                decreasing={"marker": {"color": RED}},
                totals={"marker": {"color": BLUE}},
                text=[f"{abs(v):,}" for v in l_values], textposition="outside",
            ))
            
            # Hide the legend and reduce margins to fit better in a column
            fig_line.update_layout(
                **PLOTLY_LAYOUT, 
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                title=dict(text=f"{line}<br><sup>{line_start_str} to {line_end_str}</sup>", font=dict(size=14)),
                showlegend=False
            )
            
            # Render inside the dynamically assigned column
            with cols[idx % len(cols)]:
                st.plotly_chart(fig_line, use_container_width=True, key=f"waterfall_chart_{line}")
                
    else:
        st.info("No data available for waterfall charts.")