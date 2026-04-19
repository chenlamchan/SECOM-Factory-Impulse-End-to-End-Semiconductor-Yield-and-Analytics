"""
02_SPC_Monitor.py — Statistical Process Control Monitor
────────────────────────────────────────────────────────
Real-time X-charts with all Nelson rules, Cp/Cpk gauges,
sensor health matrix heatmap, and violation drill-down.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common.utils import (
    AMBER, BLUE, GRAY, PLOTLY_LAYOUT, RED, TEAL,
    SPCEngine, apply_page_config, get_s3_filesystem,
    get_latest_generated_batch, query_trino, badge,
)

apply_page_config("SPC Monitor", "🔬")
st.title("🔬 Statistical Process Control Monitor")

TRACKED_SENSORS = ["59", "103", "511", "424", "158"]

# ---------------------------------------------------------------------------
# Reference limits from gold
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_sensor_refs():
    return query_trino("SELECT * FROM gold_sensor_stats")

@st.cache_data(ttl=60, show_spinner=False)
def load_violations(hours: int = 24, line_filter_val: str = "All"):
    line_clause = f"AND v.line_id = '{line_filter_val}'" if line_filter_val != "All" else ""

    return query_trino(f"""
        WITH LatestPerGroup AS (
            SELECT line_id, sensor_id, MAX(process_timestamp) as max_ts
            FROM gold_spc_violations
            GROUP BY line_id, sensor_id
        )

        SELECT v.sensor_id, v.rule_name, v.rule_number, v.line_id, v.shift,
               COUNT(*) AS count, MAX(v.z_score) AS max_z,
               MAX(v.process_timestamp) AS last_seen
        FROM gold_spc_violations v
        LEFT JOIN LatestPerGroup l 
          ON v.line_id = l.line_id AND v.sensor_id = l.sensor_id
        WHERE v.process_timestamp >= l.max_ts - INTERVAL '{hours}' HOUR
        {line_clause}
        GROUP BY v.sensor_id, v.rule_name, v.rule_number, v.line_id, v.shift
        ORDER BY v.rule_number, count DESC
    """)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    sensor_sel  = st.selectbox("Sensor", TRACKED_SENSORS)
    line_sel    = st.selectbox("Line filter", ["All", "LINE_A", "LINE_B", "LINE_C"])
    auto_refresh = st.checkbox("Auto-refresh (10 s)", value=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_xchart, tab_capability, tab_health, tab_violations = st.tabs([
    "📈 X Chart", "🎯 Cp/Cpk Capability", "🌡 Sensor Health Matrix", "🚨 Violation Log"
])

# ---------------------------------------------------------------------------
with tab_xchart:
# ---------------------------------------------------------------------------
    col_nav1, col_nav2 = st.columns([1, 4], vertical_alignment="bottom")

    with col_nav1:
        batch_offset = st.number_input(
            "Traverse Batches", 
            min_value=0, 
            max_value=5, 
            value=0, 
            step=1,
            help="0 = Current Batch. Increase the number to look back at previous batches."
        )
    with col_nav2:
        if batch_offset == 0:
            st.info("Showing **Current** Live Batch")
        else:
            st.warning(f"Showing Historical Batch **-{batch_offset}**")

    @st.fragment(run_every="10s" if auto_refresh and batch_offset == 0 else None)
    def render_xchart():
        refs = load_sensor_refs()
        if refs.empty:
            st.info("Sensor reference stats not available. Run dbt models.")
            return
            
        sensor_refs = refs[refs["sensor_id"] == sensor_sel]
        if line_sel != "All":
            sensor_refs = sensor_refs[sensor_refs["line_id"] == line_sel]

        if sensor_refs.empty:
            st.warning(f"No reference stats for sensor {sensor_sel} on {line_sel}.")
            return

        fs = get_s3_filesystem()
        line_arg = None if line_sel == "All" else line_sel
        batch_df = get_latest_generated_batch(fs, line_id=line_arg, batch_offset=batch_offset)

        if batch_df is None or sensor_sel not in batch_df.columns:
            st.warning("Awaiting batch data. Start the generator.")
            return

        if "line_id" not in batch_df.columns or "tester_id" not in batch_df.columns:
            st.error("Batch data is missing 'line_id' or 'tester_id' columns. Cannot group by equipment.")
            return

        # Group incoming batch data by line and tester
        grouped = batch_df.groupby(['line_id', 'tester_id'])
        rendered_any = False

        for (b_line, b_tester), group_df in grouped:
            ref_match = sensor_refs[
                (sensor_refs['line_id'] == b_line) & 
                (sensor_refs['tester_id'] == b_tester)
            ]

            if ref_match.empty:
                continue

            rendered_any = True
            ref = ref_match.iloc[0]

            series = group_df[sensor_sel].dropna().reset_index(drop=True)
            if series.empty:
                continue

            analysis = SPCEngine.analyze_batch(
                series, 
                float(ref["mu"]),
                float(ref["sigma"]),
                float(ref["usl"]), 
                float(ref["lsl"])
            )

            col_chart, col_metrics = st.columns([3, 1], gap="large")
            title = f"Line {b_line} | Tester {b_tester} — Sensor {sensor_sel}"

            with col_chart:
                fig = SPCEngine.build_xbar_chart(
                    series, 
                    float(ref["mu"]), 
                    float(ref["sigma"]), 
                    sensor_sel, 
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_metrics:
                st.markdown(f"**Status**")
                # Alarm status
                if analysis["ooc"]:
                    st.error(
                        f"⚠️ **OUT OF CONTROL**\n\n"
                        f"{analysis['ooc_count']} violation(s):\n\n"
                        f"{', '.join(analysis['violations'])}"
                    )
                else:
                    st.success("✅ **In control**\n\nNo Nelson rule violations detected.")

                # Quick stats displayed in a 2x2 grid inside the right column
                mc1, mc2 = st.columns(2)
                mc1.metric("Batch mean", f"{analysis['mean']:.4f}")
                mc2.metric("Batch std", f"{analysis['std']:.4f}")
                
                mc3, mc4 = st.columns(2)
                mc3.metric("Ref mean (μ)", f"{float(ref['mu']):.4f}")
                mc4.metric("Ref σ", f"{float(ref['sigma']):.4f}")

            # Add a visual break between different equipment groupings
            st.divider()

        if not rendered_any:
            st.warning("No matching reference stats found for the lines and testers in the current batch.")

    render_xchart()

# ---------------------------------------------------------------------------
with tab_capability:
# ---------------------------------------------------------------------------
    refs = load_sensor_refs()
    fs   = get_s3_filesystem()
    line_arg = None if line_sel == "All" else line_sel

    cap_mode = st.radio(
        "Analysis Window", 
        ["Current Batch (Short-term: Cp/Cpk)", "Last 7 Days (Long-term: Pp/Ppk)"], 
        horizontal=True
    )

    is_historical = "Long-term" in cap_mode

    if is_historical:
        line_clause = f"AND s.line_id = '{line_sel}'" if line_sel != "All" else ""
        hist_query = f"""
            WITH LatestPerGroup AS (
                SELECT line_id, MAX(process_timestamp) as max_ts
                FROM silver_secom_reporting
                GROUP BY line_id
            )

            SELECT "59", "103", "511", "424", "158"
            FROM silver_secom_reporting s
            LEFT JOIN LatestPerGroup lpg 
                ON s.line_id = lpg.line_id
            WHERE s.process_timestamp >= lpg.max_ts - INTERVAL '7' DAY
            {line_clause}
        """
        # Note: Assuming 'silver' schema based on your DBT file names
        batch_df = query_trino(hist_query, schema="silver")

        st.markdown("#### Long-term Process Capability (Pp / Ppk) — Last 7 Days")
        st.caption("Pp > 1.33 = capable | Ppk > 1.0 = centred | Ppk < 1 = process adjustment needed")
        m_prefix = "P" # Used to dynamically rename columns to Pp/Ppk
    
    else:
        batch_df = get_latest_generated_batch(fs, line_id=line_arg, batch_offset=0)

        st.markdown("#### Short-term Process Capability (Cp / Cpk) — Current Batch")
        st.caption("Cp > 1.33 = capable | Cpk > 1.0 = centred | Cpk < 1 = process adjustment needed")
        m_prefix = "C"

    if batch_df is None or refs.empty or batch_df.empty:
        st.info("Awaiting data.")
    else:
        cap_rows = []
        for sid in TRACKED_SENSORS:
            if sid not in batch_df.columns:
                continue
            ref_row = refs[refs["sensor_id"] == sid]
            if ref_row.empty:
                continue
            ref = ref_row.iloc[0]
            series = batch_df[sid].dropna()
            cap = SPCEngine.capability_indices(series, float(ref["usl"]), float(ref["lsl"]))
            cap_rows.append({
                "Sensor": f"Sensor {sid}",
                f"{m_prefix}p":  cap["Cp"],  f"{m_prefix}pk": cap["Cpk"],
                f"{m_prefix}pu": cap["Cpu"], f"{m_prefix}pl": cap["Cpl"],
                "Status": (
                    badge("Capable", "ok") if cap["Cpk"] >= 1.33 else
                    badge("Marginal", "warn") if cap["Cpk"] >= 1.0 else
                    badge("Incapable", "alarm")
                ),
            })

        cap_df = pd.DataFrame(cap_rows)
        if not cap_df.empty:
            st.write(cap_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Gauge charts
        st.markdown(f"#### {m_prefix}pk gauges")
        cols = st.columns(len(cap_rows))
        for col, row in zip(cols, cap_rows):
            cpk_val = float(row[f"{m_prefix}pk"])
            color = TEAL if cpk_val >= 1.33 else (AMBER if cpk_val >= 1.0 else RED)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cpk_val,
                title={"text": row["Sensor"], "font": {"size": 12, "color": "#8B949E"}},
                gauge=dict(
                    axis=dict(range=[0, 2]),
                    bar=dict(color=color, thickness=0.7),
                    steps=[
                        {"range": [0, 1.0],  "color": "rgba(162,45,45,0.2)"},
                        {"range": [1.0, 1.33],"color":"rgba(133,79,11,0.2)"},
                        {"range": [1.33, 2], "color": "rgba(29,158,117,0.2)"},
                    ],
                    threshold=dict(line=dict(color=AMBER, width=2), thickness=0.75, value=1.33),
                ),
                number={"valueformat": ".2f", "font": {"size": 20}},
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=200, margin=dict(l=10, r=10, t=40, b=10))
            col.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_health:
# ---------------------------------------------------------------------------
    @st.cache_data(ttl=120, show_spinner=False)
    def load_health_matrix(line_filter_val:str = "All"):
        line_clause = f"AND v.line_id = '{line_filter_val}'" if line_filter_val != "All" else ""

        return query_trino(f"""
            WITH LatestPerGroup AS (
                SELECT line_id, sensor_id, MAX(process_timestamp) as max_ts
                FROM gold_spc_violations
                GROUP BY line_id, sensor_id
            )

            SELECT
                v.sensor_id,
                CAST(v.process_timestamp AS DATE) AS process_date,
                COUNT(*) AS violation_count
            FROM gold_spc_violations v
            LEFT JOIN LatestPerGroup l 
            ON v.line_id = l.line_id AND v.sensor_id = l.sensor_id
            WHERE v.process_timestamp >= l.max_ts - INTERVAL '14' DAY
            {line_clause}
            GROUP BY 1, 2
        """)

    matrix_df = load_health_matrix()
    refs       = load_sensor_refs()

    st.markdown("#### Sensor health matrix — violations per day (last 14 days)")

    if not matrix_df.empty:
        pivot = matrix_df.pivot_table(
            index="sensor_id", columns="process_date",
            values="violation_count", fill_value=0
        )
        # Sort sensors by total violations descending
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[f"Sensor {s}" for s in pivot.index],
            colorscale=[[0, "rgba(29,158,117,0.15)"], [0.5, "#854F0B"], [1, "#A32D2D"]],
            zmin=0,
            text=pivot.values,
            texttemplate="%{text}",
            xgap=2, ygap=2,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                          xaxis_title="Date", yaxis_title="",
                          coloraxis_colorbar_title="Violations")
        st.plotly_chart(fig, use_container_width=True)

        # Summary sparklines below
        st.markdown("#### Per-sensor violation trend")
        cols = st.columns(len(pivot.index))
        for col, sensor_id in zip(cols, pivot.index):
            ts = pivot.loc[sensor_id]
            fig_sp = go.Figure(go.Scatter(
                y=ts.values, mode="lines+markers",
                line=dict(color=RED if ts.sum() > 10 else AMBER if ts.sum() > 3 else TEAL, width=2),
                marker=dict(size=4),
            ))
            fig_sp.update_layout(
                **PLOTLY_LAYOUT, height=100,
                margin=dict(l=5, r=5, t=5, b=5),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                showlegend=False,
            )
            col.caption(f"**Sensor {sensor_id}**")
            col.plotly_chart(fig_sp, use_container_width=True)
            col.caption(f"Total: {int(ts.sum())} violations")
    else:
        st.info("No violation data for the last 14 days.")

# ---------------------------------------------------------------------------
with tab_violations:
# ---------------------------------------------------------------------------
    st.markdown("#### SPC violations")

    alarm_hours = st.slider("Alarm window (hours)", 1, 72, 24, help="Filter violations by hours from the latest event")
    
    viols = load_violations(alarm_hours, line_sel)

    line_display_text = f"for {line_sel}" if line_sel != "All" else "across all lines"
    st.markdown(f"#### SPC violations — last {alarm_hours} hours {line_display_text}")

    if not viols.empty:
        for rule_num in sorted(viols["rule_number"].unique()):
            rule_df = viols[viols["rule_number"] == rule_num]
            rule_label = rule_df.iloc[0]["rule_name"]
            severity = "alarm" if rule_num == 1 else "warn" if rule_num <= 3 else "ok"
            st.markdown(
                f"{badge(rule_label, severity)} &nbsp; "
                f"{int(rule_df['count'].sum())} total occurrences across "
                f"{rule_df['sensor_id'].nunique()} sensor(s)",
                unsafe_allow_html=True,
            )
            disp = rule_df[["sensor_id", "line_id", "shift", "count", "max_z", "last_seen"]].copy()
            disp.columns = ["Sensor", "Line", "Shift", "Count", "Max |Z|", "Last seen"]
            disp["Max |Z|"] = disp["Max |Z|"].apply(lambda x: f"{x:.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.markdown("")
    else:
        st.success(f"No violations in the last {alarm_hours} hours.")