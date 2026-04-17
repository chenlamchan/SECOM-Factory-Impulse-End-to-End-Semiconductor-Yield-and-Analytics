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
def load_violations(hours: int = 24):
    return query_trino(f"""
        SELECT sensor_id, rule_name, rule_number, line_id, shift,
               COUNT(*) AS count, MAX(z_score) AS max_z,
               MAX(process_timestamp) AS last_seen
        FROM gold_spc_violations
        WHERE process_timestamp >= NOW() - INTERVAL '{hours}' HOUR
        GROUP BY sensor_id, rule_name, rule_number, line_id, shift
        ORDER BY rule_number, count DESC
    """)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    sensor_sel  = st.selectbox("Sensor", TRACKED_SENSORS)
    line_sel    = st.selectbox("Line filter", ["All", "LINE_A", "LINE_B", "LINE_C"])
    alarm_hours = st.slider("Alarm window (hours)", 1, 72, 24)
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
    @st.fragment(run_every="10s" if auto_refresh else None)
    def render_xchart():
        refs = load_sensor_refs()
        if refs.empty:
            st.info("Sensor reference stats not available. Run dbt models.")
            return

        ref_row = refs[refs["sensor_id"] == sensor_sel]
        if ref_row.empty:
            st.warning(f"No reference stats for sensor {sensor_sel}.")
            return
        ref = ref_row.iloc[0]

        fs = get_s3_filesystem()
        line_arg = None if line_sel == "All" else line_sel
        batch_df = get_latest_generated_batch(fs, line_id=line_arg)

        if batch_df is None or sensor_sel not in batch_df.columns:
            st.warning("Awaiting batch data. Start the generator.")
            return

        series = batch_df[sensor_sel].dropna().reset_index(drop=True)
        analysis = SPCEngine.analyze_batch(series, float(ref["mu"]), float(ref["sigma"]))

        # Alarm status
        if analysis["ooc"]:
            st.error(f"⚠️ OUT OF CONTROL — {analysis['ooc_count']} violation(s): "
                     f"{', '.join(analysis['violations'])}")
        else:
            st.success("✅ In control — no Nelson rule violations detected")

        # X-chart
        fig = SPCEngine.build_xbar_chart(series, float(ref["mu"]), float(ref["sigma"]), sensor_sel)
        st.plotly_chart(fig, use_container_width=True)

        # Quick stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Batch mean",    f"{analysis['mean']:.4f}")
        c2.metric("Batch std",     f"{analysis['std']:.4f}")
        c3.metric("Ref mean (μ)",  f"{float(ref['mu']):.4f}")
        c4.metric("Ref σ",         f"{float(ref['sigma']):.4f}")

    render_xchart()

# ---------------------------------------------------------------------------
with tab_capability:
# ---------------------------------------------------------------------------
    refs = load_sensor_refs()
    fs   = get_s3_filesystem()
    line_arg = None if line_sel == "All" else line_sel
    batch_df = get_latest_generated_batch(fs, line_id=line_arg)

    if batch_df is None or refs.empty:
        st.info("Awaiting data.")
    else:
        st.markdown("#### Process capability indices (Cp / Cpk) — current batch")
        st.caption("Cp > 1.33 = capable | Cpk > 1.0 = centred | Cpk < 1 = process adjustment needed")

        cap_rows = []
        for sid in TRACKED_SENSORS:
            if sid not in batch_df.columns:
                continue
            ref_row = refs[refs["sensor_id"] == sid]
            if ref_row.empty:
                continue
            ref = ref_row.iloc[0]
            series = batch_df[sid].dropna()
            cap = SPCEngine.capability_indices(series, float(ref["ucl"]), float(ref["lcl"]))
            cap_rows.append({
                "Sensor": f"Sensor {sid}",
                "Cp":  cap["Cp"],  "Cpk": cap["Cpk"],
                "Cpu": cap["Cpu"], "Cpl": cap["Cpl"],
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
        st.markdown("#### Cpk gauges")
        cols = st.columns(len(cap_rows))
        for col, row in zip(cols, cap_rows):
            cpk_val = float(row["Cpk"])
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
    def load_health_matrix():
        return query_trino("""
            SELECT
                sensor_id,
                CAST(process_timestamp AS DATE) AS process_date,
                COUNT(*) AS violation_count
            FROM gold_spc_violations
            WHERE process_timestamp >= NOW() - INTERVAL '14' DAY
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
    viols = load_violations(alarm_hours)
    st.markdown(f"#### SPC violations — last {alarm_hours} hours")

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