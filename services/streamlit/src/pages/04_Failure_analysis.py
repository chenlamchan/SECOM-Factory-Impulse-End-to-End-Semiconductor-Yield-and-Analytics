"""
04_Failure_Analysis.py — Failure Analysis
───────────────────────────────────────────
Sensor failure Pareto, correlation heatmap on failing wafers,
top-sensor scatter matrix, and failure stratification by line/shift.
Heavy computations read silver parquet directly (bypass Trino).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import pointbiserialr

from common.utils import (
    AMBER, BLUE, CORAL, GRAY, PLOTLY_LAYOUT, RED, TEAL, PURPLE,
    apply_page_config, get_s3_filesystem, query_trino,
)

apply_page_config("Failure Analysis", "🔍")
st.title("🔍 Failure Analysis")

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_pareto():
    return query_trino("""
        SELECT rank, sensor_id, correlation, abs_correlation,
               effect_strength, direction
        FROM gold_failure_pareto
        ORDER BY rank LIMIT 20
    """)

@st.cache_data(ttl=60, show_spinner=False)
def load_failure_by_line():
    return query_trino("""
        SELECT line_id, shift, process_date,
               wafers_tested, failed_wafers, yield_pct, ppm_defective
        FROM gold_shift_metrics
        ORDER BY process_date DESC LIMIT 200
    """)

@st.cache_resource(ttl=600, show_spinner=False)
def load_silver_sample():
    """Read silver parquet directly for ML-grade computations."""
    try:
        fs = get_s3_filesystem()
        files = sorted(
            fs.glob("data-lake/warehouse/silver/**/*.parquet"), reverse=True
        )[:30]
        if not files:
            return None
        frames = [pd.read_parquet(f"s3://{f}", filesystem=fs) for f in files]
        return pd.concat(frames, ignore_index=True)
    except Exception:
        return None

# ---------------------------------------------------------------------------
with st.spinner("Loading failure data …"):
    pareto_df  = load_pareto()
    line_df    = load_failure_by_line()

tab_pareto, tab_heatmap, tab_scatter, tab_line = st.tabs([
    "📊 Sensor Pareto", "🌡 Correlation Heatmap",
    "🔵 Scatter Matrix", "🏭 Failure by Line"
])

# ---------------------------------------------------------------------------
with tab_pareto:
# ---------------------------------------------------------------------------
    st.subheader("Top sensors correlated with wafer failure")
    st.caption(
        "Point-biserial |r|. Computed in gold_failure_pareto dbt model. "
        "Strong = |r| ≥ 0.3, Moderate = |r| ≥ 0.1"
    )
    if not pareto_df.empty:
        color_map = {"Strong": RED, "Moderate": AMBER, "Weak": GRAY}
        fig = go.Figure()
        for strength in ["Strong", "Moderate", "Weak"]:
            subset = pareto_df[pareto_df["effect_strength"] == strength]
            if not subset.empty:
                fig.add_trace(go.Bar(
                    x=subset["abs_correlation"],
                    y=[f"Sensor {s}" for s in subset["sensor_id"]],
                    orientation="h",
                    name=strength,
                    marker_color=color_map[strength],
                    text=[f"|r| = {v:.3f}" for v in subset["abs_correlation"]],
                    textposition="outside",
                ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=600, barmode="overlay",
            xaxis_title="|Correlation with fail label|",
            yaxis=dict(autorange="reversed"),
            title="Failure Pareto — top 20 sensors",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Direction annotation
        st.markdown("#### Failure direction")
        for _, row in pareto_df.iterrows():
            if row["effect_strength"] == "Strong":
                direction_icon = "↑" if row["correlation"] > 0 else "↓"
                st.markdown(
                    f"**Sensor {row['sensor_id']}** — {direction_icon} {row['direction']}"
                )
    else:
        st.info("Pareto data not yet available. Run dbt models.")

# ---------------------------------------------------------------------------
with tab_heatmap:
# ---------------------------------------------------------------------------
    st.subheader("Sensor–sensor correlation (failing wafers only)")
    st.caption(
        "Pearson correlation matrix among top failing sensors, "
        "computed only on wafers that failed. "
        "Correlated sensors may share a common root cause."
    )

    silver = load_silver_sample()
    if silver is not None and "wafer_status" in silver.columns:
        failed_df = silver[silver["wafer_status"] == "Fail"].copy()
        top_sensors = (
            pareto_df["sensor_id"].head(12).tolist()
            if not pareto_df.empty
            else ["59", "103", "511", "424", "158"]
        )
        available = [s for s in top_sensors if s in failed_df.columns]

        if len(available) >= 3:
            corr = failed_df[available].corr()
            mask_labels = [f"S{s}" for s in corr.index]

            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=mask_labels, y=mask_labels,
                colorscale="RdBu",
                zmid=0, zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                xgap=2, ygap=2,
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT, height=560,
                title=f"Correlation matrix — {len(failed_df):,} failing wafers",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough sensors with data for correlation matrix.")
    else:
        st.info("Silver data not accessible. Check MinIO connection and silver layer.")

# ---------------------------------------------------------------------------
with tab_scatter:
# ---------------------------------------------------------------------------
    st.subheader("Top sensor scatter matrix")
    st.caption("Pass vs Fail coloured — helps identify decision boundaries visually.")

    silver = load_silver_sample()
    if silver is not None and "wafer_status" in silver.columns:
        top5 = (
            pareto_df["sensor_id"].head(5).tolist()
            if not pareto_df.empty
            else ["59", "103", "511", "424", "158"]
        )
        avail5 = [s for s in top5 if s in silver.columns]

        if len(avail5) >= 2:
            sample = silver[avail5 + ["wafer_status"]].dropna(subset=avail5).sample(
                min(500, len(silver)), random_state=42
            )
            # Rename for display
            rename = {s: f"Sensor {s}" for s in avail5}
            sample = sample.rename(columns=rename)

            fig = px.scatter_matrix(
                sample,
                dimensions=list(rename.values()),
                color="wafer_status",
                color_discrete_map={"Pass": TEAL, "Fail": RED},
                opacity=0.5,
            )
            fig.update_traces(diagonal_visible=False, marker=dict(size=3))
            fig.update_layout(
                **PLOTLY_LAYOUT, height=700,
                title="Scatter matrix — top 5 failure-correlated sensors (sample 500 wafers)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient sensor columns for scatter matrix.")
    else:
        st.info("Silver data not accessible.")

# ---------------------------------------------------------------------------
with tab_line:
# ---------------------------------------------------------------------------
    st.subheader("Failure rate by production line & shift")

    if not line_df.empty:
        col_a, col_b = st.columns(2)

        with col_a:
            line_agg = line_df.groupby("line_id").agg(
                ppm=("ppm_defective", "mean"),
                yield_pct=("yield_pct", "mean"),
                total_failed=("failed_wafers", "sum"),
            ).reset_index()
            fig = px.bar(
                line_agg, x="line_id", y="ppm",
                color="ppm",
                color_continuous_scale=[[0, TEAL], [0.5, AMBER], [1, RED]],
                text_auto=".0f",
                labels={"ppm": "Avg DPPM", "line_id": "Line"},
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                              title="Average DPPM by line")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            shift_agg = line_df.groupby(["line_id", "shift"]).agg(
                failed=("failed_wafers", "sum"),
                tested=("wafers_tested", "sum"),
            ).reset_index()
            shift_agg["fail_rate"] = shift_agg["failed"] / shift_agg["tested"].clip(lower=1) * 100
            fig2 = px.bar(
                shift_agg, x="line_id", y="fail_rate", color="shift",
                color_discrete_map={"Day": TEAL, "Swing": BLUE, "Night": PURPLE},
                barmode="group", text_auto=".1f",
                labels={"fail_rate": "Fail rate %", "line_id": "Line"},
            )
            fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                               title="Failure rate % by line & shift")
            st.plotly_chart(fig2, use_container_width=True)

        # Rolling DPPM per line
        st.markdown("#### DPPM trend by line")
        if "process_date" in line_df.columns:
            daily_line = line_df.groupby(["process_date", "line_id"])["ppm_defective"].mean().reset_index()
            fig3 = px.line(
                daily_line.sort_values("process_date"),
                x="process_date", y="ppm_defective",
                color="line_id",
                color_discrete_map={"LINE_A": TEAL, "LINE_B": BLUE, "LINE_C": AMBER},
                markers=True,
                labels={"ppm_defective": "DPPM", "process_date": "Date"},
            )
            fig3.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No line-level failure data available.")