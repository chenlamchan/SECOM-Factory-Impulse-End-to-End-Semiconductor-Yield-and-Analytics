"""
04_Failure_Analysis.py — Failure Analysis
───────────────────────────────────────────
Sensor failure Pareto, correlation heatmap on failing wafers,
top-sensor scatter matrix, and failure stratification by line/shift.
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

from common.config import ServiceConfig

service_config = ServiceConfig()

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
            xaxis_title="|Correlation with label|",
            yaxis=dict(autorange="reversed"),
            title="Failure Pareto — top 20 sensors",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full table"):
            st.dataframe(pareto_df, use_container_width=True, hide_index=True)

        # Direction annotation
        st.markdown("#### Top 5 Failure Directions")
        
        # Take the top 5 sensors regardless of strength (pareto_df is already sorted)
        for _, row in pareto_df.head(5).iterrows():
            direction_icon = "↑" if row["correlation"] > 0 else "↓"
            st.markdown(
                f"**Sensor {row['sensor_id']}** ({row['effect_strength']}) — {direction_icon} {row['direction']}"
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

    top_sensors = (
        pareto_df["sensor_id"].astype(str).head(10).tolist() 
        if not pareto_df.empty 
        else ["59", "103", "511", "424", "158"]
    )

    trino_cols = ", ".join([f'"{s}"' for s in top_sensors])

    sql = f"""
        SELECT {trino_cols}
        FROM silver_secom_reporting
        WHERE wafer_status = 'Fail'
    """

    try:
        failed_df = query_trino(sql, schema="silver")

        if not failed_df.empty and len(failed_df.columns) >= 3:
            corr = failed_df.corr()

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
            
    except Exception as e:
        st.error(f"Failed to fetch correlation data from Trino: {e}")

# ---------------------------------------------------------------------------
with tab_scatter:
# ---------------------------------------------------------------------------
    st.subheader("Feature Relationships & Decision Boundaries")

    if not pareto_df.empty:

        # --- PLOT 1: FEATURE-FEATURE SCATTER MATRIX ---
        st.markdown("#### 1. Collinear Sensors (Failing Wafers)")
        st.caption(
            "Scatter matrix of failing sensors exhibiting strong multicollinearity **(|r| ≥ 0.70)**. "
            "This reveals redundant sensors or underlying physical coupling in the equipment."
        )
        
        if corr is not None and not corr.empty and len(corr.columns) >= 2:
            
            # 1. Apply Industry Standard Threshold
            R_THRES = 0.70
            corr_abs = corr.abs()

            diagonal_mask = np.eye(len(corr_abs), dtype=bool)
            corr_abs = corr_abs.mask(diagonal_mask, 0)
            
            # 2. Find all sensors that have at least one connection >= 0.70
            # Sort them by their highest correlation value to prioritize the strongest couples
            max_corrs = corr_abs.max()
            collinear_sensors = max_corrs[max_corrs >= R_THRES].sort_values(ascending=False).index.tolist()
            
            if len(collinear_sensors) >= 2:
                
                # 3. UI Safeguard: Cap at 6 dimensions to prevent browser crash/unreadable grids
                display_sensors = collinear_sensors[:6]
                
                if len(collinear_sensors) > 6:
                    st.warning(
                        f"⚠️ **Systemic anomaly detected:** {len(collinear_sensors)} sensors "
                        f"exhibit strong collinearity (|r| ≥ {R_THRES}). "
                        f"Displaying the top 6 to maintain chart readability."
                    )
                
                trino_cols_feat = ", ".join([f'"{s}"' for s in display_sensors])
                sql_feat = f"""
                    SELECT wafer_status, {trino_cols_feat}
                    FROM silver_secom_reporting
                    WHERE wafer_status IS NOT NULL
                    LIMIT 1000
                """
                
                try:
                    sample_feat = query_trino(sql_feat, schema="silver")
                    if not sample_feat.empty:
                        rename_feat = {s: f"S{s}" for s in display_sensors}
                        sample_feat = sample_feat.rename(columns=rename_feat)

                        fig_feat = px.scatter_matrix(
                            sample_feat,
                            dimensions=list(rename_feat.values()),
                            color="wafer_status",
                            color_discrete_map={"Pass": TEAL, "Fail": RED},
                            opacity=0.6,
                        )
                        fig_feat.update_traces(diagonal_visible=False, marker=dict(size=4))
                        fig_feat.update_layout(**PLOTLY_LAYOUT, height=700, margin=dict(t=10, b=10))
                        st.plotly_chart(fig_feat, use_container_width=True)
                    else:
                        st.warning("Could not fetch data for collinear sensors.")
                except Exception as e:
                    st.error(f"Failed to fetch feature-feature scatter data: {e}")
            else:
                st.success(
                    f"✅ **No strong multicollinearity detected.** None of the top sensors "
                    f"exceed the |r| ≥ {R_THRES} threshold with each other."
                )
        else:
            st.info("Correlation matrix from the Heatmap tab is unavailable.")

        st.divider()

        # --- PLOT 2: TARGET-FEATURE SCATTER ---
        st.markdown("#### 2. Primary Decision Boundary (Target Predictors)")
        st.caption(
            "Explore sensors with a **Moderate or Strong** correlation to Wafer Failure. "
            "Use the dropdowns to swap axes and look for clusters where red (Fail) isolates from blue (Pass)."
        )
        
        # Filter Pareto for mathematically significant predictors
        meaningful_predictors = pareto_df[
            pareto_df["effect_strength"].isin(["Strong", "Moderate"])
        ]["sensor_id"].astype(str).tolist()

        if len(meaningful_predictors) >= 2:
            # --- UI: Interactive Axis Selection ---
            col_x, col_y = st.columns(2)
            with col_x:
                top1 = st.selectbox("X-Axis Sensor", options=meaningful_predictors, index=0)
            with col_y:
                top2 = st.selectbox("Y-Axis Sensor", options=meaningful_predictors, index=1)

            if top1 == top2:
                st.warning("Please select two different sensors for a meaningful 2D scatter plot.")
            else:
                trino_cols_target = f'"{top1}", "{top2}"'
                sql_target = f"""
                    SELECT wafer_status, {trino_cols_target}
                    FROM silver_secom_reporting
                    WHERE wafer_status IS NOT NULL
                    LIMIT 1000
                """
                
                try:
                    sample_target = query_trino(sql_target, schema="silver")
                    if not sample_target.empty:
                        fig_target = px.scatter(
                            sample_target,
                            x=top1,
                            y=top2,
                            color="wafer_status",
                            color_discrete_map={"Pass": TEAL, "Fail": RED},
                            opacity=0.7,
                            labels={
                                top1: f"Sensor {top1}", 
                                top2: f"Sensor {top2}"
                            }
                        )
                        
                        fig_target.update_layout(**PLOTLY_LAYOUT, height=500, margin=dict(t=10, b=10))
                        fig_target.update_traces(marker=dict(size=6, line=dict(width=0.5, color='#161B22')))
                        st.plotly_chart(fig_target, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to fetch target scatter data: {e}")
                
        elif len(meaningful_predictors) == 1:
            # Fallback: If only 1 good predictor exists, a 1D Box plot is the best visualization
            top1 = meaningful_predictors[0]
            st.info(f"Only one significant predictor found (Sensor {top1}). Displaying distribution instead of scatter.")
            
            sql_target = f'SELECT wafer_status, "{top1}" FROM silver_secom_reporting WHERE wafer_status IS NOT NULL LIMIT 1000'
            
            try:
                sample_target = query_trino(sql_target, schema="silver")
                if not sample_target.empty:
                    fig_box = px.box(
                        sample_target, x="wafer_status", y=top1, color="wafer_status",
                        color_discrete_map={"Pass": TEAL, "Fail": RED},
                        title=f"Distribution of Sensor {top1} (Sole Significant Predictor)"
                    )
                    fig_box.update_layout(**PLOTLY_LAYOUT, height=400)
                    st.plotly_chart(fig_box, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to fetch target data: {e}")
        else:
            st.info("No sensors with a 'Moderate' or 'Strong' association were found. A reliable decision boundary plot cannot be generated.")
    else:
        st.info("Pareto data required to identify target sensors.")

# ---------------------------------------------------------------------------
with tab_line:
# ---------------------------------------------------------------------------
    st.subheader("Failure rate by production line & shift")

    if not line_df.empty:
        col_a, col_b = st.columns(2)

        with col_a:
            line_agg = line_df.groupby("line_id").agg(
                total_failed=("failed_wafers", "sum"),
                total_tested=("wafers_tested", "sum"),
            ).reset_index()

            line_agg["ppm"] = (line_agg["total_failed"] / line_agg["total_tested"].clip(lower=1)) * 1e6

            fig = px.bar(
                line_agg, x="line_id", y="ppm",
                color="ppm",
                color_continuous_scale=[[0, TEAL], [0.5, AMBER], [1, RED]],
                text_auto=".0f",
                labels={"ppm": "Avg DPPM", "line_id": "Line"},
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                              title="True Aggregate DPPM by line")
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
            daily_line = line_df.groupby(["process_date", "line_id"]).agg(
                daily_failed=("failed_wafers", "sum"),
                daily_tested=("wafers_tested", "sum")
            ).reset_index()

            daily_line = daily_line.sort_values(["line_id", "process_date"])    

            daily_line["ppm_defective"] = (daily_line["daily_failed"] / daily_line["daily_tested"].clip(lower=1)) * 1e6
            daily_line["ppm_smoothed"] = daily_line.groupby("line_id")["ppm_defective"].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
            
            fig3 = px.line(
                daily_line,
                x="process_date", 
                y="ppm_smoothed", # Use smoothed value for the line
                color="line_id",
                color_discrete_map={"LINE_A": TEAL, "LINE_B": BLUE, "LINE_C": AMBER},
                labels={"ppm_smoothed": "DPPM (7D Avg)", "process_date": "Date"},
            )

            fig3.add_trace(go.Scatter(
                x=daily_line["process_date"], y=daily_line["ppm_defective"],
                mode='markers', marker=dict(size=4, opacity=0.3),
                name="Raw DPPM", showlegend=False
            ))

            fig3.update_layout(**PLOTLY_LAYOUT, height=350, yaxis_title="DPPM")
            # Prevent lines from connecting across long gaps
            fig3.update_traces(connectgaps=False) 
            
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No line-level failure data available.")