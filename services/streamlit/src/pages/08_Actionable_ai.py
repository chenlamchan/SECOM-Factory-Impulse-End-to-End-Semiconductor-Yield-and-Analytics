"""
08_Actionable_AI.py — Actionable AI & Performance Estimation
──────────────────────────────────────────────────────────────
Sections:
  1. Estimated performance banner — CBPE AUC/F1 without ground truth
  2. Conformal interval inflation — distribution shift proxy
  3. Model staleness alert
  4. Counterfactual explorer — select a predicted-fail wafer, get recourse
  5. MILP minimal change viewer — fewest parameter changes to flip prediction
  6. DiCE diverse alternatives — 3 different change combinations
  7. Input validation — Mahalanobis distance check for any set of sensor values
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime

# --- Adapted Imports ---
from common.utils import (
    apply_page_config, get_trino_engine, TEAL, AMBER, RED, BLUE, CORAL, GRAY,
    PLOTLY_LAYOUT, badge,
)

apply_page_config("Actionable AI", "🎯")
st.title("🎯 Actionable AI — Recourse & Performance Estimation")

# --- Trino Helper Wrapper ---
def query_trino(query: str, schema: str = "gold") -> pd.DataFrame:
    """Helper to execute Trino queries using the shared engine."""
    conn = get_trino_engine()
    cur = conn.cursor()
    
    if schema != "gold":
        cur.execute(f"USE {schema}")
        
    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    
    if schema != "gold":
        cur.execute("USE gold")
        
    return pd.DataFrame(rows, columns=columns)

SERVING_URL = "http://ml-serving:8001"

st.caption(
    "Counterfactual explanations answer: *what is the minimum process change "
    "to flip a predicted failure to a pass?* "
    "Performance estimation answers: *how well is the model performing right now, "
    "without waiting for ground-truth labels?*"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Performance Estimation (CBPE — no ground truth needed)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Live Performance Estimation (CBPE)")

@st.cache_data(ttl=60)
def load_performance_estimate():
    try:
        r = requests.get(f"{SERVING_URL}/explain/performance", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

est = load_performance_estimate()

if "error" in est:
    st.error(f"Serving API unavailable: {est['error']}")
elif not est.get("estimates_available"):
    st.info(est.get("message", "No estimates available yet."))
else:
    c1, c2, c3, c4 = st.columns(4)
    
    auc = est.get("estimated_auc", 0)
    f1  = est.get("estimated_f1", 0)
    age = est.get("model_age_hours", 0)
    inf = est.get("width_inflation", 0)

    # Styling logic for alerts
    auc_color = "normal" if auc >= 0.70 else "inverse"
    inf_color = "normal" if inf <= 0.20 else "inverse"
    age_color = "normal" if age <= 168 else "inverse"

    c1.metric("Estimated AUC", f"{auc:.3f}", 
              help="Confidence-Based Performance Estimate (no labels needed)",
              delta_color=auc_color)
    c2.metric("Estimated F1 (Fail)", f"{f1:.3f}")
    
    c3.metric("Prediction Interval Inflation", f"+{inf*100:.1f}%", 
              help="Measures certainty degradation due to distribution shift",
              delta_color=inf_color)
    c4.metric("Model Age", f"{age:.1f} hrs",
              help="Time since the model was calibrated",
              delta_color=age_color)

    st.caption(f"**Method:** {est.get('method')} — {est.get('note')} (Last calculated: {est.get('estimated_at')})")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Counterfactual Explanations (Recourse)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Counterfactual Explorer (Process Recourse)")
st.markdown(
    "Select a wafer recently predicted to **Fail** to generate alternative "
    "process parameters that would have resulted in a **Pass**."
)

@st.cache_data(ttl=60)
def load_recent_failures():
    # Only load wafers predicted to FAIL in the last 24 hours
    return query_trino("""
        SELECT observation_id, prediction_timestamp, defect_probability, line_id
        FROM gold_model_predictions
        WHERE prediction = 1 
          AND prediction_timestamp >= CURRENT_TIMESTAMP - INTERVAL '24' HOUR
        ORDER BY defect_probability DESC
        LIMIT 50
    """)

recent_fails = load_recent_failures()

if recent_fails.empty:
    st.success("No predicted failures in the last 24 hours to analyze!", icon="✅")
else:
    # Build dropdown options
    options = {
        row["observation_id"]: f"{row['observation_id']} (P(Fail)={row['defect_probability']:.3f}, {row['line_id']})"
        for _, row in recent_fails.iterrows()
    }
    
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_obs = st.selectbox("Select a predicted failure:", list(options.keys()), format_func=lambda x: options[x])
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_cf = st.button("Generate Recourse", type="primary")

    if generate_cf:
        with st.spinner(f"Running DiCE and MILP solvers for {selected_obs}..."):
            try:
                r = requests.post(
                    f"{SERVING_URL}/explain/counterfactual",
                    json={"observation_id": selected_obs, "method": "both", "max_changes": 4},
                    timeout=30
                )
                cf_data = r.json()
                
                if "error" in cf_data:
                    st.error(cf_data["error"])
                else:
                    st.success(f"Original P(Fail) was {cf_data['original_fail_prob']:.4f}. Found {len(cf_data['counterfactuals'])} ways to recover this wafer.")
                    
                    tabs = st.tabs([f"Option {i+1} ({cf['method'].upper()})" for i, cf in enumerate(cf_data['counterfactuals'])])
                    
                    for i, cf in enumerate(cf_data['counterfactuals']):
                        with tabs[i]:
                            st.markdown(f"**New P(Fail):** `{cf['new_fail_prob']:.4f}`")
                            
                            if cf['method'] == 'milp':
                                st.info("💡 **MILP Guarantee:** This is mathematically the absolute minimum number of parameter changes required to pass.")
                            else:
                                st.info("💡 **DiCE Generation:** This is a diverse alternative prioritizing proximity to the original state.")
                                
                            changes = cf['changes']
                            if not changes:
                                st.warning("No actionable changes found within limits.")
                            else:
                                # Create a visual diff table
                                diff_df = pd.DataFrame(changes)
                                diff_df.columns = ["Sensor", "Original Value", "Required Value", "Change required"]
                                
                                st.dataframe(
                                    diff_df, 
                                    use_container_width=True, 
                                    hide_index=True
                                )
                                
            except Exception as e:
                st.error(f"Failed to generate counterfactuals: {e}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Input Validation / OOD Check
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("OOD Check (Mahalanobis Distance)")
st.caption("Manually test if a set of sensor readings falls outside the multivariate training distribution.")

with st.form("ood_check"):
    vc = st.columns(5)
    top5 = ["59", "103", "511", "424", "158"]
    sensor_vals = {}
    for i, s in enumerate(top5):
        with vc[i]:
            sensor_vals[s] = st.number_input(f"Sensor {s}", value=0.0, format="%.4f")

    validate_btn = st.form_submit_button("Check distribution membership")

if validate_btn:
    try:
        r = requests.post(
            f"{SERVING_URL}/explain/validate-input",
            json={"features": sensor_vals, "observation_id": "manual_check"},
            timeout=10,
        )
        val = r.json()

        dist = val.get("mahalanobis_distance", 0)
        pval = val.get("p_value", 1.0)
        in_d = val.get("is_in_distribution", True)

        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Mahalanobis D",  f"{dist:.3f}")
        vc2.metric("p-value",        f"{pval:.4f}")
        vc3.metric("In distribution", "✓ Yes" if in_d else "✗ No")

        if not in_d:
            st.error(
                f"**Out-of-distribution input** (D={dist:.2f}, p={pval:.4f}). "
                "The model has not seen sensor combinations like this during training. "
                "Prediction may be unreliable — check for sensor faults before acting.",
                icon="🚨"
            )
        else:
            st.success("Input is within the known training distribution.", icon="✅")
    except Exception as e:
        st.error(f"OOD check failed: {e}")