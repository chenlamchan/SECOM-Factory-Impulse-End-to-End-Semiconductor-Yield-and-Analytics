import streamlit as st
import trino
import pandas as pd
import plotly.graph_objects as go

# Import shared engine and S3 helpers from utils.py
from common.utils import SPCEngine, get_s3_filesystem, get_latest_generated_batch

# --- Trino Connection for Gold Metrics ---
def query_trino(sql):
    with trino.dbapi.connect(host='trino', port=8080, user='admin', catalog='secom_catalog', schema='gold') as conn:
        return pd.read_sql_query(sql, conn)

@st.fragment(run_every="10s")
def render_realtime_spc(ref_limits):
# 2. Get Latest Batch using utils.py
    fs = get_s3_filesystem()
    latest_df = get_latest_generated_batch(fs)
    
    if latest_df is not None and not ref_limits.empty:
        target_sensor = st.selectbox("Select Sensor", ref_limits['sensor_id'].tolist())
        sensor_ref = ref_limits[ref_limits['sensor_id'] == target_sensor].iloc[0]
        
        # 3. Analyze with SPC Engine
        analysis = SPCEngine.analyze_batch(latest_df[target_sensor], sensor_ref['mu'], sensor_ref['sigma'])
        
        if analysis['ooc']:
            st.error(f"⚠️ OUT OF CONTROL: {', '.join(analysis['violations'])}")
            if st.button("🚨 TRIGGER EMERGENCY INTERLOCK"):
                # Logic to publish "STOP" to NATS would go here
                st.toast("Interlock signal sent to production line!")
        
        # 4. Plotly SPC Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=latest_df[target_sensor], mode='lines+markers', name='Sensor Value'))
        fig.add_hline(y=analysis['ucl'], line_dash="dash", line_color="red", annotation_text="UCL")
        fig.add_hline(y=analysis['mean'], line_color="green", annotation_text="Mean")
        fig.add_hline(y=analysis['lcl'], line_dash="dash", line_color="red", annotation_text="LCL")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Awaiting generated batch data or sensor stats from Trino.")

st.title("🚀 SECOM Manufacturing Command Center")

tab_exec, tab_spc, tab_ctrl = st.tabs(["📈 Executive Yield", "🔬 Real-Time SPC", "⚙️ Machine Control"])



with tab_exec:
    st.subheader("Line Performance")
    try:
        kpis = query_trino("SELECT * FROM gold_daily_yield_metrics ORDER BY process_date DESC LIMIT 1")
        if not kpis.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Yield", f"{kpis['yield_percentage'].iloc[0]}%", "0.5%")
            c2.metric("DPPM", f"{kpis['ppm_defective'].iloc[0]}", "-120")
            c3.metric("Wafers Tested", int(kpis['total_wafers_tested'].iloc[0]))
        else:
            st.info("No yield metrics found in Gold layer yet.")
    except Exception as e:
        st.error(f"Failed to connect or query Trino: {e}")

with tab_spc:
    st.subheader("Real-Time Sensor Stability")
    
    try:
        # 1. Get Frozen Reference from Gold
        ref_limits = query_trino("SELECT * FROM gold_sensor_stats")
        render_realtime_spc(ref_limits)
            
    except Exception as e:
         st.error(f"Error fetching SPC Data: {e}")

with tab_ctrl:
    st.subheader("Machine Control Panel")
    st.info("Additional controls will go here.")