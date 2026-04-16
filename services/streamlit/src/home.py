import streamlit as st
from datetime import datetime
from common.utils import (
    apply_page_config, badge,
    PLOTLY_LAYOUT, TEAL, AMBER, RED, CORAL, GRAY,
)

apply_page_config("Executive Overview", "🏭")
 
st.title("🏭 SECOM Manufacturing Command Center")
st.caption(f"Last render: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------------------------------------------
# Sidebar navigation hint
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("""
    | Page | Purpose |
    |------|---------|
    | Yield Analytics | Trends, Pareto, calendar heatmap |
    | SPC Monitor | Nelson rules, Cp/Cpk |
    | OEE Equipment | Line efficiency breakdown |
    | Failure Analysis | Pareto, correlation heatmap |
    | ML Insights | Predictive yield, drift |
    | Simulator | Multi-line generator control |
    """)