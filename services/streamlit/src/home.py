import streamlit as st

st.set_page_config(page_title="SECOM Platform", layout="wide", page_icon="🏭")

st.title("🏭 SECOM Integrated Manufacturing Hub")

st.markdown("""
## Welcome to the Central Command
Select a module from the sidebar to begin:

1.  **Simulator & Validation**: Configure the data daemon and monitor drift.
2.  **Manufacturing Command Center**: View executive KPIs and Real-Time SPC charts.
""")

st.info("The navigation on the left allows you to switch between different parts of the pipeline.")