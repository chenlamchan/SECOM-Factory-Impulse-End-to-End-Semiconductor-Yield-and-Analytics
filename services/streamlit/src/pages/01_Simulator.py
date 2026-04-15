import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import wasserstein_distance
from common.config import StateStore, SimulationConfig, ServiceConfig

# Import shared logic from utils.py (which is one directory up in the root of src)
from common.utils import get_s3_filesystem, get_latest_generated_batch

service_config = ServiceConfig()

# Constants
RAW_DATA_PATH = service_config.raw_dataset_file
DB_PATH = service_config.db_path
store = StateStore(DB_PATH)

@st.cache_data
def load_baseline():
    # Load your raw dataset for comparison
    return pd.read_csv(RAW_DATA_PATH)

# --- UI Setup ---
st.title("🏭 SECOM Synthetic Data Generator")

tab1, tab2 = st.tabs(["🎛️ Control Panel", "📊 Data Validation & Reporting"])

# --- TAB 1: Control Panel ---
with tab1:
    config = store.get_config()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Global Settings")
        new_is_running = st.toggle("Enable Generator Daemon", value=config.is_running)
        new_batch_size = st.slider("Batch Size (FOUP)", 1, 100, config.batch_size, 25)
        new_interval = st.number_input("Generation Interval (Seconds)", min_value=1, value=config.generation_interval_seconds)
        new_jitter = st.slider("Micro-Noise Variance (% of Std Dev)", 0.0, 0.1, config.jitter_variance, step=0.01)
        
    with col2:
        st.subheader("Targeted Anomaly Injection (Drift)")
        st.info("Inject Sigma Shifts to specific features to simulate machine degradation.")
        
        drift_feature = st.text_input("Feature Name (e.g., '59', '124')")
        drift_sigma = st.slider("Sigma Shift", -5.0, 5.0, 0.0, step=0.5)
        
        current_drift = config.drift_config.copy()
        
        if st.button("Apply Drift to Feature"):
            if drift_feature:
                current_drift[drift_feature] = drift_sigma
                st.success(f"Added drift: {drift_feature} shifted by {drift_sigma}σ")
                
        if st.button("Clear All Drift"):
            current_drift = {}
            st.success("Cleared all drift anomalies.")
            
        st.write("**Active Drift Configurations:**")
        st.json(current_drift)

    if st.button("Save Configuration", type="primary"):
        new_config = SimulationConfig(
            is_running=new_is_running,
            batch_size=new_batch_size,
            generation_interval_seconds=new_interval,
            jitter_variance=new_jitter,
            drift_config=current_drift
        )
        store.update_config(new_config)
        st.toast("Configuration synchronized with Daemon!")

# --- TAB 2: Validation Dashboard ---
with tab2:
    st.subheader("Data Quality Monitor")
    
    if st.button("Refresh Latest Batch"):
        st.rerun()

    baseline_df = load_baseline()
    
    # Call functions from utils
    fs = get_s3_filesystem()
    latest_df = get_latest_generated_batch(fs)
    
    if latest_df is None:
        st.warning("No generated data found in MinIO yet. Turn on the generator in the Control Panel.")
    else:
        st.success(f"Loaded latest batch generated at: {latest_df['generation_timestamp'].iloc[0]}")
        st.write(f"**Applied Anomalies:** `{latest_df['applied_drift_features'].iloc[0]}`")
        
        latest_df['Time'] = pd.to_datetime(latest_df['Time'])
        baseline_df['Time'] = pd.to_datetime(baseline_df['Time'])

        batch_dates = latest_df['Time'].dt.date.unique()
        matched_baseline_df = baseline_df[baseline_df['Time'].dt.date.isin(batch_dates)]

        if matched_baseline_df.empty:
            st.error("Could not match the generated batch dates to the baseline dataset.")
            st.stop()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Feature Drift Detection (Relative Wasserstein)")
            st.caption("Compares local distributions. Score > 0.10 Threshold indicates significant drift.")

            test_features = baseline_df.select_dtypes(include=['float64']).columns[:10].tolist()
            if current_drift: 
                test_features = list(current_drift.keys()) + [f for f in test_features if f not in current_drift.keys()]
            
            results = []
            for feat in test_features[:10]:
                if feat in latest_df.columns:
                    base_data = matched_baseline_df[feat].dropna()
                    latest_data = latest_df[feat].dropna()

                    if len(base_data) > 0 and len(latest_data) > 0:
                        base_std = base_data.std()

                        if base_std == 0:
                            is_drifted = latest_data.std() > 0 or latest_data.mean() != base_data.mean()
                            rel_distance = 0.0 
                        else:
                            abs_distance = wasserstein_distance(base_data, latest_data)
                            rel_distance = abs_distance / base_std
                            tolerance_threshold = 0.10 
                            is_drifted = rel_distance > tolerance_threshold

                        status = "🚨 Drifted" if is_drifted else "✅ Stable"
                        results.append({"Feature": feat, "Relative Drift Score": round(rel_distance, 4), "Status": status})

            st.dataframe(pd.DataFrame(results), width='stretch')

        with col_b:
            st.markdown("### Missing Value Topology")
            st.caption("Ensuring null-rates remain consistent.")
            
            base_nulls = matched_baseline_df[test_features[:10]].isnull().mean() * 100
            latest_nulls = latest_df[test_features[:10]].isnull().mean() * 100
            
            null_df = pd.DataFrame({'Baseline %': base_nulls, 'Generated %': latest_nulls}).reset_index()
            fig = px.bar(null_df, x='index', y=['Baseline %', 'Generated %'], barmode='group', labels={'index': 'Feature', 'value': '% Missing'})
            st.plotly_chart(fig, width='stretch')