"""
common/utils.py — Shared utilities for the SECOM dashboard.
 
Provides:
  - Chart theme constants (Plotly dark industrial)
  - Generator Engine UI

"""

import numpy as np
import pandas as pd
import streamlit as st
import s3fs
from typing import Dict, List, Any, Optional
from common.config import ServiceConfig
import trino

# ---------------------------------------------------------------------------
# Chart theme
# ---------------------------------------------------------------------------

TEAL   = "#1D9E75"
AMBER  = "#EF9F27"
CORAL  = "#D85A30"
RED    = "#E24B4A"
BLUE   = "#378ADD"
PURPLE = "#7F77DD"
GRAY   = "#888780"

LINE_COLORS = [TEAL, BLUE, AMBER, CORAL, PURPLE, RED]

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", size=13, color="#E6EDF3"),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    # xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    # yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
)

# ---------------------------------------------------------------------------
# Page config + CSS injection
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* Metric cards */
[data-testid="stMetric"] {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { color: #8B949E; font-size: 12px; }
/* ADDED COLOR HERE -> */
[data-testid="stMetricValue"] { color: #E6EDF3; font-size: 26px; font-weight: 500; } 
 
/* Sidebar */
[data-testid="stSidebar"] { background: #0D1117; border-right: 1px solid #21262D; }
/* ADDED SIDEBAR TEXT OVERRIDE HERE -> */
[data-testid="stSidebar"] * { color: #C9D1D9 !important; }
 
/* Tabs */
[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 2px solid #1D9E75;
    color: #1D9E75;
}
 
/* DataFrames */
/* ADDED BACKGROUND AND TEXT COLOR HERE -> */
[data-testid="stDataFrame"] { 
    background: #0D1117; 
    border: 1px solid #30363D; 
    border-radius: 8px; 
}
[data-testid="stDataFrame"] * { color: #E6EDF3; }

/* Status badges */
.badge-ok    { background:#0F6E56; color:#E1F5EE; padding:2px 8px; border-radius:4px; font-size:12px; }
.badge-warn  { background:#854F0B; color:#FAEEDA; padding:2px 8px; border-radius:4px; font-size:12px; }
.badge-alarm { background:#A32D2D; color:#FCEBEB; padding:2px 8px; border-radius:4px; font-size:12px; }
.badge-idle  { background:#3d3d3a; color:#D3D1C7; padding:2px 8px; border-radius:4px; font-size:12px; }
 
/* Dividers */
hr { border-color: #21262D; }
 
/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
"""

def apply_page_config(title: str, icon: str = "🏭") -> None:
    st.set_page_config(page_title=f"{title} | SECOM", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

def badge(text: str, level: str = "ok") -> str:
    """Return HTML badge string. level: ok | warn | alarm | idle"""
    return f'<span class="badge-{level}">{text}</span>'


# Initialize config once to use in shared functions
service_config = ServiceConfig()

# ---------------------------------------------------------------------------
# Trino connection
# ---------------------------------------------------------------------------
@st.cache_resource(ttl=3600)
def get_trino_engine():
    """Cached Trino connection. Retries on first connect."""
    for attempt in range(5):
        try:
            conn = trino.dbapi.connect(
                host=service_config.trino_host,
                port=service_config.trino_port,
                user="admin",
                catalog="secom_catalog",
                schema="gold",
                request_timeout=60,
            )
            conn.cursor().execute("SELECT 1")   # warm-up
            return conn
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
 
@st.cache_data(ttl=60, show_spinner=False)
def query_trino(sql: str, schema: str = "gold") -> pd.DataFrame:
    """Execute SQL against Trino, return DataFrame. Results cached 60 s."""
    conn = trino.dbapi.connect(
        host=service_config.trino_host,
        port=service_config.trino_port,
        user="admin",
        catalog="secom_catalog",
        schema=schema,
    )

    return pd.read_sql_query(sql, conn)

# ---------------------------------------------------------------------------
# S3 / MinIO helpers
# ---------------------------------------------------------------------------
@st.cache_resource(ttl=3600)
def get_s3_filesystem():
    """Shared function to establish MinIO connection."""
    return s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': service_config.minio_endpoint}, 
        key=service_config.minio_access_key, 
        secret=service_config.minio_secret_key
    )

def get_latest_generated_batch(
    fs: s3fs.S3FileSystem, 
    bucket:str = service_config.minio_bucket,
    line_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Shared function to fetch the most recently generated parquet file from MinIO."""
    try:
        fs.invalidate_cache()
        prefix = f"{bucket.strip('/')}/line_id={line_id}" if line_id else bucket
        files = sorted(fs.glob(f"{prefix}/**/*.parquet"), reverse=True)

        if not files: 
            return None

        with fs.open(files[0], 'rb') as f:
            return pd.read_parquet(f)

    except Exception as e:
        st.error(f"Unable to get latest batch, Error: {e}")
        return None

class SPCEngine:
    """Production-grade Statistical Process Control (SPC) logic."""

    @staticmethod
    def analyze_batch(data: pd.Series, mu: float, sigma: float) -> Dict[str, Any]:
        """Applies Western Electric Rules to a data series."""
        if data.empty or sigma == 0:
            return {"ooc": False, "violations": []}

        violations = []
        
        # Rule 1: Point outside 3-sigma
        ooc_points = data[np.abs(data - mu) > 3 * sigma]
        if not ooc_points.empty:
            violations.append(f"Rule 1: {len(ooc_points)} point(s) outside 3σ")

        # Rule 2: 2 out of 3 consecutive points outside 2-sigma
        if len(data) >= 3:
            beyond_2s = np.abs(data - mu) > 2 * sigma
            if beyond_2s.rolling(3).sum().max() >= 2:
                violations.append("Rule 2: 2/3 points beyond 2σ")

        # Rule 4: 8 consecutive points on one side of the mean
        if len(data) >= 8:
            above = (data > mu).rolling(8).sum()
            below = (data < mu).rolling(8).sum()
            if above.max() == 8 or below.max() == 8:
                violations.append("Rule 4: 8 consecutive points on one side of mean")

        return {
            "ooc": len(violations) > 0,
            "violations": violations,
            "mean": mu,
            "ucl": mu + 3 * sigma,
            "lcl": mu - 3 * sigma
        }