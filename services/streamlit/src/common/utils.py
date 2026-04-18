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
import plotly.graph_objects as go
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
    # margin=dict(l=50, r=20, t=40, b=40),
    # legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
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
        base_path = bucket.strip('/')

        # Case 1: A specific line is selected
        if line_id:
            prefix = f"{base_path}/line_id={line_id}"
            files = sorted(fs.glob(f"{prefix}/**/*.parquet"), reverse=True)
            
            if not files: 
                return None

            with fs.open(files[0], 'rb') as f:
                return pd.read_parquet(f)
        
        # Case 2: "All" is selected (line_id is None)
        else:
            line_dirs = fs.glob(f"{base_path}/line_id=*")

            latest_dfs = []
            for directory in line_dirs:
                # Get the latest file for this specific line
                files = sorted(fs.glob(f"{directory}/**/*.parquet"), reverse=True)
                if files:
                    with fs.open(files[0], 'rb') as f:
                        latest_dfs.append(pd.read_parquet(f))
                        
            if not latest_dfs:
                return None
                
            # Combine the latest batch from all lines into a single DataFrame
            return pd.concat(latest_dfs, ignore_index=True)

    except Exception as e:
        st.error(f"Unable to get latest batch, Error: {e}")
        return None


# ---------------------------------------------------------------------------
# SPC Engine
# ---------------------------------------------------------------------------

class SPCEngine:
    """Statistical Process Control calculations for SECOM sensor data."""

    @staticmethod
    def check_nelson_rules(data: pd.Series,mean: float,sigma: float) -> dict[str, list[int]]:
        """
        Returns {rule_label: [positional indices in violation]}.
        Empty dict = all in control.
        """
        # Helper to convert a few violation endpoints into full window ranges
        def expand_windows(end_indices: np.ndarray, window_size: int) -> list[int]:
            indices = set()
            for idx in end_indices:
                indices.update(range(idx - window_size + 1, idx + 1))
            return sorted(list(indices))

        if sigma == 0 or len(data) < 9:
            return {}
        z = (data - mean) / sigma
        n = len(z)
        violations: dict[str, list[int]] = {}
 
        # Rule 1 — beyond ±3σ
        r1 = np.where(z.abs() > 3)[0].tolist()
        if r1:
            violations["Rule 1: Beyond 3σ"] = r1
 
        # Rule 2 — 9 consecutive on same side
        pos_9 = (z > 0).rolling(9).sum() == 9
        neg_9 = (z < 0).rolling(9).sum() == 9
        r2_ends = np.where(pos_9 | neg_9)[0]
        if len(r2_ends) > 0:
            violations["Rule 2: 9 on one side"] = expand_windows(r2_ends, 9)
 
        # Rule 3 — 6 consecutive strictly trending
        diffs = data.diff()
        up_5 = (diffs > 0).rolling(5).sum() == 5
        down_5 = (diffs < 0).rolling(5).sum() == 5
        r3_ends = np.where(up_5 | down_5)[0]
        if len(r3_ends) > 0:
            violations["Rule 3: 6 trending"] = expand_windows(r3_ends, 6)
 
        # Rule 4 — 14 alternating
        signs = np.sign(diffs.fillna(0))
        switches = (signs * signs.shift(1)) < 0  # True if sign flips
        alt_12 = switches.rolling(12).sum() == 12
        r4_ends = np.where(alt_12)[0]
        if len(r4_ends) > 0:
            violations["Rule 4: 14 alternating"] = expand_windows(r4_ends, 14)
 
        # Rule 5 — 2 of 3 consecutive beyond ±2σ
        pos_2 = (z > 2).rolling(3).sum() >= 2
        neg_2 = (z < -2).rolling(3).sum() >= 2
        r5_ends = np.where(pos_2 | neg_2)[0]
        if len(r5_ends) > 0:
            violations["Rule 5: 2/3 beyond 2σ"] = expand_windows(r5_ends, 3)
 
        return violations

    @staticmethod
    def capability_indices(data: pd.Series,ucl: float,lcl: float,) -> dict[str, float]:
        """Compute Cp, Cpk, Cpu, Cpl."""
        clean = data.dropna()
        
        if len(clean) < 5:
            return {"Cp": 0.0, "Cpk": 0.0, "Cpu": 0.0, "Cpl": 0.0}
        mean = clean.mean()
        std  = clean.std(ddof=1)
        
        if std == 0:
            return {"Cp": 0.0, "Cpk": 0.0, "Cpu": 0.0, "Cpl": 0.0}
        cp  = (ucl - lcl) / (6 * std)
        cpu = (ucl - mean) / (3 * std)
        cpl = (mean - lcl) / (3 * std)
        cpk = min(cpu, cpl)
        
        return {
            "Cp":  round(cp, 3),
            "Cpk": round(cpk, 3),
            "Cpu": round(cpu, 3),
            "Cpl": round(cpl, 3),
        }

    @staticmethod
    def analyze_batch(series: pd.Series,mu: float,sigma: float,) -> dict:
        ucl = mu + 3 * sigma
        lcl = mu - 3 * sigma
        clean = series.dropna()
        violations = SPCEngine.check_nelson_rules(clean, mu, sigma)
        unique_ooc_count = len(set().union(*violations.values())) if violations else 0
        
        return {
            "mean": round(float(clean.mean()), 4) if len(clean) else 0,
            "std": round(float(clean.std()),  4) if len(clean) else 0,
            "ucl": round(ucl, 4),
            "lcl": round(lcl, 4),
            "ooc": unique_ooc_count > 0,
            "ooc_count": unique_ooc_count,
            "violations": list(violations.keys()),
            "capability": SPCEngine.capability_indices(clean, ucl, lcl),
        }
    
    @staticmethod
    def build_xbar_chart(data: pd.Series,mu: float,sigma: float,sensor_id: str,title: str = "",) -> go.Figure:
        ucl = mu + 3 * sigma
        uwl = mu + 2 * sigma
        lwl = mu - 2 * sigma
        lcl = mu - 3 * sigma

        clean_data = data.dropna().reset_index(drop=True)
 
        violations = SPCEngine.check_nelson_rules(clean_data, mu, sigma)
        viol_idx = sorted(list(set().union(*violations.values()))) if violations else []
 
        x = list(range(len(clean_data)))
        fig = go.Figure()
 
        # Main trace
        fig.add_trace(go.Scatter(
            x=x, y=clean_data.values,
            mode="lines+markers",
            name=f"Sensor {sensor_id}",
            line=dict(color=TEAL, width=1.5),
            marker=dict(size=5, color=TEAL),
        ))
 
        # Violation markers
        if viol_idx:
            fig.add_trace(go.Scatter(
                x=viol_idx,
                y=clean_data.iloc[viol_idx].values,
                mode="markers",
                name="Violation",
                marker=dict(size=10, color=RED, symbol="x", line=dict(width=2, color=RED)),
            ))
 
        # Control lines
        for y_val, color, dash, label in [
            (ucl, RED,   "dash",  "UCL (3σ)"),
            (uwl, AMBER, "dot",   "UWL (2σ)"),
            (mu,  TEAL,  "solid", "Mean"),
            (lwl, AMBER, "dot",   "LWL (2σ)"),
            (lcl, RED,   "dash",  "LCL (3σ)"),
        ]:
            fig.add_hline(
                y=y_val, line_dash=dash, line_color=color, line_width=1,
                annotation_text=label,
                annotation_position="right",
                annotation_font_color=color,
            )
 
        # Y-axis padding
        buf = (ucl - lcl) * 0.25
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=title or f"Sensor {sensor_id} — X Chart",
            yaxis_range=[lcl - buf, ucl + buf],
            height=380,
            uirevision=f"spc_{sensor_id}",
        )

        return fig

# ---------------------------------------------------------------------------
# OEE
# ---------------------------------------------------------------------------

class OEEEngine:
    """Helpers for OEE display and gauge charts."""
 
    @staticmethod
    def oee_gauge(value: float, title: str) -> go.Figure:
        color = TEAL if value >= 85 else (AMBER if value >= 65 else RED)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(value, 1),
            number={"suffix": "%", "font": {"size": 28, "color": "#E6EDF3"}},
            title={"text": title, "font": {"size": 14, "color": "#8B949E"}},
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#444"),
                bar=dict(color=color, thickness=0.7),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    {"range": [0, 50],  "color": "rgba(162,45,45,0.15)"},
                    {"range": [50, 65], "color": "rgba(133,79,11,0.15)"},
                    {"range": [65, 85], "color": "rgba(29,158,117,0.1)"},
                    {"range": [85, 100],"color": "rgba(29,158,117,0.2)"},
                ],
                threshold=dict(line=dict(color=AMBER, width=2), thickness=0.75, value=85),
            ),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=220, margin=dict(l=20, r=20, t=40, b=10))
        return fig
 
    @staticmethod
    def classification_badge(oee_pct: float) -> str:
        if oee_pct >= 85:
            return badge("World Class ≥85%", "ok")
        if oee_pct >= 65:
            return badge("Good 65–85%", "warn")
        if oee_pct >= 50:
            return badge("Average 50–65%", "warn")
        return badge("Poor <50%", "alarm")