import numpy as np
import pandas as pd
import streamlit as st
import s3fs
from typing import Dict, List, Any
from common.config import ServiceConfig

# Initialize config once to use in shared functions
service_config = ServiceConfig()

@st.cache_resource
def get_s3_filesystem():
    """Shared function to establish MinIO connection."""
    return s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': service_config.minio_endpoint}, 
        key=service_config.minio_access_key, 
        secret=service_config.minio_secret_key
    )

def get_latest_generated_batch(fs):
    """Shared function to fetch the most recently generated parquet file from MinIO."""
    try:
        fs.invalidate_cache()
        search_path = f"{service_config.minio_bucket.strip('/')}/**/*.parquet"
        files = fs.glob(search_path)

        if not files: 
            return None

        latest_file = max(files, key=lambda x: fs.info(x)['LastModified'])
        
        with fs.open(latest_file, 'rb') as f:
            df = pd.read_parquet(f)
            return df.copy()

    except Exception as e:
        st.error(f"MinIO Connection Error: {e}")
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