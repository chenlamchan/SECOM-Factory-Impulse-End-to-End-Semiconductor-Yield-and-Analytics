"""
batch_inference.py — Batch Scoring Pipeline (Trino + PyIceberg)
─────────────────────────────────────────────────────────────────
Scores the silver reporting table against the current Champion model.

Run pattern:
  • Triggered as an Airflow task on a separate daily scoring schedule 
    (secom_ml_scoring_dag) or directly after ingestion.

Design:
  - Loads the Champion model artifact and its exact manifest from MLflow.
  - Queries the silver Iceberg table via TRINO to get un-scored rows efficiently.
  - Applies median imputation from the manifest to guarantee training/serving parity.
  - Scores in bulk using Pandas.
  - Appends results to secom_catalog.ml.predictions via PyIceberg.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import trino
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    NestedField, StringType, DoubleType, IntegerType, TimestampType, DateType
)
from pyiceberg.exceptions import NoSuchTableError

import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import s3fs
from config import ServiceConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config = ServiceConfig()

MINIO_ENDPOINT = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key
CATALOG_URI = config.catalog_uri
S3_WAREHOUSE_PATH = config.minio_warehouse

# CRITICAL: Map custom MinIO vars to standard AWS vars for MLflow artifact downloading
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


def _get_catalog() -> SqlCatalog:
    """Iceberg catalog connection used strictly for WRITING the predictions."""
    return SqlCatalog(
        "secom_catalog",
        uri=CATALOG_URI,
        warehouse=S3_WAREHOUSE_PATH,
        **{
            "s3.endpoint": MINIO_ENDPOINT,
            "s3.access-key-id": MINIO_ACCESS_KEY,
            "s3.secret-access-key": MINIO_SECRET_KEY,
            "s3.path-style-access": "true",
            "s3.region": "us-east-1",
            "downcast-ns-timestamp-to-us-on-write": "true"
        }
    )


def load_champion_model_and_manifest(model_name: str, alias: str):
    """Downloads the champion model and its exact manifest from MLflow."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()

    try:
        v = client.get_model_version_by_alias(model_name, alias)
    except Exception as e:
        raise RuntimeError(f"No model found with alias '{alias}' for '{model_name}'.")

    run = client.get_run(v.run_id)
    m_type = run.data.tags.get("model_type", "xgboost")
    
    logger.info("Identified %s model v%s (run_id: %s) as %s", m_type, v.version, v.run_id, alias)

    # 1. Download Manifest
    manifest_uri = f"runs:/{v.run_id}/metadata/feature_manifest.json"
    local_manifest_path = mlflow.artifacts.download_artifacts(artifact_uri=manifest_uri)
    with open(local_manifest_path, "r") as f:
        manifest = json.load(f)

    # 2. Download Model Weights
    model_uri = f"models:/{model_name}@{alias}"
    if m_type == "lightgbm":
        model = mlflow.lightgbm.load_model(model_uri)
    else:
        model = mlflow.xgboost.load_model(model_uri)

    return model, manifest, v.version, m_type

def get_unscored_data_via_trino(lookback_days: int, catalog_name: str) -> pd.DataFrame:
    """Uses Trino to query the Iceberg silver table for recent records efficiently."""
    trino_host = os.environ.get("TRINO_HOST", "trino")
    trino_port = int(os.environ.get("TRINO_PORT", 8080))
    trino_user = os.environ.get("TRINO_USER", "airflow")

    logger.info("Connecting to Trino at %s:%d...", trino_host, trino_port)
    
    conn = trino.dbapi.connect(
        host=trino_host,
        port=trino_port,
        user=trino_user,
        catalog=catalog_name,
        schema="silver"
    )

    cursor = conn.cursor()

    cursor.execute("SELECT MAX(process_timestamp) FROM silver_secom_reporting")
    max_ts_row = cursor.fetchone()

    if not max_ts_row or not max_ts_row[0]:
        logger.warning("Silver table is empty. No data to score.")
        return pd.DataFrame()

    max_ts = max_ts_row[0]
    logger.info("Simulated current timestamp (Max in DB): %s", max_ts)
    
    # Trino syntax for date intervals
    query = f"""
        SELECT *
        FROM silver_secom_reporting
        WHERE process_timestamp >= TIMESTAMP '{max_ts}' - INTERVAL '{lookback_days}' DAY
    """
    
    logger.info("Executing query: %s", query.strip())

    with conn:
        df = pd.read_sql_query(query, conn)
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    logger.info("Loaded %d recent rows for scoring from Trino.", len(df))
    return df


def main():
    parser = argparse.ArgumentParser(description="SECOM ML — Batch Inference")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--model-name", default="secom_yield_predictor")
    parser.add_argument("--model-alias", default="champion")
    parser.add_argument("--output-namespace", default="ml")
    parser.add_argument("--output-table", default="predictions")
    args = parser.parse_args()

    # ── 1. Load Model & Contract ──────────────────────────────────────────────
    model, manifest, model_version, m_type = load_champion_model_and_manifest(args.model_name, args.model_alias)

    active_sensors = manifest.get("active_features_list", [])
    cat_features   = manifest.get("categorical_features", [])
    lag_features   = manifest.get("lag_features", [])
    cat_modes      = manifest.get("categorical_modes", {})
    medians        = manifest.get("medians", {})
    
    feature_names  = active_sensors + cat_features + lag_features + ["missing_sensor_rate"]

    # ── 2. Load Data via Trino ────────────────────────────────────────────────
    catalog_name = os.environ.get("TRINO_CATALOG", "secom_catalog")
    raw_df = get_unscored_data_via_trino(args.lookback_days, catalog_name)

    if raw_df.empty:
        logger.info("No new rows to score. Exiting safely.")
        sys.exit(0)

    # ── 3. Preprocess to Model Contract ───────────────────────────────────────
    if "missing_sensor_rate" not in raw_df.columns:
        valid_sensors = [c for c in active_sensors if c in raw_df.columns]
        raw_df["missing_sensor_rate"] = raw_df[valid_sensors].isnull().sum(axis=1) / 590.0

    data = {f: [] for f in feature_names}
    for f in feature_names:
        if f in raw_df.columns:
            fallback = cat_modes.get(f, "UNKNOWN") if f in cat_features else medians.get(f, 0.0)
            data[f] = raw_df[f].fillna(fallback).values
        else:
            fallback = cat_modes.get(f, "UNKNOWN") if f in cat_features else medians.get(f, 0.0)
            logger.warning("Feature '%s' entirely missing from silver data. Injecting fallback: %s", f, fallback)
            data[f] = [fallback] * len(raw_df)

    X = pd.DataFrame(data)

    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].astype("category")
            
    numeric_cols = [c for c in feature_names if c not in cat_features]
    X[numeric_cols] = X[numeric_cols].astype(np.float32)

    # ── 4. Predict ────────────────────────────────────────────────────────────
    logger.info("Executing batch predictions...")
    probs = model.predict_proba(X)[:, 1]
    predictions = (probs >= 0.5).astype(int)

    now = datetime.now(timezone.utc)
    
    results_df = pd.DataFrame({
        "observation_id":       raw_df["observation_id"].values,
        "prediction_timestamp": [now] * len(raw_df),
        "prediction_date":      [now.date()] * len(raw_df), 
        "defect_probability":   probs.astype(float),
        "yield_probability":    (1.0 - probs).astype(float),
        "prediction":           predictions.astype(int),
        "model_name":           [args.model_name] * len(raw_df),
        "model_version":        [str(model_version)] * len(raw_df),
        "model_alias":          [args.model_alias] * len(raw_df),
    })

    # ── 5. Write to Iceberg via PyIceberg ─────────────────────────────────────
    # We use PyIceberg here instead of Trino because appending Arrow tables
    # directly to Iceberg in Python is significantly faster and cleaner than 
    # building giant SQL INSERT statements.
    catalog = _get_catalog()

    arrow_schema = pa.schema([
        pa.field("observation_id", pa.string(), nullable=False),
        pa.field("prediction_timestamp", pa.timestamp('us'), nullable=False), 
        pa.field("prediction_date", pa.date32(), nullable=False),
        pa.field("defect_probability", pa.float64(), nullable=False),
        pa.field("yield_probability", pa.float64(), nullable=False),
        pa.field("prediction", pa.int32(), nullable=False),
        pa.field("model_name", pa.string(), nullable=False),
        pa.field("model_version", pa.string(), nullable=False),
        pa.field("model_alias", pa.string(), nullable=False),
    ])

    arrow_table = pa.Table.from_pandas(results_df, schema=arrow_schema, preserve_index=False)
    
    try:
        catalog.create_namespace(args.output_namespace)
    except Exception:
        pass 

    table_id = f"{args.output_namespace}.{args.output_table}"
    
    try:
        iceberg_table = catalog.load_table(table_id)
        iceberg_table.append(arrow_table)
        logger.info("Appended %d rows to existing table %s", len(results_df), table_id)
    except NoSuchTableError:
        logger.info("Creating new table %s", table_id)
        schema = Schema(
            NestedField(1, "observation_id", StringType(), required=True),
            NestedField(2, "prediction_timestamp", TimestampType(), required=True),
            NestedField(3, "prediction_date", DateType(), required=True),
            NestedField(4, "defect_probability", DoubleType(), required=True),
            NestedField(5, "yield_probability", DoubleType(), required=True),
            NestedField(6, "prediction", IntegerType(), required=True),
            NestedField(7, "model_name", StringType(), required=True),
            NestedField(8, "model_version", StringType(), required=True),
            NestedField(9, "model_alias", StringType(), required=True),
        )
        
        catalog.create_table(
            identifier=table_id,
            schema=schema,
            properties={"format-version": "2"}
        )
        iceberg_table = catalog.load_table(table_id)
        iceberg_table.append(arrow_table)

    logger.info("Batch inference complete.")

if __name__ == "__main__":
    main()