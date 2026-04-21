"""
extract_features.py — Step 1: Data Extraction
───────────────────────────────────────────────
PySpark job that reads the silver Iceberg reporting table and writes
a point-in-time feature snapshot to secom_catalog.ml.feature_snapshot.

Design decisions:
  - Reads directly from Iceberg using the Spark Iceberg catalog.
  - Applies a lookback window so training only sees recent data.
  - DYNAMIC FEATURE SELECTION: Calculates stats in a single pass to identify 
    columns with >40% missing data or 0.0 variance.
  - Keeps all features in the snapshot but writes selected features to manifest.
  - Remaps label_numeric (-1/1) → binary_label (0/1).
  - Writes to a dedicated ml namespace.
"""

import json
import sys
import argparse
import logging
from datetime import datetime, timedelta, timezone

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, round as spark_round, lit, current_timestamp, 
    sum as spark_sum, isnull, isnan, variance, max as spark_max
    )

from common.config import ServiceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = ServiceConfig()

EXTRACT_PIPELINE_VERSION = "1.0.0"

# Constants
MINIO_ENDPOINT = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key
CATALOG_URI = config.catalog_uri
CATALOG_USER = config.catalog_user
CATALOG_PASSWORD = config.catalog_password
S3_WAREHOUSE_PATH = config.minio_warehouse

META_COLS = [
    "observation_id", "process_timestamp", "line_id", "tester_id",
    "shift", "lot_id", "wafer_status", "label_numeric", "missing_sensor_count", 
    "silver_created_at", "processing_logic", "is_synthetic", "generation_timestamp",
    "applied_drift_features", "year", "month", "day", "source_file", "bronze_ingested_at",
    "bronze_pipeline_version", "silver_created_at", "processing_logic"
]

# Excluded silver_created_at and processing_logic
SELECTED_META_COLS = [
    # Core & Targets
    "observation_id", "process_timestamp", "label_numeric", "missing_sensor_count",
    # Factory Context (For Error Analysis)
    "line_id", "tester_id", "shift", "lot_id", "wafer_status",
    # Lineage (For Debugging)
    "bronze_pipeline_version", "source_file",
    # Experimentation (For strict Train/Test isolation)
    "is_synthetic", "applied_drift_features"
]

def create_spark_session(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    catalog_uri: str,
    catalog_user: str,
    catalog_password: str,
    warehouse_path: str
    ) -> SparkSession:
    """Configures Spark with Iceberg and MinIO capabilities."""
    return SparkSession.builder \
        .appName("SECOM_Raw_to_Bronze") \
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
        .config("spark.sql.catalog.secom_catalog", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.secom_catalog.type", "jdbc") \
        .config("spark.sql.catalog.secom_catalog.uri", catalog_uri) \
        .config("spark.sql.catalog.secom_catalog.jdbc.user", catalog_user) \
        .config("spark.sql.catalog.secom_catalog.jdbc.password", catalog_password) \
        .config("spark.sql.catalog.secom_catalog.jdbc.schema-version", "V1") \
        .config("spark.sql.catalog.secom_catalog.warehouse", warehouse_path) \
        .config("spark.sql.catalog.secom_catalog.io-impl", "org.apache.iceberg.hadoop.HadoopFileIO") \
        .config("spark.hadoop.fs.s3a.endpoint", minio_endpoint) \
        .config("spark.hadoop.fs.s3a.access.key", minio_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", minio_secret_key) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.client.region", "us-east-1") \
        .getOrCreate()

def main():
    from common.config import ServiceConfig
    config = ServiceConfig()

    # Parse the exact file paths passed by Airflow
    parser = argparse.ArgumentParser(description="SECOM ML — Feature Extraction")
    parser.add_argument("--lookback-days", type=int, default=90, help="How many calendar days of silver data to include.")
    parser.add_argument("--min-rows", type=int, default=500, help="Abort pipeline if fewer rows available.")
    parser.add_argument("--missing-threshold", type=float, default=0.40, help="Drop columns with missing data > this fraction.")
    parser.add_argument("--output-table", default="secom_catalog.ml.feature_snapshot")
    args = parser.parse_args()

    spark = create_spark_session(
        minio_endpoint=config.minio_endpoint,
        minio_access_key=config.minio_access_key,
        minio_secret_key=config.minio_secret_key,
        catalog_uri=config.catalog_uri,
        catalog_user=config.catalog_user,
        catalog_password=config.catalog_password,
        warehouse_path=config.minio_warehouse
    )
    logger.info("Spark Session initialized.")

    # ── Read silver reporting table (Iceberg)
    silver_table = "secom_catalog.silver.silver_secom_reporting"
    silver_df = spark.table(silver_table)

    max_ts_row = silver_df.select(spark_max("process_timestamp").alias("max_ts")).collect()[0]
    max_ts = max_ts_row["max_ts"]
    max_ts_str = max_ts.strftime("%Y-%m-%d %H:%M:%S")

    if not max_ts:
        logger.error("The silver table is empty. Please run the ingestor first.")
        spark.stop()
        sys.exit(1)
    
    cutoff = max_ts - timedelta(days=args.lookback_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info("Max timestamp in data: %s", max_ts_str)
    logger.info("Reading with lookback=%d days (cutoff: %s)", args.lookback_days, cutoff_str)

    # Apply lookback filter
    filtered_df = silver_df.filter(
        col("process_timestamp") >= lit(cutoff_str)
    )

    total_rows = filtered_df.count()
    logger.info("Silver rows in lookback window: %d", total_rows)

    if total_rows < args.min_rows:
        logger.error(
            "Insufficient data: %d rows < minimum %d. "
            "Increase --lookback-days or run the data generator.",
            total_rows, args.min_rows
        )
        spark.stop()
        sys.exit(1)


    # ── Dynamic Feature Selection
    all_cols = filtered_df.columns
    candidate_cols = [c for c in all_cols if c not in META_COLS]
    logger.info("Evaluating %d candidate sensor columns...", len(candidate_cols))
    
    agg_exprs = []
    for c in candidate_cols:
        # Protect column names with backticks in case they are pure numbers (like "59")
        safe_col = col(f"`{c}`")
        
        # 1. Count Missing (Nulls or NaNs)
        agg_exprs.append(
            spark_sum(when(isnull(safe_col) | isnan(safe_col), 1).otherwise(0)).alias(f"{c}_missing")
        )
        # 2. Calculate Variance
        agg_exprs.append(
            variance(safe_col).alias(f"{c}_var")
        )

    # Execute the single-pass aggregation
    stats_row = filtered_df.select(*agg_exprs).collect()[0]

    # Evaluate the results against our thresholds
    feature_status = {}
    active_count = 0

    for c in candidate_cols:
        missing_count = stats_row[f"{c}_missing"] or 0
        missing_rate = missing_count / total_rows
        var_val = stats_row[f"{c}_var"]

        if missing_rate > args.missing_threshold:
            feature_status[c] = "dropped_40%_missing"
        elif var_val is None or var_val == 0.0:
            feature_status[c] = "dropped_0_variance"
        else:
            feature_status[c] = "active"
            active_count += 1

    logger.info("Evaluated feature statuses. Total Active: %d", active_count)

    # ── Select and transform columns
    # Backtick-quote numeric column names for PySpark compatibility
    sensor_select = [col(f"`{s}`").alias(s) for s in candidate_cols]
    meta_select = [col(c) for c in SELECTED_META_COLS if c in all_cols]

    feature_df = (
        filtered_df
        .select(*meta_select, *sensor_select)
        .filter(col("label_numeric").isNotNull())

        # Remap label: -1 → 0 (Pass), 1 → 1 (Fail)
        .withColumn(
            "binary_label",
            when(col("label_numeric") == 1, 1).otherwise(0)
        )
        .withColumn(
            "missing_sensor_rate",
            spark_round(col("missing_sensor_count").cast("double") / lit(590.0), 4)
        )
        .withColumn("snapshot_timestamp", current_timestamp())
        .withColumn("snapshot_lookback_days", lit(args.lookback_days))
        .withColumn("feature_extraction_version", lit(EXTRACT_PIPELINE_VERSION))
        .orderBy("process_timestamp")
    )

    # ── Write to Iceberg ml namespace
    spark.sql("CREATE NAMESPACE IF NOT EXISTS secom_catalog.ml")

    output_table = args.output_table
    tables_exists = spark.catalog.tableExists(output_table)
    
    # Append schema-binds the data to the Iceberg table
    writer = (
        feature_df.writeTo(output_table)
        .tableProperty("write.format.default", "parquet")
        )
    
    if tables_exists:
        writer.overwritePartitions()
        logger.info("Overwrote existing feature snapshot at %s", output_table)
    else:
        writer.create()
        logger.info("Created new feature snapshot at %s", output_table)

    manifest = {
        "extraction_pipeline_version": EXTRACT_PIPELINE_VERSION,
        "lookback_start_date": cutoff_str,
        "lookback_end_date": max_ts_str,
        "lookback_days": args.lookback_days,
        "feature_status": feature_status,
        "extracted_at": datetime.now(timezone.utc).isoformat()
    }

    manifest_path = "/tmp/feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved initial feature selection to manifest: %s", manifest_path)
 
    logger.info("Step 1 complete — feature snapshot written.")
    spark.stop()

if __name__ == "__main__":
    main()