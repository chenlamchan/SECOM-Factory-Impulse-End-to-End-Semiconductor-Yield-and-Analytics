"""
secom_ml_training_dag.py — MLOps Level 2 Training Pipeline
────────────────────────────────────────────────────────────
Full automated training pipeline aligned with Google MLOps Level 2:
 
  Step 1 — extract_features    : PySpark reads silver Iceberg → writes ml.feature_snapshot
 
All steps run as DockerOperator using secom-ml-trainer:latest.
PySpark steps connect to the existing Spark cluster (spark://spark-master:7077).
"""

import os
import json
import asyncio
import logging
import nats
from nats.errors import TimeoutError
from nats.js.api import ConsumerConfig, AckPolicy
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from airflow.operators.python import ShortCircuitOperator, PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from common.config import ServiceConfig

logger = logging.getLogger(__name__)

config = ServiceConfig()

# Config
COMMON_ENV = {

}
NATS_URL = config.nats_endpoint
STREAM_NAME = None
SUBJECT = None
CONSUMER_NAME = "airflow_ml_trainer_consumer"

DBT_PROJECT_PATH = os.environ.get('DBT_PROJECT_PATH', '/opt/airflow/dbt_analytics')
MINIO_USER_FILEPATH = os.environ.get('MINIO_USER_FILEPATH')
MINIO_PWD_FILEPATH = os.environ.get('MINIO_PWD_FILEPATH')
AIRFLOW_DB_PWD_FILEPATH = os.environ.get('AIRFLOW_DB_PWD_FILEPATH')

SECRET_MOUNTS = [
    Mount(source=MINIO_USER_FILEPATH, target="/run/secrets/minio_user", type="bind", read_only=True),
    Mount(source=MINIO_PWD_FILEPATH, target="/run/secrets/minio_password", type="bind", read_only=True),
    Mount(source=AIRFLOW_DB_PWD_FILEPATH, target="/run/secrets/airflow_db_password", type="bind", read_only=True),
]

default_args = {
    "owner": "ml_engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}

with DAG(
    "secom_ml_training_pipeline",
    default_args=default_args,
    description="MLOps Level 2 — automated training pipeline with champion gate",
    schedule_interval=timedelta(minutes=5), # Wakes up every 5 mins to check queue
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "xgboost", "lightgbm", "mlflow", "secom"],
) as dag:

    # ── Step 1: Data Extraction ─────────────────────────────────────────────
    # PySpark reads secom_catalog.silver.secom_reporting → ml.feature_snapshot
    # Connects to Spark cluster for distributed Iceberg read.
    extract_features = SparkSubmitOperator(
        task_id="extract_features",
        conn_id='spark_default', 
        application='/opt/airflow/spark_jobs/extract_features.py',
        name='secom_extract_features',
        application_args=[
            "--lookback-days", "90",
            "--min-rows", "500",
            "--missing-threshold", "0.40"
        ],
        conf={
            "spark.executor.memory": "2g",
            "spark.executor.cores": "2",
            "spark.cores.max": "2",
            "spark.executor.memoryOverhead": "512m",
            "spark.jars.packages": (
                "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.7.0,"
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "org.postgresql:postgresql:42.6.0"
            ),
        },
    )

    extract_features