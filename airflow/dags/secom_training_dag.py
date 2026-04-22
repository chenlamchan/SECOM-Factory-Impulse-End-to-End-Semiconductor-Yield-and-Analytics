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
    "MINIO_ENDPOINT": "http://minio:9000",
    "MINIO_ACCESS_KEY_FILE": "/run/secrets/minio_user",
    "MINIO_SECRET_KEY_FILE": "/run/secrets/minio_password",
    "AIRFLOW_DB_PASSWORD_FILE": "/run/secrets/airflow_db_password",
    "CATALOG_NAME": "data_catalog",
    "CATALOG_USER": "airflow",
}

NATS_URL = config.nats_endpoint
STREAM_NAME = None
SUBJECT = None
CONSUMER_NAME = "airflow_ml_trainer_consumer"

MANIFEST_S3_URI = "s3://ml-metadata/manifests/feature_manifest.json"
HELD_OUT_CSV_S3_URI = "s3://ml-metadata/holdout-test-data/uci-secom.csv"
DBT_PROJECT_PATH = os.environ.get('DBT_PROJECT_PATH', '/opt/airflow/dbt_analytics')
MINIO_USER_FILEPATH = os.environ.get('MINIO_USER_FILEPATH')
MINIO_PWD_FILEPATH = os.environ.get('MINIO_PWD_FILEPATH')
AIRFLOW_DB_PWD_FILEPATH = os.environ.get('AIRFLOW_DB_PWD_FILEPATH')
HOST_SPARK_JOBS_PATH = os.environ.get('SPARK_JOB_FILEPATH')

DOCKER_MOUNTS = [
    Mount(source=MINIO_USER_FILEPATH, target="/run/secrets/minio_user", type="bind", read_only=True),
    Mount(source=MINIO_PWD_FILEPATH, target="/run/secrets/minio_password", type="bind", read_only=True),
    Mount(source=AIRFLOW_DB_PWD_FILEPATH, target="/run/secrets/airflow_db_password", type="bind", read_only=True),
    Mount(source=HOST_SPARK_JOBS_PATH, target="/spark_jobs", type="bind", read_only=True),
]

def _check_promotion(**context):
        """Checks the JSON output from validate_model to see if we should reload serving."""
        validation_output_str = context['ti'].xcom_pull(task_ids='validate_model')
        if not validation_output_str:
            return False
            
        try:
            # Parse the JSON string printed by validate_model
            result = json.loads(validation_output_str)
            if result.get("promoted") is True:
                logger.info(f"Model Promoted! Version: {result.get('version')}. Proceeding to deployment.")
                return True
            else:
                logger.info(f"Model Rejected. Reason: {result.get('reason')}. Halting pipeline.")
                return False
        except Exception as e:
            logger.error(f"Failed to parse validation output: {e}")
            return False

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
            "--lookback-days", "730",
            "--min-rows", "500",
            "--missing-threshold", "0.40",
            "--manifest-path", MANIFEST_S3_URI
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

    prepare_features = SparkSubmitOperator(
        task_id='prepare_features',
        conn_id='spark_default', 
        application='/opt/airflow/spark_jobs/prepare_features.py',
        name='secom_prepare_features',
        application_args=[
            "--test-ratio", "0.20",
            "--manifest-path", MANIFEST_S3_URI
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

    train_model = DockerOperator(
        task_id="train_model",
        image="secom-ml-trainer:latest",
        api_version="auto",
        auto_remove=True, # Clean up container after it finishes
        command=f"python /spark_jobs/train_model.py --manifest-path {MANIFEST_S3_URI}",
        docker_url="unix://var/run/docker.sock", # Connects to host Docker daemon
        environment=COMMON_ENV,
        mounts=DOCKER_MOUNTS,
        mount_tmp_dir=False,
        do_xcom_push=False, 
        network_mode='end-to-end-semiconductor-yield-and-analytics_default',
    )

    evaluate_model = DockerOperator(
        task_id="evaluate_model",
        image="secom-ml-trainer:latest",
        api_version="auto",
        auto_remove=True,
        command=(
            f"python /spark_jobs/evaluate_model.py "
            f"--manifest-path {MANIFEST_S3_URI} "
            f"--held-out-data {HELD_OUT_CSV_S3_URI} "
        ),
        docker_url="unix://var/run/docker.sock",
        environment=COMMON_ENV,
        mounts=DOCKER_MOUNTS,
        mount_tmp_dir=False,
        network_mode='end-to-end-semiconductor-yield-and-analytics_default',
    )

    validate_model = DockerOperator(
        task_id="validate_model",
        image="secom-ml-trainer:latest",
        api_version="auto",
        auto_remove=True,
        # Pulls run_id from train_model just like evaluate_model did
        command=(
            f"python /spark_jobs/validate_model.py "
            f"--manifest-path {MANIFEST_S3_URI} "
        ),
        docker_url="unix://var/run/docker.sock",
        environment=COMMON_ENV,
        mounts=DOCKER_MOUNTS,
        mount_tmp_dir=False,
        do_xcom_push=True, # Pushes the {"promoted": true/false} JSON to XCom
        network_mode='end-to-end-semiconductor-yield-and-analytics_default',
    )

    promotion_gate = ShortCircuitOperator(
        task_id="promotion_gate",
        python_callable=_check_promotion,
    )

    # If gate passes, trigger an API reload
    reload_serving_api = BashOperator(
        task_id="reload_serving_api",
        bash_command="curl -X POST http://ml-serving:8001/reload"
    )

    extract_features >> prepare_features >> train_model >> evaluate_model >> validate_model >> promotion_gate >> reload_serving_api