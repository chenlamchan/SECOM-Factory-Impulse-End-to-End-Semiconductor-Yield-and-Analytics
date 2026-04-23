"""
secom_scoring_dag.py — MLOps Batch Inference Pipeline
────────────────────────────────────────────────────────
Scores recent un-scored records from the silver reporting table using 
the current champion model.

Workflow:
  1. batch_inference  : Python script runs in the ML container, queries Iceberg,
                        predicts using the champion model, writes to ml.predictions.
  2. update_gold_layer: dbt task materializes the predictions with actuals into 
                        gold_model_predictions.sql.
"""

import os
import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

logger = logging.getLogger(__name__)

# Config
COMMON_ENV = {
    "MINIO_ENDPOINT": "http://minio:9000",
    "MINIO_ACCESS_KEY_FILE": "/run/secrets/minio_user",
    "MINIO_SECRET_KEY_FILE": "/run/secrets/minio_password",
    "AIRFLOW_DB_PASSWORD_FILE": "/run/secrets/airflow_db_password",

    "CATALOG_NAME": "data_catalog",
    "CATALOG_USER": "airflow",
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",

    "TRINO_CATALOG": "secom_catalog",
    "TRINO_HOST": "trino",
    "TRINO_PORT": "8080",
    "TRINO_USER": "airflow"
}

DBT_PROJECT_PATH = os.environ.get('DBT_PROJECT_PATH', '/opt/airflow/dbt_analytics')
MINIO_USER_FILEPATH = os.environ.get('MINIO_USER_FILEPATH')
MINIO_PWD_FILEPATH = os.environ.get('MINIO_PWD_FILEPATH')
AIRFLOW_DB_PWD_FILEPATH = os.environ.get('AIRFLOW_DB_PWD_FILEPATH')
BATCH_JOBS_PATH = os.environ.get('BATCH_JOBS_PATH') # Assuming batch_inference.py is here

DOCKER_MOUNTS = [
    Mount(source=MINIO_USER_FILEPATH, target="/run/secrets/minio_user", type="bind", read_only=True),
    Mount(source=MINIO_PWD_FILEPATH, target="/run/secrets/minio_password", type="bind", read_only=True),
    Mount(source=AIRFLOW_DB_PWD_FILEPATH, target="/run/secrets/airflow_db_password", type="bind", read_only=True),
    Mount(source=BATCH_JOBS_PATH, target="/inference", type="bind", read_only=True),
]

default_args = {
    "owner": "ml_engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "secom_ml_scoring_pipeline",
    default_args=default_args,
    description="Batch scores silver records using the Champion model",
    schedule_interval="0 2 * * *", # Runs every day at 2:00 AM (Adjust as needed)
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "inference", "scoring", "secom"],
) as dag:

    # ── Step 1: Batch Inference ──────────────────────────────────────────────
    run_batch_inference = DockerOperator(
        task_id="run_batch_inference",
        image="secom-ml-trainer:latest", # Uses your existing lightweight ML container
        api_version="auto",
        auto_remove=True,
        command="python /inference/batch_inference.py --lookback-days 7 --model-alias champion",
        docker_url="unix://var/run/docker.sock",
        environment=COMMON_ENV,
        mounts=DOCKER_MOUNTS,
        mount_tmp_dir=False,
        network_mode='end-to-end-semiconductor-yield-and-analytics_default',
    )

    # ── Step 2: Update Gold Layer ────────────────────────────────────────────
    update_gold_predictions = DockerOperator(
        task_id='dbt_run_gold_predictions',
        image='end-to-end-semiconductor-yield-and-analytics-dbt:latest',
        command='bash -c "dbt deps && dbt run --profiles-dir . --select gold_model_predictions"',
        working_dir='/dbt',
        mounts=[
            Mount(
                source=DBT_PROJECT_PATH, 
                target='/dbt',
                type='bind'
            )
        ],
        network_mode='end-to-end-semiconductor-yield-and-analytics_default',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
    )

    run_batch_inference >> update_gold_predictions