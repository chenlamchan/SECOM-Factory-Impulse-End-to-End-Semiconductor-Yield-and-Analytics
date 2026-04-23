"""
secom_ml_monitoring_dag.py — Unified Monitoring & Scoring Pipeline
────────────────────────────────────────────────────────────────────
Two DAGs in one file:

  1. secom_ml_daily_operations   — runs daily. 
     • Executes drift_monitor.py (computes PSI, publishes NATS alert).
     • Executes batch_inference.py (scores today's silver data).
     • Triggers dbt to refresh gold_model_predictions.

  2. secom_ml_drift_listener  — polls the NATS ml.drift.alert subject
     every 5 minutes. If a message is present, triggers the training
     pipeline DAG via Airflow REST API (Continuous Training).
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta

import nats
import requests
from airflow import DAG
from airflow.operators.python import ShortCircuitOperator, PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

logger = logging.getLogger(__name__)

# ─── Configuration & Paths ───────────────────────────────────────────────────
ML_TRAINER_IMAGE    = os.environ.get("ML_TRAINER_IMAGE", "secom-ml-trainer:latest")
MINIO_USER_FILE     = os.environ.get("MINIO_USER_FILEPATH")
MINIO_PWD_FILE      = os.environ.get("MINIO_PWD_FILEPATH")
AIRFLOW_DB_PWD_FILE = os.environ.get("AIRFLOW_DB_PWD_FILEPATH")
AIRFLOW_ADMIN_PWD_FILE = os.environ.get("AIRFLOW_ADMIN_PWD_FILEPATH")
NETWORK             = os.environ.get("COMPOSE_NETWORK", "end-to-end-semiconductor-yield-and-analytics_default")
NATS_URL            = os.environ.get("NATS_ENDPOINT", "nats://nats:4222")
NATS_SUBJECT        = "ml.drift.alert"
NATS_STREAM         = "ML_MONITORING"
NATS_CONSUMER       = "airflow_drift_listener"
AIRFLOW_API_BASE    = os.environ.get("AIRFLOW_API_BASE", "http://airflow-webserver:8080/api/v1")
TRAINING_DAG_ID     = "secom_ml_training_pipeline"

DBT_PROJECT_PATH    = os.environ.get("DBT_PROJECT_PATH", "/opt/airflow/dbt_analytics")
BATCH_JOBS_PATH     = os.environ.get("BATCH_JOBS_PATH") 
MONITORING_JOBS_PATH= os.environ.get("MONITORING_JOBS_PATH") 

COMMON_ENV = {
    "MLFLOW_TRACKING_URI":      "http://mlflow:5000",
    "MLFLOW_S3_ENDPOINT_URL":   "http://minio:9000",
    "MINIO_ENDPOINT":           "http://minio:9000",
    "MINIO_ACCESS_KEY_FILE":    "/run/secrets/minio_user",
    "MINIO_SECRET_KEY_FILE":    "/run/secrets/minio_password",
    "AIRFLOW_DB_PASSWORD_FILE": "/run/secrets/airflow_db_password",
    "AIRFLOW_ADMIN_PWD_FILE": "/run/secrets/airflow_admin_password",
    "CATALOG_NAME":             "data_catalog",       # PyIceberg Postgres DB
    "CATALOG_USER":             "airflow",
    "TRINO_CATALOG":            "secom_catalog",      # Trino connector name
    "TRINO_HOST":               "trino",
    "TRINO_PORT":               "8080",
    "SPARK_MASTER":             "spark://spark-master:7077",
    "MODEL_NAME":               "secom_yield_predictor",
    "MODEL_ALIAS":              "champion",           
    "DRIFT_PSI_THRESHOLD":      "0.20",
    "DRIFT_LOOKBACK_DAYS":      "7",
    "NATS_ENDPOINT":            NATS_URL,
    "EVIDENTLY_BUCKET":         "evidently-reports",
}

# Unified Mounts for Secrets, Inference, and Monitoring scripts
DOCKER_MOUNTS = [
    Mount(source=MINIO_USER_FILE,     target="/run/secrets/minio_user",         type="bind", read_only=True),
    Mount(source=MINIO_PWD_FILE,      target="/run/secrets/minio_password",      type="bind", read_only=True),
    Mount(source=AIRFLOW_DB_PWD_FILE, target="/run/secrets/airflow_db_password", type="bind", read_only=True),
    Mount(source=AIRFLOW_ADMIN_PWD_FILE, target="/run/secrets/airflow_admin_password", type="bind", read_only=True),
    Mount(source=BATCH_JOBS_PATH,     target="/inference",                      type="bind", read_only=True),
    Mount(source=MONITORING_JOBS_PATH,target="/monitoring",                     type="bind", read_only=True),
]

# ═══════════════════════════════════════════════════════════════════════════════
# DAG 1 — Daily Operations (Monitor -> Score -> dbt)
# ═══════════════════════════════════════════════════════════════════════════════
with DAG(
    "secom_ml_daily_operations",
    default_args={
        "owner":           "ml_engineering",
        "depends_on_past": False,
        "retries":         1,
        "retry_delay":     timedelta(minutes=1),
    },
    description="Daily Drift Monitoring and Batch Scoring",
    schedule_interval="0 2 * * *", # Runs every day at 2:00 AM 
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "monitoring", "scoring", "secom"],
) as daily_ops_dag:

    # Step 1: Run drift_monitor.py
    run_drift_monitor = DockerOperator(
        task_id="run_drift_monitor",
        image=ML_TRAINER_IMAGE,
        command="python /monitoring/drift_monitor.py",
        network_mode=NETWORK,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        mounts=DOCKER_MOUNTS,
        environment=COMMON_ENV,
        mount_tmp_dir=False,
        do_xcom_push=True,
    )

    # Step 2: Run batch inference to score recent silver data
    run_batch_inference = DockerOperator(
        task_id="run_batch_inference",
        image=ML_TRAINER_IMAGE,
        command="python /inference/batch_inference.py --lookback-days 7 --model-alias champion",
        network_mode=NETWORK,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        mounts=DOCKER_MOUNTS,
        environment=COMMON_ENV,
        mount_tmp_dir=False,
    )

    # Step 3: Refresh gold_model_predictions via dbt
    refresh_gold_predictions = DockerOperator(
        task_id="refresh_gold_predictions",
        image="end-to-end-semiconductor-yield-and-analytics-dbt:latest",
        command='bash -c "dbt deps && dbt run --profiles-dir . --select gold_model_predictions"',
        working_dir="/dbt",
        mounts=[Mount(source=DBT_PROJECT_PATH, target="/dbt", type="bind")],
        network_mode=NETWORK,
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
    )

    run_drift_monitor >> run_batch_inference >> refresh_gold_predictions


# ═══════════════════════════════════════════════════════════════════════════════
# DAG 2 — Drift alert listener (polls NATS, triggers retraining)
# ═══════════════════════════════════════════════════════════════════════════════
async def _check_nats_drift_alert() -> bool:
    try:
        from nats.js.api import ConsumerConfig, AckPolicy

        nc = await nats.connect(NATS_URL)
        js = nc.jetstream()

        consumer_config = ConsumerConfig(
            durable_name=NATS_CONSUMER,
            filter_subject=NATS_SUBJECT,
            ack_policy=AckPolicy.EXPLICIT,
            ack_wait=60,
            max_deliver=1,
        )
        try:
            await js.add_consumer(NATS_STREAM, consumer_config)
        except Exception:
            pass   

        sub = await js.pull_subscribe(NATS_SUBJECT, NATS_CONSUMER)

        try:
            msgs = await sub.fetch(1, timeout=3)
            if msgs:
                payload = json.loads(msgs[0].data.decode())
                await msgs[0].ack()
                await nc.close()
                logger.info("Drift alert received: %s", payload.get("alert_type"))
                return True
        except Exception:
            pass

        await nc.close()
        return False

    except Exception as e:
        logger.warning("NATS poll error: %s — defaulting to no alert.", e)
        return False


def poll_drift_alert_sync(**kwargs) -> bool:
    return asyncio.run(_check_nats_drift_alert())


def trigger_training_dag(**kwargs) -> None:
    admin_user = os.environ.get("AIRFLOW_ADMIN_USER", "admin")
    password_file = os.environ.get("AIRFLOW_ADMIN_PWD_FILE", "/run/secrets/airflow_admin_password")

    try:
        with open(password_file, "r") as f:
            admin_pass = f.read().strip()
    except Exception as e:
        logger.error(f"Could not read admin password file: {e}")
        raise ValueError("Missing Airflow admin password secret.")
    
    url = f"{AIRFLOW_API_BASE}/dags/{TRAINING_DAG_ID}/dagRuns"
    payload = {
        "conf": {
            "trigger_reason": "drift_alert",
            "source_dag":     "secom_ml_drift_listener",
        }
    }
    try:
        resp = requests.post(
            url, json=payload,
            auth=(admin_user, admin_pass),
            timeout=30,
        )
        resp.raise_for_status()
        run_id = resp.json().get("dag_run_id", "unknown")
        logger.info("Training pipeline triggered — dag_run_id=%s", run_id)
    except Exception as e:
        logger.error("Failed to trigger training DAG: %s", e)
        raise


with DAG(
    "secom_ml_drift_listener",
    default_args={
        "owner":           "ml_engineering",
        "depends_on_past": False,
        "retries":         0,
    },
    description="Polls NATS for drift alerts and triggers CT retraining",
    schedule_interval=timedelta(minutes=5),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "ct", "continuous-training", "nats"],
) as listener_dag:

    check_for_alert = ShortCircuitOperator(
        task_id="check_for_drift_alert",
        python_callable=poll_drift_alert_sync,
    )

    trigger_retrain = PythonOperator(
        task_id="trigger_retraining_pipeline",
        python_callable=trigger_training_dag,
    )

    check_for_alert >> trigger_retrain