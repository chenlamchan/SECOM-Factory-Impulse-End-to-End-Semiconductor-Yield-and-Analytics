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
NATS_URL = config.nats_endpoint
STREAM_NAME = config.nats_stream_name
SUBJECT = config.nats_subject
CONSUMER_NAME = "airflow_bronze_ingestion_consumer"
BATCH_THRESHOLD = 5
DBT_PROJECT_PATH = os.environ.get('DBT_PROJECT_PATH', '/opt/airflow/dbt_analytics')
MINIO_USER_FILEPATH = os.environ.get('MINIO_USER_FILEPATH')
MINIO_PWD_FILEPATH = os.environ.get('MINIO_PWD_FILEPATH')
AIRFLOW_DB_PWD_FILEPATH = os.environ.get('AIRFLOW_DB_PWD_FILEPATH')

async def _pull_nats_messages(**context):
    """Async worker to connect to NATS JetStream and pull batch paths."""
    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()

    consumer_config = ConsumerConfig(
        durable_name=CONSUMER_NAME,
        filter_subject=SUBJECT,
        ack_policy=AckPolicy.EXPLICIT,
        ack_wait=600,
        max_deliver=3
    )

    # Idempotent consumer creation
    try:
        await js.add_consumer(STREAM_NAME, consumer_config)
    except Exception as e:
        logger.debug(f"Consumer already exists or check: {e}")

    sub = await js.pull_subscribe(SUBJECT, CONSUMER_NAME)
    
    file_paths = []
    
    try:
        # Attempt to fetch up to BATCH_THRESHOLD messages. Timeout prevents infinite hanging.
        msgs = await sub.fetch(BATCH_THRESHOLD, timeout=5)
        
        for msg in msgs:
            payload = json.loads(msg.data.decode())
            raw_path = payload.get("file_path", "")

            clean_path = raw_path.replace("s3a://", "").replace("s3://", "")

            file_paths.append(clean_path)
            
            await msg.ack()
            logger.info(f"Acked NATS message, path queued: {raw_path}")
            
    except TimeoutError:
        logger.info("No new messages found in JetStream or timeout reached.")
    
    await nc.close()
    
    # If no files were fetched, halt the DAG run gracefully
    if not file_paths:
        logger.info("Batch threshold not met or stream is empty. Halting downstream tasks.")
        return False 
    
    # Push the comma-separated paths to XCom for the Spark task
    paths_string = ",".join(file_paths)
    logger.info(f"Passing paths to Spark: {paths_string}")
    context['ti'].xcom_push(key='bronze_file_paths', value=paths_string)
    
    return True

def pull_from_nats_sync(**kwargs):
    """Synchronous wrapper for Airflow's PythonOperator."""
    return asyncio.run(_pull_nats_messages(**kwargs))


default_args = {
    'owner': 'data_engineering',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
}

with DAG(
    'secom_ingestion_processsing_event_driven',
    default_args=default_args,
    description='Event-driven ingestion from NATS to Iceberg Bronze',
    schedule_interval=timedelta(minutes=1), # Wakes up every 5 mins to check queue
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['medallion', 'bronze', 'secom'],
) as dag:

    # 1. Pull Paths & Gatekeeper
    # ShortCircuitOperator skips downstream tasks if the python callable returns False
    pull_nats_batches = ShortCircuitOperator(
        task_id='pull_nats_batches',
        python_callable=pull_from_nats_sync,
    )

    # 2. Spark Job: Raw to Bronze
    # We pull the paths from XCom and pass them as an application argument
    ingest_to_bronze = DockerOperator(
        task_id='pyiceberg_raw_to_bronze',
        image='pyiceberg-ingestor:latest',
        command=[
            "/app/ingest_bronze.py",
            "--file-paths", "{{ ti.xcom_pull(key='bronze_file_paths', task_ids='pull_nats_batches') }}"
        ],
        network_mode='end-to-end-semiconductor-yield-and-analytics_default',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        mounts=[
            Mount(
                source=MINIO_USER_FILEPATH,
                target="/run/secrets/minio_user",
                type="bind",
                read_only=True,
            ),
            Mount(
                source=MINIO_PWD_FILEPATH,
                target="/run/secrets/minio_password",
                type="bind",
                read_only=True,
            ),
            Mount(
                source=AIRFLOW_DB_PWD_FILEPATH,
                target="/run/secrets/airflow_db_password",
                type="bind",
                read_only=True,
            ),
        ],
        environment={
            "MINIO_ENDPOINT": "http://minio:9000",
            "MINIO_ACCESS_KEY_FILE": "/run/secrets/minio_user",
            "MINIO_SECRET_KEY_FILE": "/run/secrets/minio_password",
            "AIRFLOW_DB_PASSWORD_FILE": "/run/secrets/airflow_db_password",
            "CATALOG_NAME": "data_catalog",
            "CATALOG_USER": "airflow",
        }
    )

    build_silver_gold_reporting = DockerOperator(
        task_id='dbt_run_silver_gold_reporting',
        image='end-to-end-semiconductor-yield-and-analytics-dbt:latest',
        command='bash -c "dbt deps && dbt run --profiles-dir . --select models/staging models/silver models/gold"',
        working_dir='/dbt',
        mounts=[
            Mount(
                source=DBT_PROJECT_PATH,  # must be absolute path
                target='/dbt',
                type='bind'
            )
        ],
        network_mode='end-to-end-semiconductor-yield-and-analytics_default',  # same network as trino
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
    )

    pull_nats_batches >> ingest_to_bronze >> build_silver_gold_reporting 