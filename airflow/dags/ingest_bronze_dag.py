import json
import asyncio
import logging
import nats
from nats.errors import TimeoutError
from nats.js.api import ConsumerConfig, AckPolicy
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
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
            
            # Spark requires s3a:// prefix for the S3AFileSystem, but the generator outputs s3://
            if raw_path.startswith("s3://"):
                raw_path = raw_path.replace("s3://", "s3a://", 1)
            
            file_paths.append(raw_path)
            
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
    'secom_bronze_ingestion_event_driven',
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
    ingest_to_bronze = SparkSubmitOperator(
        task_id='spark_raw_to_bronze',
        conn_id='spark_default', 
        application='/opt/airflow/spark_jobs/ingest_bronze.py',
        name='secom_ingest_bronze',
        application_args=[
            "--file-paths", 
            "{{ ti.xcom_pull(key='bronze_file_paths', task_ids='pull_nats_batches') }}"
        ],
        conf={
            "spark.executor.memory": "1g",
            "spark.executor.cores": "2",
            "spark.cores.max":"2",
            "spark.executor.memoryOverhead": "512m",

            "spark.jars.packages": (
                "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.7.0,"
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "org.postgresql:postgresql:42.6.0"
            ),
            # Crucial for Iceberg
            "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            "spark.sql.catalog.secom_catalog": "org.apache.iceberg.spark.SparkCatalog",
            "spark.sql.catalog.secom_catalog.warehouse": "s3a://data-lake/warehouse",
        }
    )

    pull_nats_batches >> ingest_to_bronze