import sys
import argparse
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit, input_file_name, to_date, col
from common.config import ServiceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = ServiceConfig()

# Constants
MINIO_ENDPOINT = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key
CATALOG_URI = config.catalog_uri
CATALOG_USER = config.catalog_user
CATALOG_PASSWORD = config.catalog_password
S3_WAREHOUSE_PATH = config.minio_warehouse

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
        .config("spark.sql.catalog.secom_catalog.warehouse", warehouse_path) \
        .config("spark.sql.catalog.secom_catalog.io-impl", "org.apache.iceberg.hadoop.HadoopFileIO") \
        .config("spark.hadoop.fs.s3a.endpoint", minio_endpoint) \
        .config("spark.hadoop.fs.s3a.access.key", minio_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", minio_secret_key) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.client.region", "us-east-1") \
        .getOrCreate()

def main():
    # Parse the exact file paths passed by Airflow
    parser = argparse.ArgumentParser(description="Ingest SECOM Raw to Bronze")
    parser.add_argument('--file-paths', required=True, help='Comma separated list of S3A paths')
    args = parser.parse_args()

    if not args.file_paths:
        logger.error("No file paths provided. Exiting.")
        sys.exit(1)

    from common.config import ServiceConfig
    config = ServiceConfig()

    file_paths_list = args.file_paths.split(',')
    logger.info(f"Target paths to process: {file_paths_list}")

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

    # 1. Read Raw Data directly from specific files
    try:
        # PySpark can accept a list of paths directly
        raw_df = spark.read.parquet(*file_paths_list)
        logger.info(f"Loaded {raw_df.count()} records from Raw storage.")
    except Exception as e:
        logger.error(f"Failed to read parquet files: {e}")
        spark.stop()
        sys.exit(1)

    # 2. Append Medallion Lineage Metadata
    bronze_df = raw_df \
        .withColumn("event_date", to_date(col("Time"))) \
        .withColumn("ingestion_timestamp", current_timestamp()) \
        .withColumn("ingestion_date", to_date(current_timestamp())) \
        .withColumn("source_file", input_file_name()) \
        .withColumn("pipeline_version", lit("0.1.0-event-driven"))

    # Generate synthetic observation_id (hash of timestamp and source)
    bronze_df.createOrReplaceTempView("bronze_temp")
    bronze_df = spark.sql("""
        SELECT 
            md5(concat(cast(Time as string), source_file)) as observation_id,
            * FROM bronze_temp
    """)

    # 3. Create Namespace & Append to Bronze Table
    spark.sql("CREATE NAMESPACE IF NOT EXISTS secom_catalog.bronze")
    
    tables_exists = spark.catalog.tableExists("secom_catalog.bronze.secom_data")
    
    # Append schema-binds the data to the Iceberg table
    writer = (bronze_df.writeTo("secom_catalog.bronze.secom_data") \
        .tableProperty("write.format.default", "parquet") \
        .partitionedBy("event_date")
        )
    
    if tables_exists:
        writer.append()
        logger.info("Appended batch to existing Iceberg Bronze table.")
    else:
        writer.create()
        logger.info("Created Iceberg Bronze table and wrote first batch.")
 


    logger.info("Successfully ingested event-driven batch into Iceberg Bronze layer.")
    spark.stop()

if __name__ == "__main__":
    main()