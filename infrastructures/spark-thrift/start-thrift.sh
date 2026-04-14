#!/bin/bash
set -e

echo "Starting Spark Thrift Server..."

# Dynamically read secrets from Docker's memory files
MINIO_ACCESS_KEY=$(cat /run/secrets/minio_user)
MINIO_SECRET_KEY=$(cat /run/secrets/minio_password)
AIRFLOW_PASSWORD=$(cat /run/secrets/airflow_db_password)

exec /opt/spark/bin/spark-submit \
  --class org.apache.spark.sql.hive.thriftserver.HiveThriftServer2 \
  --master spark://spark-master:7077 \
  --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
  --conf spark.sql.catalog.my_catalog=org.apache.iceberg.spark.SparkCatalog \
  --conf spark.sql.catalog.my_catalog.type=jdbc \
  --conf spark.sql.catalog.my_catalog.uri=jdbc:postgresql://postgres:5432/postgres \
  --conf spark.sql.catalog.my_catalog.jdbc.user=airflow \
  --conf spark.sql.catalog.my_catalog.jdbc.password="${AIRFLOW_PASSWORD}" \
  --conf spark.sql.catalog.my_catalog.warehouse=s3a://warehouse \
  --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 \
  --conf spark.hadoop.fs.s3a.access.key="${MINIO_ACCESS_KEY}" \
  --conf spark.hadoop.fs.s3a.secret.key="${MINIO_SECRET_KEY}" \
  --conf spark.hadoop.fs.s3a.path.style.access=true \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem