#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

# Read secrets and strip newlines/carriage returns/spaces
DB_PASS=$(cat /run/secrets/mlflow_db_password | tr -d '\n\r ')
MINIO_KEY=$(cat /run/secrets/minio_user | tr -d '\n\r ')
MINIO_SEC=$(cat /run/secrets/minio_password | tr -d '\n\r ')

# Set AWS credentials for MinIO/S3 access
export AWS_ACCESS_KEY_ID="$MINIO_KEY"
export AWS_SECRET_ACCESS_KEY="$MINIO_SEC"

# Start MLflow server
# Using 'exec' ensures mlflow replaces the shell script as PID 1 for proper signal handling (graceful stops)
echo "Starting MLflow server..."
exec mlflow server \
  --backend-store-uri "postgresql+psycopg2://mlflow:${DB_PASS}@postgres/mlflow" \
  --default-artifact-root "s3://mlflow-artifacts/" \
  --host 0.0.0.0 \
  --port 5000 \
  --serve-artifacts