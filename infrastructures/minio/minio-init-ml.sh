#!/bin/bash
# init-minio-ml.sh
# Creates the dedicated mlflow-artifacts bucket in MinIO.
# Extends the main init-minio.sh — run after the main script.
#
# Usage:
#   docker exec -it minio-init sh /init-minio-ml.sh
#
set -e

echo "Setting up MLflow artifacts bucket in MinIO..."

MINIO_USER=$(cat /run/secrets/minio_user | tr -d '\n\r ')
MINIO_PASS=$(cat /run/secrets/minio_password | tr -d '\n\r ')

MLFLOW_BUCKET=${MINIO_MLFLOW:-mlflow-artifacts}
EVIDENTLY_BUCKET=${MINIO_EVIDENTLY:-evidently-reports}
ML_PREDICTIONS_BUCKET=${MINIO_PREDICTIONS:-ml-predictions}
ML_METADATA_BUCKET=${MINIO_ML_METADATA:-ml-metadata}

mc alias set secom-minio http://minio:9000 "${MINIO_USER}" "${MINIO_PASS}"

# MLflow artifact store
mc mb --ignore-existing "secom-minio/${MLFLOW_BUCKET}"
mc anonymous set private "secom-minio/${MLFLOW_BUCKET}"
echo "✓ Created bucket: ${MLFLOW_BUCKET}"

# Evidently drift reports
mc mb --ignore-existing "secom-minio/${EVIDENTLY_BUCKET}"
mc anonymous set private "secom-minio/${EVIDENTLY_BUCKET}"
echo "✓ Created bucket: ${EVIDENTLY_BUCKET}"

# ML batch predictions (written by batch_inference.py, read by dbt)
mc mb --ignore-existing "secom-minio/${ML_PREDICTIONS_BUCKET}"
mc anonymous set private "secom-minio/${ML_PREDICTIONS_BUCKET}"
echo "✓ Created bucket: ${ML_PREDICTIONS_BUCKET}"

# ML Metadata (written by training-dag.py, read within airflow pipelines)
mc mb --ignore-existing "secom-minio/${ML_METADATA_BUCKET}"
mc anonymous set private "secom-minio/${ML_METADATA_BUCKET}"
echo "✓ Created bucket: ${ML_METADATA_BUCKET}"

echo "Uploading holdset test data..."
mc cp uci-secom.csv "secom-minio/${ML_METADATA_BUCKET}/holdset-test-data/"
echo "✓ Uploaded CSV to ${ML_METADATA_BUCKET}/holdout-test-data/"

echo "MinIO ML setup complete."