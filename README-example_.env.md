```
# =============================================================================
# MINIO CONFIGURATION
# =============================================================================

MINIO_ENDPOINT=http://minio:9000
MINIO_PATH_STYLE_ACCESS=True
MINIO_BUCKET=s3a://data-lake

MINIO_WAREHOUSE=s3a://warehouse
MINIO_WAREHOUSE_S3=s3://warehouse

MINIO_MLFLOW = s3a://mlflow-artifacts
MINIO_EVIDENTLY = s3a://evidently-reports
MINIO_PREDICTIONS = s3a://ml-predictions
MINIO_ML_METADATA=s3a://ml-metadata

# =============================================================================
# NATS CONFIGURATION
# =============================================================================

NATS_ENDPOINT=nats://nats:4222
NATS_SUBJECT = "secom.data.generated"
NATS_STREAM_NAME = "SECOM_PIPELINE"

# =============================================================================
# POSTGRES
# =============================================================================

CATALOG_NAME=data_catalog

# =============================================================================
# AIRFLOW
# =============================================================================
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_EMAIL=data.team@example.com
AIRFLOW__WEBSERVER__SECRET_KEY=abc123

DOCKER_GID=1001

MANIFEST_S3_URI="s3://ml-metadata/manifests/feature_manifest.json"
```
