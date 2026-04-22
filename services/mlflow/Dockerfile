FROM ghcr.io/mlflow/mlflow:v3.11.1

# Install dependencies during build time
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir psycopg2-binary boto3