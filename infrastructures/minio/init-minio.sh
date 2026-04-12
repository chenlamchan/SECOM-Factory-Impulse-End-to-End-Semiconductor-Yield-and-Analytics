#!/bin/bash
set -e # Fail immediately if any command fails

echo 'Setting up MinIO client (mc)...'
# FIX: Use single $ for shell command substitution
MINIO_USER=$(cat /run/secrets/minio_user)
MINIO_PASS=$(cat /run/secrets/minio_password)

# Ensure BUCKET_NAME has a fallback value if the env var is missing
BUCKET_NAME=${MINIO_BUCKET:-data-lake}

# FIX: Use single $ for variables
mc alias set secom-minio http://minio:9000 ${MINIO_USER} ${MINIO_PASS}

echo 'Creating MinIO buckets...'
# Use --ignore-existing to avoid "Bucket already exists" errors cleanly
mc mb --ignore-existing secom-minio/${BUCKET_NAME}

echo 'Setting MinIO bucket policy...'
# Updated to use 'anonymous' instead of deprecated 'policy'
mc anonymous set public secom-minio/${BUCKET_NAME}

echo "Generating Prometheus Token..."
JSON_OUTPUT=$(mc admin prometheus generate secom-minio --json)

# Parse JSON using Bash string manipulation (no extra tools needed)
CLEAN=${JSON_OUTPUT//\"/}   # Remove all quotes
TOKEN_TEMP=${CLEAN#*: }     # Remove everything before the colon
TOKEN=${TOKEN_TEMP%\}}      # Remove the trailing brace

echo $TOKEN > /secrets/minio_prometheus_token
echo "✓ Token saved to /secrets/minio_prometheus_token"

echo 'MinIO init complete.'