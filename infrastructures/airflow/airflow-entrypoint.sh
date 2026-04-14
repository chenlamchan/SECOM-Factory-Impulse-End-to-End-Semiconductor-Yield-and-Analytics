#!/bin/bash
set -e

# Securely construct the DB Connection String in memory before Airflow starts
DB_PASSWORD=$(cat /run/secrets/airflow_db_password | tr -d '\n\r ')
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://airflow:${DB_PASSWORD}@postgres/${CATALOG_NAME}"

# Execute the original standard Airflow entrypoint, passing along any commands (like 'webserver' or 'scheduler')
exec /entrypoint "$@"