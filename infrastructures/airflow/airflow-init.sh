#!/bin/bash
set -e

# 1. Securely construct the DB Connection String in memory
DB_PASSWORD=$(cat /run/secrets/airflow_db_password | tr -d '\n\r ')
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://airflow:${DB_PASSWORD}@postgres/${CATALOG_NAME}"

echo "Running Airflow DB migrations..."
airflow db migrate

echo "Creating Airflow Admin user..."
# Read the admin password from the mounted Docker secret
if [ -f "/run/secrets/airflow_admin_password" ]; then
    ADMIN_PASSWORD=$(cat /run/secrets/airflow_admin_password | tr -d '\n\r ')
else
    echo "ERROR: Secret /run/secrets/airflow_admin_password not found!"
    exit 1
fi

# Use environment variables for user/email with fallbacks
ADMIN_USER=${AIRFLOW_ADMIN_USER}
ADMIN_EMAIL=${AIRFLOW_ADMIN_EMAIL:-admin@example.com}

set +e

# Create user ONLY if it doesn't exist
airflow users create \
    --role Admin \
    --username "$ADMIN_USER" \
    --password "$ADMIN_PASSWORD" \
    --email "$ADMIN_EMAIL" \
    --firstname Admin \
    --lastname User

if [ $? -eq 0 ]; then
    echo "Admin user created successfully."
else
    echo "Admin user already exists or creation bypassed."
fi

set -e

create_spark_conn() {
  airflow connections add 'spark_default' \
    --conn-type 'spark' \
    --conn-host 'spark://spark-master' \
    --conn-port '7077' \
    --conn-extra '{
      "queue": "default",
      "spark-binary": "spark-submit",
      "deploy-mode": "client"
    }'
}

airflow connections get spark_default > /dev/null 2>&1 || create_spark_conn

echo "Airflow initialization complete!"