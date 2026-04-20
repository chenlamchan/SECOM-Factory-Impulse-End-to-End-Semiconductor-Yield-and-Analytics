#!/bin/sh
# postgres-init-ml.sh
# Run once to create the dedicated MLflow database and role in the existing Postgres instance.
# Uses the same superuser secrets as the main postgres-init.sh.
#
# Usage (from docker-compose exec or a one-shot container):
#   docker exec -it postgres sh /docker-entrypoint-initdb.d/postgres-init-ml.sh
#
set -e

SUPER_USER=$(cat /run/secrets/postgres_user | tr -d '\n\r ')
SUPER_PASS=$(cat /run/secrets/postgres_password | tr -d '\n\r ')
MLFLOW_PASS=$(cat /run/secrets/mlflow_db_password | tr -d '\n\r ')

export PGPASSWORD="$SUPER_PASS"

echo "Creating MLflow Postgres role and database..."

psql -h postgres -U "$SUPER_USER" -d postgres \
  -v mlflow_pass="$MLFLOW_PASS" <<-'EOSQL'
    -- Create the mlflow role (idempotent)
    SELECT format('CREATE ROLE mlflow WITH LOGIN PASSWORD %L', :'mlflow_pass')
    WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mlflow')\gexec

    -- Create the mlflow database owned by the mlflow role
    SELECT 'CREATE DATABASE mlflow OWNER mlflow'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

    GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
EOSQL

# Grant schema privileges inside the mlflow database
psql -h postgres -U "$SUPER_USER" -d mlflow <<-'EOSQL'
    GRANT ALL ON SCHEMA public TO mlflow;
    ALTER SCHEMA public OWNER TO mlflow;
EOSQL

echo "MLflow Postgres setup complete."