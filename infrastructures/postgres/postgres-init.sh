#!/bin/sh
set -e

SUPER_USER=$(cat /run/secrets/postgres_user | tr -d '\n\r ')
SUPER_PASS=$(cat /run/secrets/postgres_password | tr -d '\n\r ')
AIRFLOW_PASS=$(cat /run/secrets/airflow_db_password | tr -d '\n\r ')
TARGET_DB=${CATALOG_NAME}

export PGPASSWORD="$SUPER_PASS"

psql -h postgres -U "$SUPER_USER" -d postgres \
  -v targetdb="$TARGET_DB" \
  -v airflow_pass="$AIRFLOW_PASS" <<-'EOSQL'
    -- Create airflow role if not exists
    SELECT format('CREATE ROLE airflow WITH LOGIN PASSWORD %L', :'airflow_pass')
    WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'airflow')\gexec

    -- Create DB if not exists (may already exist from POSTGRES_DB auto-creation)
    SELECT format('CREATE DATABASE %I OWNER airflow', :'targetdb')
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'targetdb')\gexec

    -- Always ensure correct ownership & privileges regardless of who created it
    SELECT format('ALTER DATABASE %I OWNER TO airflow', :'targetdb')\gexec
    SELECT format('GRANT ALL PRIVILEGES ON DATABASE %I TO airflow', :'targetdb')\gexec
EOSQL

# Connect to the airflow DB specifically to grant schema privileges
psql -h postgres -U "$SUPER_USER" -d "$TARGET_DB" <<-'EOSQL'
    GRANT ALL ON SCHEMA public TO airflow;
    ALTER SCHEMA public OWNER TO airflow;
EOSQL

echo "Postgres initialization complete."