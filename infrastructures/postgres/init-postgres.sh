#!/bin/sh
set -e

SUPER_USER=$(cat /run/secrets/postgres_user | tr -d '\n\r ')
SUPER_PASS=$(cat /run/secrets/postgres_password | tr -d '\n\r ')
TARGET_DB="${TARGET_DB:?TARGET_DB must be set}"

export PGPASSWORD="$SUPER_PASS"

echo "Connecting as superuser: $SUPER_USER to ensure DB: $TARGET_DB exists"

psql -h postgres -U "$SUPER_USER" -d postgres \
  -v targetdb="$TARGET_DB" <<-'EOSQL'
    SELECT format('CREATE DATABASE %I', :'targetdb')
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'targetdb')\gexec
EOSQL

echo "Postgres initialization complete."