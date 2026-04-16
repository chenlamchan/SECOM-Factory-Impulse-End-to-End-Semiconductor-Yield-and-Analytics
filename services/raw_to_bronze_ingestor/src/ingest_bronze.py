import os
import argparse
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.dataset as ds
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NoSuchTableError, NamespaceAlreadyExistsError
from datetime import datetime
from datetime import timezone
from config import ServiceConfig

config = ServiceConfig()

# Constants
MINIO_ENDPOINT = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key
CATALOG_URI = config.catalog_uri
CATALOG_USER = config.catalog_user
CATALOG_PASSWORD = config.catalog_password
S3_WAREHOUSE_PATH = config.minio_warehouse_s3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-paths', required=True)
    args = parser.parse_args()

    # 1. Initialize PyIceberg Catalog connecting to your existing Postgres
    catalog = SqlCatalog(
        "secom_catalog",
        uri=CATALOG_URI,
        warehouse=S3_WAREHOUSE_PATH,
        **{
            "s3.endpoint": MINIO_ENDPOINT,
            "s3.access-key-id": MINIO_ACCESS_KEY,
            "s3.secret-access-key": MINIO_SECRET_KEY,
            "s3.path-style-access": "true",
            "s3.region": "us-east-1"
        }
    )

    # 2. Read Raw Data via PyArrow (Lightning fast, infers schema)
    # PyArrow handles S3 naturally
    s3_fs = pa.fs.S3FileSystem(
        endpoint_override=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        scheme="http"
    )
    
    paths = args.file_paths.split(',')

    # use large_string in the HivePartitioning to prevent the "Unable to merge" error during the PyArrow read
    partitioning = ds.HivePartitioning(
        pa.schema([
            ("line_id", pa.large_string()),
            ("year", pa.large_string()),
            ("month", pa.large_string()),
            ("day", pa.large_string()),
        ])
    )

    dataset = ds.dataset(paths, filesystem=s3_fs, format="parquet", partitioning=partitioning)

    scanner = dataset.scanner(columns=dataset.schema.names + ["__filename"])
    arrow_table = scanner.to_table()

    new_fields = []
    for field in arrow_table.schema:
        name = "source_file" if field.name == "__filename" else field.name
        if field.type == pa.large_string():
            new_fields.append(pa.field(name, pa.string(), nullable=field.nullable)) # Iceberg uses utf8 (standard string)
        else:
            new_fields.append(pa.field(name, field.type, nullable=field.nullable))

    renamed_table = arrow_table.rename_columns([f.name for f in new_fields])
    final_table = renamed_table.cast(pa.schema(new_fields))

    # 3. Append Metadata (PyArrow compute functions)
    # E.g., adding ingestion timestamps
    num_rows = final_table.num_rows
    ingestion_ts = [datetime.now(timezone.utc)] * num_rows
    final_table = final_table.append_column("ingestion_timestamp", pa.array(ingestion_ts, pa.timestamp('us')))
    final_table = final_table.append_column("pipeline_version", pa.array(["0.1.0-pyiceberg"] * num_rows))

    table_identifier = "bronze.secom_data"
    
    try:
    # 4. Append to Iceberg
        table = catalog.load_table(table_identifier)
        print(f"Table '{table_identifier}' found. Appending data...")

    except NoSuchTableError:
        print(f"Table '{table_identifier}' does not exist. Creating it now...")

        try:
            catalog.create_namespace("bronze")
        except NamespaceAlreadyExistsError:
            pass

        table = catalog.create_table(
            identifier=table_identifier,
            schema=final_table.schema,
            location=f"{S3_WAREHOUSE_PATH}/bronze/secom_data"
        )
        print(f"Successfully created Iceberg table '{table_identifier}'.")

    table.append(final_table)
    print(f"Successfully appended {num_rows} rows to Iceberg Bronze.")

if __name__ == "__main__":
    main()