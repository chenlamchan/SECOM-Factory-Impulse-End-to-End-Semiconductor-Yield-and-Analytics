{{ config(
    materialized='table',
    partition_by=['wafer_status']
) }}

WITH staged_data AS (
    SELECT * FROM {{ ref('stg_bronze_secom') }}
)

SELECT 
    *,
    CURRENT_TIMESTAMP() AS silver_created_at,
    'dbt_reporting_v1' AS processing_logic
FROM staged_data
-- Data Contract: Only allow rows with less than 10% missing sensor data (59 features)
WHERE missing_sensor_count < 59