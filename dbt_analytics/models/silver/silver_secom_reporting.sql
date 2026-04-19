{{ config(materialized='table') }}

WITH staged_data AS (
    SELECT * FROM {{ ref('stg_bronze_secom') }}
)

SELECT 
    *,
    CURRENT_TIMESTAMP AS silver_created_at,
    'dbt_reporting_v1' AS processing_logic
FROM staged_data
-- Data Contract: Only allow rows with less than 20% missing sensor data (118 features)
WHERE missing_sensor_count < 118