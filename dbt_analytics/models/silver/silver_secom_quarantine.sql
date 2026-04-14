{{ config(
    materialized='table'
) }}

WITH staged_data AS (
    SELECT * FROM {{ ref('stg_bronze_secom') }}
)

SELECT 
    observation_id,
    process_timestamp,
    missing_sensor_count,
    'CRITICAL_NULL_THRESHOLD_EXCEEDED' AS quarantine_reason,
    CURRENT_TIMESTAMP() AS quarantined_at
FROM staged_data
-- Capture everything that failed the reporting criteria
WHERE missing_sensor_count >= 59