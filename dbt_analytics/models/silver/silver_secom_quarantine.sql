{{ config(
    materialized='table'
) }}

WITH staged_data AS (
    SELECT * FROM {{ ref('stg_bronze_secom') }}
)

SELECT 
    observation_id,
    line_id,
    shift,
    tester_id,
    lot_id,
    process_timestamp,
    missing_sensor_count,
    'CRITICAL_NULL_THRESHOLD_EXCEEDED' AS quarantine_reason,
    CURRENT_TIMESTAMP AS quarantined_at
FROM staged_data
-- Capture everything that failed the reporting criteria (20%)
WHERE missing_sensor_count >= 118