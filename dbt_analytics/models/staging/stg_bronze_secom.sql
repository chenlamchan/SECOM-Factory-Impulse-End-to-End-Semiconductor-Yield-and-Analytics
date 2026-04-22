{{ config(materialized='ephemeral') }}

WITH raw_bronze AS (
    SELECT * FROM {{ source('secom_catalog','secom_data') }}
)

SELECT
    {{ dbt_utils.generate_surrogate_key(['"Time"', 'source_file']) }} AS observation_id,
    CAST("Time" AS TIMESTAMP) AS process_timestamp,
    
    -- Translate binary labels to business logic
    CASE 
        WHEN "Pass/Fail" = -1 THEN 'Pass'
        WHEN "Pass/Fail" = 1 THEN 'Fail'
        ELSE 'Unknown'
    END AS wafer_status,
    "Pass/Fail" AS label_numeric,

    -- Dynamic macro to calculate nulls across all 590 features
    (
        {% for i in range(0, 590) %}
            CASE WHEN {{ i }} IS NULL THEN 1 ELSE 0 END
            {% if not loop.last %} + {% endif %}
        {% endfor %}
    ) AS missing_sensor_count,

    line_id,
    tester_id,
    shift,
    lot_id,
    
    -- Include all raw features for downstream selection
    {{ dbt_utils.star(from=source('secom_catalog','secom_data'), 
    except=[
        "Time", 
        "Pass/Fail", 
        "ingestion_timestamp",  
        "pipeline_version", 
        "line_id",
        "tester_id",
        "shift",
        "lot_id",
    ]) }},
    
    -- Lineage
    ingestion_timestamp AS bronze_ingested_at,
    pipeline_version AS bronze_pipeline_version

FROM raw_bronze