{{ 
    config(
        materialized= 'table',
        file_format= 'parquet'
    )
}}

WITH silver_data AS (
    SELECT * FROM {{ ref('silver_secom_reporting' )}}
),

daily_counts AS (
    SELECT
        CAST(process_timestamp AS DATE) AS process_date,
        COUNT(observation_id) AS total_wafers_tested,
        SUM(CASE WHEN wafer_status='Pass' THEN 1 ELSE 0 END) AS passed_wafers,
        SUM(CASE WHEN wafer_status='Fail' THEN 1 ELSE 0 END) AS failed_wafers
    FROM silver_data
    GROUP BY CAST(process_timestamp AS DATE)
),

process_kpis AS (
    SELECT 
        process_date,
        total_wafers_tested,
        passed_wafers,
        failed_wafers,

        -- Yield %: (Pass / Total) * 100
        ROUND(
            (passed_wafers / CAST(total_wafers_tested AS DOUBLE)) * 100, 
            2
        ) AS yield_percentage,

        -- PPM Defective: (Fail / Total) * 1,000,000
        ROUND(
            (failed_wafers / CAST(total_wafers_tested AS DOUBLE)) * 1000000, 
            0
        ) AS ppm_defective,

        CURRENT_TIMESTAMP AS last_updated_at
    FROM daily_counts
)

SELECT * FROM process_kpis