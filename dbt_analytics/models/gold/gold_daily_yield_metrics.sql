{{ 
    config(
        materialized= 'table',
        file_format= 'parquet',
        properties={"partitioning": "ARRAY['process_date']"}
    )
}}

WITH silver_data AS (
    SELECT * FROM {{ ref('silver_secom_reporting' )}}
),

daily_counts AS (
    SELECT
        CAST(process_timestamp AS DATE) AS process_date,
        line_id,
        COUNT(observation_id) AS total_wafers_tested,
        SUM(CASE WHEN wafer_status='Pass' THEN 1 ELSE 0 END) AS passed_wafers,
        SUM(CASE WHEN wafer_status='Fail' THEN 1 ELSE 0 END) AS failed_wafers
    FROM silver_data
    GROUP BY 
        CAST(process_timestamp AS DATE),
        line_id
),

process_kpis AS (
    SELECT 
        process_date,
        line_id,
        total_wafers_tested,
        passed_wafers,
        failed_wafers,

        -- Yield %: (Pass / Total) * 100 (Protected against divide-by-zero)
        ROUND(
            (passed_wafers / CAST(NULLIF(total_wafers_tested, 0) AS DOUBLE)) * 100, 
            2
        ) AS yield_percentage,

        -- PPM Defective: (Fail / Total) * 1,000,000
        ROUND(
            (failed_wafers / CAST(NULLIF(total_wafers_tested, 0) AS DOUBLE)) * 1000000, 
            0
        ) AS ppm_defective,

        CURRENT_TIMESTAMP AS last_updated_at
    FROM daily_counts
)

SELECT * FROM process_kpis