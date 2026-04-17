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

quarantine_data AS (
    SELECT * FROM {{ ref('silver_secom_quarantine' )}}
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

daily_quarantine AS (
    SELECT
        CAST(process_timestamp AS DATE) AS process_date,
        line_id,
        COUNT(observation_id) AS quarantined_wafers
    FROM quarantine_data
    GROUP BY 
        CAST(process_timestamp AS DATE),
        line_id
),

process_kpis AS (
    SELECT 
        c.process_date,
        c.line_id,
        c.total_wafers_tested,
        COALESCE(q.quarantined_wafers, 0) AS quarantined_wafers,
        c.passed_wafers,
        c.failed_wafers,

        -- Yield %: (Pass / Total) * 100
        ROUND(
            (c.passed_wafers / CAST(NULLIF(c.total_wafers_tested, 0) AS DOUBLE)) * 100, 
            2
        ) AS yield_percentage,

        -- PPM Defective: (Fail / Total) * 1,000,000
        ROUND(
            (c.failed_wafers / CAST(NULLIF(c.total_wafers_tested, 0) AS DOUBLE)) * 1000000, 
            0
        ) AS ppm_defective,

        CURRENT_TIMESTAMP AS last_updated_at
    FROM daily_counts c
    LEFT JOIN daily_quarantine q
        ON c.process_date = q.process_date AND c.line_id = q.line_id
)

SELECT * FROM process_kpis