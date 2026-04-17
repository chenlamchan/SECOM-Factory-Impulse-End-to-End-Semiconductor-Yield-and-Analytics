{{ config(
    materialized='table',
    file_format='parquet',
    properties={"partitioning": "ARRAY['process_date']"}
)}}

/*
  gold_shift_metrics
  ────────────────────
  Aggregates yield, DPPM, throughput, and scrap by shift × line × date.
  Enables OEE / shift comparison views in the dashboard.
*/

WITH silver AS (
    SELECT 
        CAST(process_timestamp AS DATE) AS process_date,
        line_id,
        tester_id,
        shift,
        wafer_status,
        missing_sensor_count,
        observation_id,
        lot_id
    FROM {{ ref('silver_secom_reporting') }}
),

quarantine AS (
    SELECT
        CAST(process_timestamp AS DATE) AS process_date,
        line_id,
        tester_id,
        shift,
        COUNT(*) AS quarantined_wafers
    FROM {{ ref('silver_secom_quarantine') }}
    GROUP BY 1, 2, 3, 4
),

shift_agg AS (
    SELECT
        process_date,
        line_id,
        tester_id,
        shift,
        COUNT(observation_id) AS wafers_tested,
        SUM(CASE WHEN wafer_status='Pass' THEN 1 ELSE 0 END) AS passed_wafers,
        SUM(CASE WHEN wafer_status='Fail' THEN 1 ELSE 0 END) AS failed_wafers,
        COUNT(DISTINCT lot_id) AS lots_processed,
        AVG(missing_sensor_count) AS avg_missing_sensors
    FROM silver
    GROUP BY process_date, line_id, tester_id, shift
)

SELECT
    s.process_date,
    s.line_id,
    s.tester_id,
    s.shift,
    s.wafers_tested,
    s.passed_wafers,
    s.failed_wafers,
    COALESCE(q.quarantined_wafers, 0) AS quarantined_wafers,
    s.lots_processed,
 
    -- Yield
    ROUND(
        s.passed_wafers / CAST(NULLIF(s.wafers_tested, 0) AS DOUBLE) * 100, 2
    ) AS yield_pct,
 
    -- DPPM (defective parts per million, based on tested)
    ROUND(
        s.failed_wafers / CAST(NULLIF(s.wafers_tested, 0) AS DOUBLE) * 1e6, 0
    ) AS ppm_defective,
 
    -- Scrap rate (quarantined / (tested + quarantined))
    ROUND(
        COALESCE(q.quarantined_wafers, 0)
        / CAST(NULLIF(s.wafers_tested + COALESCE(q.quarantined_wafers, 0), 0) AS DOUBLE) * 100,
        2
    ) AS scrap_rate_pct,
 
    ROUND(s.avg_missing_sensors, 2) AS avg_missing_sensors,
 
    -- Shift order for sorting in UI
    CASE s.shift 
    WHEN 'Day' THEN 1 
    WHEN 'Swing' THEN 2 
    WHEN 'Night' THEN 3 
    ELSE 4 END AS shift_order,
 
    CURRENT_TIMESTAMP AS last_updated_at

FROM shift_agg s
LEFT JOIN quarantine q
    ON  s.process_date = q.process_date
    AND s.line_id = q.line_id
    AND s.tester_id = q.tester_id
    AND s.shift = q.shift