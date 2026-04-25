{{
    config(
        materialized='table',
        file_format='parquet',
        properties={"partitioning": "ARRAY['process_date']"}
    )
}}

{% set scheduled_shift = var('oee_scheduled_shifts_per_day', 1) %}
{% set theoretical_max_wafer = var('oee_theoretical_max_wafers_per_day', 10) %}

/*
  gold_oee_metrics
  ──────────────────
  Computes OEE (Overall Equipment Effectiveness) at the process_date × line_id
  grain from shift_metrics.

  OEE = Availability × Performance × Quality

  Definitions:
    Availability  = actual running shifts / scheduled shifts per day ( assume 3 shifts = full availability baseline)
    Performance   = actual throughput / theoretical max throughput (theoretical max = highest wafer count seen across all lines/days)
    Quality       = First Pass Yield (passed / tested)

  Working from aggregated shift data (not raw equipment logs),
  Availability is approximated from shifts that produced data vs 3 expected.

*/

WITH shift_data AS (
    SELECT * FROM {{ ref('gold_shift_metrics') }}
),

-- Daily line totals
daily_line AS (
    SELECT
        process_date,
        line_id,
        tester_id,
        COUNT(DISTINCT shift) AS shifts_with_data,
        SUM(wafers_tested) AS total_wafers_tested,
        SUM(passed_wafers) AS total_passed,
        SUM(failed_wafers) AS total_failed,
        SUM(quarantined_wafers) AS total_quarantined,
        SUM(lots_processed) AS total_lots
    FROM shift_data
    GROUP BY process_date, line_id, tester_id
),

oee AS (
    SELECT
        process_date,
        line_id,
        tester_id,
        shifts_with_data,
        total_wafers_tested,
        total_passed,
        total_failed,
        total_quarantined,
        total_lots,

        -- Availability: shifts that ran / 3 expected shifts
        ROUND(
            LEAST(shifts_with_data / CAST({{scheduled_shift}} AS DOUBLE), 1.0) * 100, 2
        ) AS availability_pct,

        -- Performance: avg(actual / theoretical) across shifts
        ROUND(
            LEAST(total_wafers_tested / CAST({{theoretical_max_wafer}} AS DOUBLE), 1.0) * 100, 2
        ) AS performance_pct,

        -- Quality: FPY
        ROUND(
            total_passed / CAST(NULLIF(total_wafers_tested, 0) AS DOUBLE) * 100, 2
        ) AS quality_pct

    FROM daily_line
)

SELECT
    process_date,
    line_id,
    tester_id,
    shifts_with_data,
    total_wafers_tested,
    total_passed,
    total_failed,
    total_quarantined,
    total_lots,
    availability_pct,
    performance_pct,
    quality_pct,

    -- OEE = A × P × Q  (expressed 0–100)
    ROUND(
        (availability_pct / 100.0)
        * (performance_pct / 100.0)
        * (quality_pct    / 100.0)
        * 100.0,
        2
    ) AS oee_pct,

    -- World-class OEE benchmark = 85%
    CASE
        WHEN (availability_pct/100.0)*(performance_pct/100.0)*(quality_pct/100.0)*100 >= 85
            THEN 'World Class'
        WHEN (availability_pct/100.0)*(performance_pct/100.0)*(quality_pct/100.0)*100 >= 65
            THEN 'Good'
        WHEN (availability_pct/100.0)*(performance_pct/100.0)*(quality_pct/100.0)*100 >= 50
            THEN 'Average'
        ELSE 'Poor'
    END AS oee_classification,

    CURRENT_TIMESTAMP AS last_updated_at
FROM oee