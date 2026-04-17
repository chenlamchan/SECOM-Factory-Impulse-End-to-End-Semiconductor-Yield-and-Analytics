{{
    config(
        materialized='table',
        file_format='parquet',
        properties={"partitioning": "ARRAY['process_date']"}
    )
}}

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

-- Max throughput across all lines per shift as performance baseline
throughput_reference AS (
    SELECT
        shift,
        MAX(wafers_tested) AS max_shift_throughput
    FROM shift_data
    GROUP BY shift
),

-- Daily line totals
daily_line AS (
    SELECT
        s.process_date,
        s.line_id,
        s.tester_id,
        COUNT(DISTINCT s.shift) AS shifts_with_data,
        SUM(s.wafers_tested) AS total_wafers_tested,
        SUM(s.passed_wafers) AS total_passed,
        SUM(s.failed_wafers) AS total_failed,
        SUM(s.quarantined_wafers) AS total_quarantined,
        SUM(s.lots_processed) AS total_lots,
        SUM(
            s.wafers_tested / CAST(NULLIF(t.max_shift_throughput, 0) AS DOUBLE)
        ) AS performance_sum
    FROM shift_data s
    LEFT JOIN throughput_reference t ON s.shift = t.shift
    GROUP BY s.process_date, s.line_id, s.tester_id
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
            LEAST(shifts_with_data / 3.0, 1.0) * 100, 2
        ) AS availability_pct,

        -- Performance: avg(actual / theoretical) across shifts
        ROUND(
            LEAST(performance_sum / NULLIF(shifts_with_data, 0), 1.0) * 100, 2
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