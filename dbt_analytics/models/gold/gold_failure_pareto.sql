{{
    config(
        materialized='table',
        file_format='parquet'
    )
}}

/*
  gold_failure_pareto
  ─────────────────────
  Computes point-biserial correlation between each of the top 20 most
  discriminative sensors and the binary Pass/Fail label.

  Point-biserial r = (μ_pass - μ_fail) / σ_pooled
  which in Trino = CORR(val, label_numeric)

  Output: one row per sensor, sorted by |correlation| DESC.
  Used by the Failure Analysis page Pareto chart.
*/

WITH silver AS (
    SELECT
        label_numeric,   -- -1 = Pass, 1 = Fail
        -- Include all candidate sensors (top 30 by data availability)
        "59",  "103", "511", "424", "158",
        "4",   "5",   "6",   "7",   "8",
        "9",   "10",  "11",  "12",  "13",
        "57",  "58",  "60",  "61",  "62",
        "100", "101", "102", "104", "105",
        "200", "201", "202", "300", "400"
    FROM {{ ref('silver_secom_reporting') }}
),

-- Compute correlation per sensor using a static list
-- (Trino does not support dynamic unpivot natively)
correlations AS (
    SELECT '59'  AS sensor_id, CORR("59",  label_numeric) AS r, COUNT("59")  AS n FROM silver
    UNION ALL
    SELECT '103', CORR("103", label_numeric), COUNT("103") FROM silver
    UNION ALL
    SELECT '511', CORR("511", label_numeric), COUNT("511") FROM silver
    UNION ALL
    SELECT '424', CORR("424", label_numeric), COUNT("424") FROM silver
    UNION ALL
    SELECT '158', CORR("158", label_numeric), COUNT("158") FROM silver
    UNION ALL
    SELECT '4',   CORR("4",   label_numeric), COUNT("4")   FROM silver
    UNION ALL
    SELECT '5',   CORR("5",   label_numeric), COUNT("5")   FROM silver
    UNION ALL
    SELECT '6',   CORR("6",   label_numeric), COUNT("6")   FROM silver
    UNION ALL
    SELECT '7',   CORR("7",   label_numeric), COUNT("7")   FROM silver
    UNION ALL
    SELECT '8',   CORR("8",   label_numeric), COUNT("8")   FROM silver
    UNION ALL
    SELECT '9',   CORR("9",   label_numeric), COUNT("9")   FROM silver
    UNION ALL
    SELECT '10',  CORR("10",  label_numeric), COUNT("10")  FROM silver
    UNION ALL
    SELECT '11',  CORR("11",  label_numeric), COUNT("11")  FROM silver
    UNION ALL
    SELECT '12',  CORR("12",  label_numeric), COUNT("12")  FROM silver
    UNION ALL
    SELECT '13',  CORR("13",  label_numeric), COUNT("13")  FROM silver
    UNION ALL
    SELECT '57',  CORR("57",  label_numeric), COUNT("57")  FROM silver
    UNION ALL
    SELECT '58',  CORR("58",  label_numeric), COUNT("58")  FROM silver
    UNION ALL
    SELECT '60',  CORR("60",  label_numeric), COUNT("60")  FROM silver
    UNION ALL
    SELECT '61',  CORR("61",  label_numeric), COUNT("61")  FROM silver
    UNION ALL
    SELECT '62',  CORR("62",  label_numeric), COUNT("62")  FROM silver
    UNION ALL
    SELECT '100', CORR("100", label_numeric), COUNT("100") FROM silver
    UNION ALL
    SELECT '101', CORR("101", label_numeric), COUNT("101") FROM silver
    UNION ALL
    SELECT '102', CORR("102", label_numeric), COUNT("102") FROM silver
    UNION ALL
    SELECT '104', CORR("104", label_numeric), COUNT("104") FROM silver
    UNION ALL
    SELECT '105', CORR("105", label_numeric), COUNT("105") FROM silver
    UNION ALL
    SELECT '200', CORR("200", label_numeric), COUNT("200") FROM silver
    UNION ALL
    SELECT '201', CORR("201", label_numeric), COUNT("201") FROM silver
    UNION ALL
    SELECT '202', CORR("202", label_numeric), COUNT("202") FROM silver
    UNION ALL
    SELECT '300', CORR("300", label_numeric), COUNT("300") FROM silver
    UNION ALL
    SELECT '400', CORR("400", label_numeric), COUNT("400") FROM silver
)

SELECT
    ROW_NUMBER() OVER (ORDER BY ABS(r) DESC NULLS LAST) AS rank,
    sensor_id,
    ROUND(r, 6) AS correlation,
    ROUND(ABS(r), 6) AS abs_correlation,
    n AS sample_size,
    CASE
        WHEN ABS(r) >= 0.3 THEN 'Strong'
        WHEN ABS(r) >= 0.1 THEN 'Moderate'
        ELSE 'Weak'
    END AS effect_strength,
    CASE
        WHEN r > 0 THEN 'Higher reading → more failures'
        WHEN r < 0 THEN 'Higher reading → fewer failures'
        ELSE 'No association'
    END AS direction,
    CURRENT_TIMESTAMP AS computed_at
FROM correlations
WHERE r IS NOT NULL AND n >= 30   -- statistical significance guard
ORDER BY abs_correlation DESC