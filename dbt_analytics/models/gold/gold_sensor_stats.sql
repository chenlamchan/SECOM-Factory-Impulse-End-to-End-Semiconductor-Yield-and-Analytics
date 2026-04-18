{{ config(materialized='table') }}

{% set phase1_size = var('spc_phase1_sample_size', 100) %}

WITH silver_data AS (
    SELECT * FROM {{ ref('silver_secom_reporting') }}
),

-- 1. Standardize the Top 5 Sensors and carry the timestamp forward
unpivoted AS (
    SELECT
        line_id, tester_id, '59' AS sensor_id, "59" AS val, process_timestamp FROM silver_data
    UNION ALL 
    SELECT line_id, tester_id, '103', "103", process_timestamp FROM silver_data
    UNION ALL 
    SELECT line_id, tester_id, '511', "511", process_timestamp FROM silver_data
    UNION ALL 
    SELECT line_id, tester_id, '424', "424", process_timestamp FROM silver_data
    UNION ALL 
    SELECT line_id, tester_id, '158', "158", process_timestamp FROM silver_data
),

-- Step 2: Identify the Phase I window
baseline_data AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY line_id, tester_id, sensor_id 
               ORDER BY process_timestamp
           ) as rn
    FROM unpivoted
    WHERE val IS NOT NULL
),

-- Step 3: Calculate raw stats for outlier detection (Pass 1)
raw_stats AS (
    SELECT 
        line_id, tester_id, sensor_id,
        AVG(val) as raw_mu,
        STDDEV(val) as raw_sigma
    FROM baseline_data
    WHERE rn <= {{ phase1_size }}
    GROUP BY 1, 2, 3
),

-- Step 4: Final calculation excluding Pass 1 anomalies (Pass 2)
final_calculation AS (
    SELECT 
        b.line_id,
        b.tester_id,
        b.sensor_id,
        AVG(b.val) as mu,
        STDDEV(b.val) as sigma,
        COUNT(b.val) as sample_size
    FROM baseline_data b
    INNER JOIN raw_stats r 
      ON b.line_id = r.line_id 
     AND b.tester_id = r.tester_id 
     AND b.sensor_id = r.sensor_id
    WHERE b.rn <= {{ phase1_size }}
      AND ABS(b.val - r.raw_mu) <= (3 * COALESCE(r.raw_sigma, 0.0001))
    GROUP BY 1, 2, 3
)

SELECT 
    line_id,
    tester_id,
    sensor_id,
    mu,
    sigma,
    (mu + (3 * sigma)) as ucl,
    (mu - (3 * sigma)) as lcl,
    (mu + (2 * sigma)) as uwl, -- Upper Warning Limit
    (mu - (2 * sigma)) as lwl, -- Lower Warning Limit

    (mu + (3.99 * sigma)) as usl, --Derived Specification Limits targeting Cpk = 1.33
    (mu - (3.99 * sigma)) as lsl, --Derived Specification Limits targeting Cpk = 1.33

    CURRENT_TIMESTAMP as frozen_at,
    'v1.0_phase1_frozen' as logic_version
FROM final_calculation
WHERE sample_size > 30 -- Ensure statistical significance