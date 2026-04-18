{{ config(materialized='table') }}

WITH silver_data AS (
    SELECT * FROM {{ ref('silver_secom_reporting') }}
),

-- 1. Standardize the Top 5 Sensors and carry the timestamp forward
stats_unpivoted AS (
    SELECT
        line_id, tester_id, '59' AS sensor_id, "59" AS val 
    FROM silver_data
    UNION ALL 
    SELECT line_id, tester_id, '103', "103" FROM silver_data
    UNION ALL 
    SELECT line_id, tester_id, '511', "511" FROM silver_data
    UNION ALL 
    SELECT line_id, tester_id, '424', "424" FROM silver_data
    UNION ALL 
    SELECT line_id, tester_id, '158', "158" FROM silver_data
),

-- 2. Define the Phase I Window (First 100 valid observations per machine/sensor)
phase_1_window AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY line_id, tester_id, sensor_id 
               ORDER BY process_timestamp
           ) as rn
    FROM unpivoted
    WHERE val IS NOT NULL
),

baseline_data AS (
    SELECT * FROM phase_1_window 
    WHERE rn <= {{ var('spc_phase1_sample_size', 100) }}  -- The industry standard sample size
),

-- 3. Calculate raw stats (Pass 1)
raw_stats AS (
    SELECT 
        line_id, tester_id, sensor_id,
        AVG(val) as raw_mu,
        STDDEV(val) as raw_sigma
    FROM baseline_data
    GROUP BY line_id, tester_id, sensor_id
),

-- 4. Filter out Phase I anomalies (Pass 2 - removing points beyond 3 raw sigma)
clean_baseline_data AS (
    SELECT b.*
    FROM baseline_data b
    JOIN raw_stats r 
      ON b.line_id = r.line_id 
     AND b.tester_id = r.tester_id 
     AND b.sensor_id = r.sensor_id
    -- Keep only points within 3 sigma of the raw mean to prevent inflated variance
    WHERE ABS(b.val - r.raw_mu) <= (3 * COALESCE(r.raw_sigma, 0.0001))
),

-- 5. Calculate the final, frozen baseline stats
final_calculation AS (
    SELECT 
        line_id,
        tester_id,
        sensor_id,
        AVG(val) as mu,
        STDDEV(val) as sigma,
        COUNT(val) as sample_size
    FROM clean_baseline_data
    GROUP BY line_id, tester_id, sensor_id
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
    CURRENT_TIMESTAMP as frozen_at,
    'v1.0_phase1_frozen' as logic_version
FROM final_calculation
WHERE sample_size > 30 -- Ensure statistical significance