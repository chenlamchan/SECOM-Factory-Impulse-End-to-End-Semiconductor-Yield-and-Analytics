{{ config(materialized='table') }}

WITH silver_data AS (
    SELECT * FROM {{ ref('silver_secom_reporting') }}
),

-- Standardize the Top 5 Sensors for POC
stats_unpivoted AS (
    SELECT 
        '59' AS sensor_id,
        "59" AS val 
    FROM silver_data
    UNION ALL 
    
    SELECT '103', "103" FROM silver_data
    UNION ALL 

    SELECT '511', "511" FROM silver_data
    UNION ALL 

    SELECT '424', "424" FROM silver_data
    UNION ALL 

    SELECT '158', "158" FROM silver_data
),

calculation AS (
    SELECT 
        sensor_id,
        AVG(val) as mu,
        STDDEV(val) as sigma,
        COUNT(val) as sample_size
    FROM stats_unpivoted
    GROUP BY sensor_id
)

SELECT 
    sensor_id,
    mu,
    sigma,
    (mu + (3 * sigma)) as ucl,
    (mu - (3 * sigma)) as lcl,
    (mu + (2 * sigma)) as uwl, -- Upper Warning Limit
    (mu - (2 * sigma)) as lwl, -- Lower Warning Limit
    CURRENT_TIMESTAMP as frozen_at,
    'v1.0' as logic_version
FROM calculation
WHERE sample_size > 30 -- Ensure statistical significance