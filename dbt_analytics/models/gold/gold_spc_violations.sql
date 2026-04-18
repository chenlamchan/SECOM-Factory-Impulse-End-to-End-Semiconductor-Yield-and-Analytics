{{
    config(
        materialized='table',
        file_format='parquet',
        properties={"partitioning": "ARRAY['sensor_id']"}
    )
}}

/*
  gold_spc_violations
  ────────────────────
  Detects Nelson rule violations for the 5 tracked sensors.
  Each row = one observation that violated at least one rule.

  Rules implemented:
    Rule 1 — One point beyond ±3σ                         (critical)
    Rule 2 — 9 consecutive points on same side of mean    (run rule)
    Rule 3 — 6 consecutive points strictly trending       (trend rule)
    Rule 5 — 2 of 3 consecutive points beyond ±2σ         (warning zone)
*/

WITH sensor_ref AS (
    SELECT * FROM {{ ref('gold_sensor_stats') }}
),

silver AS (
    SELECT
        observation_id,
        process_timestamp,
        line_id,
        tester_id,
        shift,
        lot_id,
        wafer_status,
        "59"  AS val_59,
        "103" AS val_103,
        "511" AS val_511,
        "424" AS val_424,
        "158" AS val_158
    FROM {{ ref('silver_secom_reporting') }}
),

-- Unpivot to one row per (observation, sensor)
unpivoted AS (
    SELECT observation_id, process_timestamp, line_id, tester_id, shift, lot_id,
           wafer_status, '59'  AS sensor_id, val_59  AS val FROM silver
    UNION ALL
    SELECT observation_id, process_timestamp, line_id, tester_id, shift, lot_id,
           wafer_status, '103' AS sensor_id, val_103 AS val FROM silver
    UNION ALL
    SELECT observation_id, process_timestamp, line_id, tester_id, shift, lot_id,
           wafer_status, '511' AS sensor_id, val_511 AS val FROM silver
    UNION ALL
    SELECT observation_id, process_timestamp, line_id, tester_id, shift, lot_id,
           wafer_status, '424' AS sensor_id, val_424 AS val FROM silver
    UNION ALL
    SELECT observation_id, process_timestamp, line_id, tester_id, shift, lot_id,
           wafer_status, '158' AS sensor_id, val_158 AS val FROM silver
),

-- Join reference limits
enriched AS (
    SELECT
        u.*,
        r.mu,
        r.sigma,
        r.ucl,
        r.lcl,
        r.uwl,
        r.lwl,
        -- Z-score for this observation
        (u.val - r.mu) / NULLIF(r.sigma, 0) AS z_score,
        -- Side of mean (+1 above, -1 below, 0 on mean)
        SIGN(u.val - r.mu) AS side_of_mean,
        -- Row number within (line, sensor) ordered by time — for run detection
        ROW_NUMBER() OVER (
            PARTITION BY u.sensor_id, u.line_id, u.tester_id
            ORDER BY u.process_timestamp
        ) AS rn
    FROM unpivoted u
    LEFT JOIN sensor_ref r ON (u.sensor_id = r.sensor_id AND u.line_id = r.line_id AND u.tester_id = r.tester_id)
    WHERE u.val IS NOT NULL
),

-- Rule 1: beyond ±3σ
rule1 AS (
    SELECT *, 'Rule 1: Beyond 3σ' AS rule_name, 1 AS rule_number
    FROM enriched
    WHERE ABS(z_score) > 3
),

-- Rule 2: 9 consecutive same side — flag the 9th point
-- Using window SUM of 9 rows; if all positive or all negative → violation
rule2_base AS (
    SELECT *,
        SUM(CASE WHEN side_of_mean > 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY sensor_id, line_id, tester_id ORDER BY rn ROWS BETWEEN 8 PRECEDING AND CURRENT ROW)
            AS pos_run,
        SUM(CASE WHEN side_of_mean < 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY sensor_id, line_id, tester_id ORDER BY rn ROWS BETWEEN 8 PRECEDING AND CURRENT ROW)
            AS neg_run
    FROM enriched
),
rule2 AS (
    SELECT *, 'Rule 2: 9 on one side' AS rule_name, 2 AS rule_number
    FROM rule2_base
    WHERE pos_run = 9 OR neg_run = 9
),

-- Rule 3: 6 consecutive strictly trending
-- lag(5) and lag(4)...current must all increase or all decrease
rule3_base AS (
    SELECT *,
        LAG(val, 1) OVER (PARTITION BY sensor_id, line_id, tester_id ORDER BY rn) AS v1,
        LAG(val, 2) OVER (PARTITION BY sensor_id, line_id, tester_id ORDER BY rn) AS v2,
        LAG(val, 3) OVER (PARTITION BY sensor_id, line_id, tester_id ORDER BY rn) AS v3,
        LAG(val, 4) OVER (PARTITION BY sensor_id, line_id, tester_id ORDER BY rn) AS v4,
        LAG(val, 5) OVER (PARTITION BY sensor_id, line_id, tester_id ORDER BY rn) AS v5
    FROM enriched
),
rule3 AS (
    SELECT *, 'Rule 3: 6 trending' AS rule_name, 3 AS rule_number
    FROM rule3_base
    WHERE
        v5 IS NOT NULL AND (
            -- Strictly increasing
            (v5 < v4 AND v4 < v3 AND v3 < v2 AND v2 < v1 AND v1 < val)
            OR
            -- Strictly decreasing
            (v5 > v4 AND v4 > v3 AND v3 > v2 AND v2 > v1 AND v1 > val)
        )
),

-- Rule 4: 14 consecutive points alternating up and down
rule4_step1 AS (
    SELECT *,
        -- 1 for up, -1 for down, 0 for flat
        SIGN(val - LAG(val, 1) OVER (PARTITION BY sensor_id, line_id, tester_id ORDER BY rn)) AS step_sign
    FROM enriched
),
rule4_step2 AS (
    SELECT *,
        -- Check if the current direction is the exact opposite of the previous direction
        CASE 
            WHEN step_sign * LAG(step_sign, 1) OVER (PARTITION BY sensor_id, line_id, tester_id ORDER BY rn) < 0 
            THEN 1 
            ELSE 0 
        END AS is_flip
    FROM rule4_step1
),
rule4_base AS (
    SELECT *,
        -- Sum the flips over a rolling window of 12 rows. 
        -- 12 consecutive flips = 14 alternating points.
        SUM(is_flip) OVER (
            PARTITION BY sensor_id, line_id, tester_id 
            ORDER BY rn 
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
        ) AS flip_count
    FROM rule4_step2
),
rule4 AS (
    SELECT *, 'Rule 4: 14 alternating' AS rule_name, 4 AS rule_number
    FROM rule4_base
    WHERE flip_count = 12
),

-- Rule 5: 2 of 3 consecutive points beyond ±2σ on same side
rule5_base AS (
    SELECT *,
        -- Count of points > +2σ in this and the two preceding rows
        SUM(CASE WHEN z_score > 2 THEN 1 ELSE 0 END)
            OVER (PARTITION BY sensor_id, line_id, tester_id ORDER BY rn ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
            AS above_2sigma_run,
        SUM(CASE WHEN z_score < -2 THEN 1 ELSE 0 END)
            OVER (PARTITION BY sensor_id, line_id, tester_id ORDER BY rn ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
            AS below_2sigma_run
    FROM enriched
),
rule5 AS (
    SELECT *, 'Rule 5: 2/3 beyond 2σ' AS rule_name, 5 AS rule_number
    FROM rule5_base
    WHERE above_2sigma_run >= 2 OR below_2sigma_run >= 2
),

-- Union all rules and deduplicate to one row per observation per sensor (keep worst rule)
all_violations AS (
    SELECT observation_id, process_timestamp, line_id, tester_id, shift, lot_id,
           wafer_status, sensor_id, val, mu, sigma, z_score, rule_name, rule_number
    FROM rule1
    UNION ALL
    SELECT observation_id, process_timestamp, line_id, tester_id, shift, lot_id,
           wafer_status, sensor_id, val, mu, sigma, z_score, rule_name, rule_number
    FROM rule2
    UNION ALL
    SELECT observation_id, process_timestamp, line_id, tester_id, shift, lot_id,
           wafer_status, sensor_id, val, mu, sigma, z_score, rule_name, rule_number
    FROM rule3
    UNION ALL
    SELECT observation_id, process_timestamp, line_id, tester_id, shift, lot_id,
           wafer_status, sensor_id, val, mu, sigma, z_score, rule_name, rule_number
    FROM rule4
    UNION ALL
    SELECT observation_id, process_timestamp, line_id, tester_id, shift, lot_id,
           wafer_status, sensor_id, val, mu, sigma, z_score, rule_name, rule_number
    FROM rule5
)

SELECT
    observation_id,
    process_timestamp,
    line_id,
    tester_id,
    shift,
    lot_id,
    wafer_status,
    sensor_id,
    ROUND(val, 6)     AS sensor_value,
    ROUND(mu, 6)      AS reference_mean,
    ROUND(sigma, 6)   AS reference_sigma,
    ROUND(z_score, 4) AS z_score,
    rule_name,
    rule_number,
    CURRENT_TIMESTAMP AS computed_at
FROM all_violations