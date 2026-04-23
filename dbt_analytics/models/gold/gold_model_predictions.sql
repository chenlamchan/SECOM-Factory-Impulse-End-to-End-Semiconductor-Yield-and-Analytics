{{
    config(
        materialized='table',
        file_format='parquet',
        properties={"partitioning": "ARRAY['prediction_date']"}
    )
}}

/*
  gold_model_predictions
  ───────────────────────
  Materialises the output of batch_inference.py (Iceberg table ml.predictions)
  into the gold reporting layer so Streamlit pages can query it via Trino
  alongside the other gold tables without knowing about the ml namespace.
*/

WITH predictions AS (
    SELECT *
    FROM secom_catalog.ml.predictions       -- written by batch_inference.py
),

actuals AS (
    SELECT
        observation_id,
        wafer_status,
        CASE WHEN label_numeric = 1 THEN 1 ELSE 0 END AS actual_binary,
        process_timestamp,
        line_id,
        tester_id,
        shift,
        lot_id
    FROM {{ ref('silver_secom_reporting') }}
)

SELECT
    p.observation_id,
    p.prediction_date,
    p.prediction_timestamp,

    -- ── Actual ground truth (joined from silver) ─────────────────────────
    a.wafer_status                          AS actual_status,
    a.actual_binary,
    a.line_id,
    a.tester_id,
    a.shift,
    a.lot_id,

    -- ── Model output ──────────────────────────────────────────────────────
    ROUND(p.defect_probability, 4)          AS defect_probability,
    ROUND(p.yield_probability, 4)           AS yield_probability,
    p.prediction,                           -- 0 = Pass, 1 = Fail
    CASE p.prediction
        WHEN 1 THEN 'Fail'
        ELSE 'Pass'
    END                                     AS predicted_status,

    -- ── Outcome flags ─────────────────────────────────────────────────────
    CASE WHEN p.prediction = a.actual_binary THEN 1 ELSE 0 END AS is_correct,

    CASE
        WHEN p.prediction = 1 AND a.actual_binary = 1 THEN 'True Positive'
        WHEN p.prediction = 0 AND a.actual_binary = 0 THEN 'True Negative'
        WHEN p.prediction = 1 AND a.actual_binary = 0 THEN 'False Positive'
        WHEN p.prediction = 0 AND a.actual_binary = 1 THEN 'False Negative'
        ELSE 'Unknown'
    END                                     AS confusion_category,

    -- ── Confidence band ───────────────────────────────────────────────────
    CASE
        WHEN p.defect_probability >= 0.8 OR p.defect_probability <= 0.2 THEN 'High'
        WHEN p.defect_probability >= 0.65 OR p.defect_probability <= 0.35 THEN 'Medium'
        ELSE 'Low'
    END                                     AS confidence_band,

    -- ── Model provenance ──────────────────────────────────────────────────
    p.model_name,
    p.model_version,
    p.model_alias,

    CURRENT_TIMESTAMP                       AS last_updated_at

FROM predictions p
LEFT JOIN actuals a ON p.observation_id = a.observation_id
WHERE p.prediction_timestamp IS NOT NULL