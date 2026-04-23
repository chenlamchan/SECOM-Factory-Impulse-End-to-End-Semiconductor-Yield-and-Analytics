"""
performance_estimator.py — Confidence-Based Performance Estimation (CBPE)
─────────────────────────────────────────────────────────────────────────
Estimates model performance WITHOUT waiting for ground-truth labels.
This matters in semiconductor manufacturing where true yield outcomes
can take days/weeks to confirm via downstream testing.

Strategy:
  1. Load the calibration reference dataset from the champion model's
     MLflow artifacts (cbpe_reference_dataset.parquet, written by
     calibrate_model.py). This gives us calibrated_prob ↔ true_label pairs.

  2. Load recent production predictions from Trino gold_model_predictions.

  3. Estimate performance using the CBPE approach:
       - Match each production prediction's score to the nearest calibration
         bucket and use the bucket's average accuracy as a proxy for ground truth.
       - Compute estimated AUC and F1 from these proxy labels.
       - Measure prediction interval width inflation as a distribution-shift proxy.

  4. Try NannyML CBPE first (full implementation); fall back to our own
     estimator if nannyml is not installed or the reference dataset is missing.

  5. Save results to MinIO as a Parquet file under:
       ml-predictions/performance_estimates/<YYYYMMDD_HHMMSS>/estimates.parquet

  6. Print JSON summary to stdout (captured as Airflow XCom).

Run pattern:
  Triggered as an Airflow task in secom_ml_daily_operations DAG,
  immediately after run_batch_inference so predictions are fresh.

"""

import os
import json
import logging
import argparse
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import s3fs
from config import ServiceConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config = ServiceConfig()

MINIO_ENDPOINT   = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key

# CHANGE: same env var pattern as drift_monitor.py / batch_inference.py
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]      = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]  = MINIO_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"]     = "us-east-1"

MODEL_NAME       = os.environ.get("MODEL_NAME",          "secom_yield_predictor")
MODEL_ALIAS      = os.environ.get("MODEL_ALIAS",         "champion")
LOOKBACK_DAYS    = int(os.environ.get("DRIFT_LOOKBACK_DAYS", "7"))
ML_PRED_BUCKET   = os.environ.get("ML_PREDICTIONS_BUCKET", "ml-predictions")


# ─── MinIO filesystem ──────────────────────────────────────────────────────────
def _get_s3():
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
    )


# ─── Champion manifest loader ──────────────────────────────────────────────────
# CHANGE: identical pattern to drift_monitor.py _get_champion_active_sensors()
def _load_champion_manifest() -> tuple[dict, str]:
    """
    Downloads the champion model's manifest from MLflow.
    Returns (manifest dict, run_id).
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()

    v   = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    manifest_uri = f"runs:/{v.run_id}/metadata/feature_manifest.json"
    logger.info("Downloading manifest from %s", manifest_uri)
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=manifest_uri)

    with open(local_path) as f:
        manifest = json.load(f)

    logger.info("Loaded manifest for model v%s (run_id=%s)", v.version, v.run_id)
    return manifest, v.run_id, v.version


# ─── Calibration reference dataset loader ─────────────────────────────────────
# CHANGE: loads cbpe_reference_dataset.parquet logged by calibrate_model.py
def _load_calibration_reference(run_id: str) -> pd.DataFrame | None:
    """
    Downloads the calibration reference dataset from MLflow.
    This was logged by calibrate_model.py as calibration/cbpe_reference_dataset.parquet.
    Contains: feature columns + binary_label + calibrated_prob + raw_prob.
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    try:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/calibration/cbpe_reference_dataset.parquet"
        )
        df = pd.read_parquet(local_path)
        logger.info("Loaded calibration reference: %d rows", len(df))
        return df
    except Exception as e:
        logger.warning("Could not load calibration reference dataset: %s", e)
        return None


# ─── Recent prediction loader ──────────────────────────────────────────────────
# CHANGE: Trino query pattern from drift_monitor.py _load_prediction_scores()
def _load_recent_predictions() -> pd.DataFrame:
    """Load recent prediction scores from Trino gold_model_predictions."""
    try:
        import trino
        host    = os.environ.get("TRINO_HOST",    "trino")
        port    = int(os.environ.get("TRINO_PORT", "8080"))
        catalog = os.environ.get("TRINO_CATALOG", "secom_catalog")

        conn = trino.dbapi.connect(host=host, port=port, user="admin",
                                   catalog=catalog, schema="gold")
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(prediction_timestamp) FROM gold_model_predictions")
        max_ts_row = cursor.fetchone()

        if not max_ts_row or not max_ts_row[0]:
            logger.warning("No predictions found in gold_model_predictions.")
            return pd.DataFrame()

        max_ts = max_ts_row[0]
        df = pd.read_sql_query(
            f"""
            SELECT defect_probability, model_version, model_alias, prediction_timestamp
            FROM gold_model_predictions
            WHERE prediction_timestamp >= TIMESTAMP '{max_ts}' - INTERVAL '{LOOKBACK_DAYS}' DAY
            """,
            conn,
        )
        conn.close()
        logger.info("Loaded %d recent predictions (simulated current: %s)", len(df), max_ts)
        return df
    except Exception as e:
        logger.warning("Could not load recent predictions from Trino: %s", e)
        return pd.DataFrame()


# ─── NannyML CBPE ─────────────────────────────────────────────────────────────
def _estimate_with_nannyml(ref_df: pd.DataFrame,
                            current_scores: pd.Series,
                            feature_cols: list) -> dict | None:
    """
    Use NannyML's CBPE estimator if available.
    Requires calibrated_prob and binary_label in ref_df.
    """
    try:
        import nannyml as nml

        # NannyML needs a 'timestamp' column and the score column
        ref_nml = ref_df.copy()
        ref_nml["timestamp"] = pd.date_range(end=pd.Timestamp.now(), periods=len(ref_nml), freq="h")
        ref_nml["y_pred_proba"] = ref_nml.get("calibrated_prob", ref_nml.get("raw_prob", 0.5))
        ref_nml["y_pred"] = (ref_nml["y_pred_proba"] >= 0.5).astype(int)

        cur_nml = pd.DataFrame({
            "timestamp":     pd.date_range(end=pd.Timestamp.now(), periods=len(current_scores), freq="h"),
            "y_pred_proba":  current_scores.values,
            "y_pred":        (current_scores >= 0.5).astype(int),
        })

        estimator = nml.CBPE(
            y_pred_proba="y_pred_proba",
            y_pred="y_pred",
            y_true="binary_label",
            metrics=["roc_auc", "f1"],
            chunk_size=min(300, len(cur_nml) // 2 or len(cur_nml)),
            problem_type="binary_classification",
        )
        estimator.fit(ref_nml)
        results = estimator.estimate(cur_nml)
        summary = results.to_df()

        estimated_auc = float(summary["estimated_roc_auc"].dropna().iloc[-1]) if "estimated_roc_auc" in summary.columns else None
        estimated_f1  = float(summary["estimated_f1"].dropna().iloc[-1]) if "estimated_f1" in summary.columns else None

        logger.info("NannyML CBPE — estimated AUC=%.4f F1=%.4f", estimated_auc or 0, estimated_f1 or 0)
        return {"estimated_auc": estimated_auc, "estimated_f1": estimated_f1, "method": "nannyml_cbpe"}
    except ImportError:
        logger.info("NannyML not installed — using custom CBPE estimator.")
        return None
    except Exception as e:
        logger.warning("NannyML CBPE failed: %s", e)
        return None


# ─── Custom CBPE estimator ─────────────────────────────────────────────────────
def _estimate_custom_cbpe(ref_df: pd.DataFrame, current_scores: pd.Series) -> dict:
    """
    Custom CBPE proxy when NannyML is unavailable.

    Method (Brier score decomposition / calibration bucket approach):
      1. Bin the reference calibration data into 10 probability buckets.
      2. For each bucket, record the empirical positive rate (= local accuracy).
      3. For each production prediction, look up its bucket's positive rate
         as the estimated probability of a true positive.
      4. Use these proxy labels to compute estimated AUC and F1.

    Prediction interval width inflation:
      Compare the IQR of current scores vs reference calibrated_prob.
      Widening → distribution shift → less confident predictions.
    """
    ref_probs  = ref_df.get("calibrated_prob", ref_df.get("raw_prob", None))
    ref_labels = ref_df.get("binary_label", None)

    if ref_probs is None or ref_labels is None:
        logger.warning("Reference dataset missing required columns — returning fallback estimate.")
        avg_score = float(current_scores.mean())
        return {
            "estimated_auc":  round(0.5 + abs(avg_score - 0.5), 4),
            "estimated_f1":   None,
            "interval_width": None,
            "width_inflation": None,
            "method":         "fallback_heuristic",
        }

    # Build calibration buckets from reference data
    n_bins    = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_pos_rate = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (ref_probs >= bin_edges[i]) & (ref_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bucket_pos_rate[i] = float(ref_labels[mask].mean())

    # Assign proxy labels to current predictions
    bin_idx    = np.digitize(current_scores.values, bin_edges) - 1
    bin_idx    = np.clip(bin_idx, 0, n_bins - 1)
    proxy_pos  = bucket_pos_rate[bin_idx]   # estimated P(true positive) per observation

    # Estimated AUC via proxy positives
    # Sort by predicted score; AUC = P(score_pos > score_neg) estimated via proxy
    scores_arr = current_scores.values
    order      = np.argsort(scores_arr)
    sorted_pos = proxy_pos[order]
    sorted_neg = 1.0 - sorted_pos
    cum_neg    = np.cumsum(sorted_neg)
    est_auc    = float(np.sum(sorted_pos * cum_neg) / max(np.sum(sorted_pos) * np.sum(sorted_neg), 1e-9))
    est_auc    = round(min(max(est_auc, 0.5), 1.0), 4)

    # Estimated F1 at threshold 0.5 using proxy labels
    y_pred     = (scores_arr >= 0.5).astype(int)
    est_tp     = float(np.sum(proxy_pos * y_pred))
    est_fp     = float(np.sum((1.0 - proxy_pos) * y_pred))
    est_fn     = float(np.sum(proxy_pos * (1 - y_pred)))
    est_prec   = est_tp / max(est_tp + est_fp, 1e-9)
    est_rec    = est_tp / max(est_tp + est_fn, 1e-9)
    est_f1     = round(2 * est_prec * est_rec / max(est_prec + est_rec, 1e-9), 4)

    # Interval width inflation (distribution shift proxy)
    ref_iqr     = float(np.percentile(ref_probs, 75) - np.percentile(ref_probs, 25))
    cur_iqr     = float(np.percentile(current_scores, 75) - np.percentile(current_scores, 25))
    interval_w  = round(cur_iqr, 4)
    width_infl  = round((cur_iqr - ref_iqr) / max(ref_iqr, 1e-9), 4) if ref_iqr > 0 else 0.0

    logger.info(
        "Custom CBPE — estimated AUC=%.4f F1=%.4f IQR_inflation=%.4f",
        est_auc, est_f1, width_infl
    )
    return {
        "estimated_auc":   est_auc,
        "estimated_f1":    est_f1,
        "interval_width":  interval_w,
        "width_inflation": width_infl,
        "method":          "custom_cbpe",
    }


# ─── Model age calculator ──────────────────────────────────────────────────────
def _model_age_hours(run_id: str) -> float:
    """Hours since the champion model's training run ended."""
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = MlflowClient()
        run    = client.get_run(run_id)
        end_ts = run.info.end_time  # milliseconds epoch
        if end_ts:
            age_ms = datetime.now(timezone.utc).timestamp() * 1000 - end_ts
            return round(age_ms / 3_600_000, 1)
    except Exception:
        pass
    return 0.0


# ─── Save results to MinIO ─────────────────────────────────────────────────────
def _save_to_minio(estimates: dict, tag: str) -> str:
    """Write estimates Parquet to MinIO under ml-predictions/performance_estimates/."""
    fs  = _get_s3()
    key = f"{ML_PRED_BUCKET}/performance_estimates/{tag}/estimates.parquet"

    row_df = pd.DataFrame([estimates])
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        row_df.to_parquet(tmp.name, index=False)
        fs.put(tmp.name, key)

    logger.info("Performance estimates saved to s3://%s", key)
    return f"s3://{key}"


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SECOM ML — Performance Estimation (CBPE)")
    parser.add_argument("--model-name",  default=MODEL_NAME)
    parser.add_argument("--model-alias", default=MODEL_ALIAS)
    args = parser.parse_args()

    now_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info("Starting CBPE performance estimation (%s)", now_tag)

    # ── 1. Load champion manifest (dynamic features) ──────────────────────────
    try:
        manifest, run_id, model_version = _load_champion_manifest()
    except Exception as e:
        logger.error("Could not load champion manifest: %s — aborting.", e)
        return

    active_sensors = manifest.get("active_features_list", [])
    feature_cols   = active_sensors  # numeric sensors for reference dataset alignment

    # ── 2. Load calibration reference dataset ─────────────────────────────────
    ref_df = _load_calibration_reference(run_id)

    # ── 3. Load recent production predictions ─────────────────────────────────
    pred_df = _load_recent_predictions()
    if pred_df.empty or len(pred_df) < 10:
        logger.warning("Fewer than 10 predictions available — skipping estimation.")
        print(json.dumps({"estimates_available": False,
                          "reason": "Insufficient prediction volume"}))
        return

    current_scores = pred_df["defect_probability"].dropna()
    model_age      = _model_age_hours(run_id)

    # ── 4. Estimate performance ────────────────────────────────────────────────
    estimates = None

    # Try NannyML first if we have reference data
    if ref_df is not None:
        estimates = _estimate_with_nannyml(ref_df, current_scores, feature_cols)

    # Custom CBPE fallback
    if estimates is None:
        if ref_df is not None:
            estimates = _estimate_custom_cbpe(ref_df, current_scores)
        else:
            # No reference data at all — heuristic only
            avg  = float(current_scores.mean())
            std  = float(current_scores.std())
            estimates = {
                "estimated_auc":   round(0.5 + abs(avg - 0.5) * 2, 4),
                "estimated_f1":    None,
                "interval_width":  round(std * 2, 4),
                "width_inflation": None,
                "method":          "heuristic_no_reference",
            }
            logger.warning("No calibration reference dataset — using heuristic estimate.")

    # ── 5. Assemble final payload ──────────────────────────────────────────────
    estimates.update({
        "estimated_at":         datetime.now(timezone.utc).isoformat(),
        "model_version":        str(model_version),
        "model_alias":          MODEL_ALIAS,
        "n_predictions_scored": int(len(current_scores)),
        "lookback_days":        LOOKBACK_DAYS,
        "model_age_hours":      model_age,
    })

    # ── 6. Save to MinIO ──────────────────────────────────────────────────────
    save_path = _save_to_minio(estimates, now_tag)
    estimates["saved_to"] = save_path

    logger.info(
        "CBPE complete — AUC≈%.4f  F1≈%s  method=%s",
        estimates.get("estimated_auc") or 0,
        estimates.get("estimated_f1"),
        estimates.get("method"),
    )

    print(json.dumps(estimates, default=str))


if __name__ == "__main__":
    main()