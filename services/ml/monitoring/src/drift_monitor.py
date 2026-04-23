"""
drift_monitor.py — Continuous Monitoring with Evidently AI
────────────────────────────────────────────────────────────
Runs as a dedicated Airflow task (daily schedule, separate from training).
Also triggered by the secom_ingestion_processing_event_driven DAG whenever
a new bronze batch is ingested.

Modifications:
  - Dynamically fetches the active feature list from the @champion model manifest.
  - Resolves historical data by dynamically finding MAX(timestamp) in Trino.
  - Automatically maps Docker secrets to AWS env vars for MLflow artifact downloads.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import nats
import mlflow
from mlflow.tracking import MlflowClient
import s3fs
import tempfile
from config import ServiceConfig

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config = ServiceConfig()

MINIO_ENDPOINT = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key

PSI_THRESHOLD       = float(os.environ.get("DRIFT_PSI_THRESHOLD", "0.20"))
SCORE_PSI_THRESHOLD = 0.15
LOOKBACK_DAYS       = int(os.environ.get("DRIFT_LOOKBACK_DAYS", "7"))
NATS_URL            = os.environ.get("NATS_ENDPOINT", "nats://nats:4222")
NATS_SUBJECT        = "ml.drift.alert"
NATS_STREAM         = "ML_MONITORING"
EVIDENTLY_BUCKET    = os.environ.get("EVIDENTLY_BUCKET", "evidently-reports")
MODEL_NAME          = os.environ.get("MODEL_NAME", "secom_yield_predictor")
MODEL_ALIAS         = os.environ.get("MODEL_ALIAS", "champion")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


def _get_champion_active_sensors() -> list:
    """Downloads the manifest from the champion model and extracts active sensors."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()

    try:
        v = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        manifest_uri = f"runs:/{v.run_id}/metadata/feature_manifest.json"
        
        logger.info("Downloading manifest from %s", manifest_uri)
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=manifest_uri)
        
        with open(local_path, "r") as f:
            manifest = json.load(f)
            
        active_sensors = manifest.get("active_features_list", [])
        logger.info("Loaded %d active sensors for monitoring.", len(active_sensors))
        return active_sensors
    except Exception as e:
        logger.warning("Could not fetch active sensors from champion manifest: %s", e)
        return []


def _load_reference_from_trino() -> pd.DataFrame:
    """Load Phase I baseline stats from Trino gold_sensor_stats."""
    import trino
    host = os.environ.get("TRINO_HOST", "trino")
    port = int(os.environ.get("TRINO_PORT", "8080"))
    trino_catalog = os.environ.get("TRINO_CATALOG", "secom_catalog")

    conn = trino.dbapi.connect(
        host=host, port=port, user="admin", catalog=trino_catalog, schema="gold",
    )
    df = pd.read_sql_query(
        "SELECT sensor_id, mu, sigma FROM gold_sensor_stats WHERE line_id = 'LINE_A'",
        conn
    )
    conn.close()
    return df


def _load_current_window(active_sensors: list) -> pd.DataFrame:
    """Load the last LOOKBACK_DAYS of silver data using dynamic sensors and simulated time."""
    if not active_sensors:
        return pd.DataFrame()
        
    import trino
    host = os.environ.get("TRINO_HOST", "trino")
    port = int(os.environ.get("TRINO_PORT", "8080"))
    trino_catalog = os.environ.get("TRINO_CATALOG", "secom_catalog")

    conn = trino.dbapi.connect(
        host=host, port=port, user="admin", catalog=trino_catalog, schema="silver",
    )
    cursor = conn.cursor()
    
    # Find simulated current time
    cursor.execute("SELECT MAX(process_timestamp) FROM silver_secom_reporting")
    max_ts_row = cursor.fetchone()
    
    if not max_ts_row or not max_ts_row[0]:
        logger.warning("No data in silver_secom_reporting to monitor.")
        return pd.DataFrame()
        
    max_ts = max_ts_row[0]
    sensor_list = ", ".join([f'"{s}"' for s in active_sensors])

    query = f"""
        SELECT {sensor_list},
               CASE WHEN label_numeric = 1 THEN 1 ELSE 0 END AS binary_label
        FROM silver_secom_reporting
        WHERE process_timestamp >= TIMESTAMP '{max_ts}' - INTERVAL '{LOOKBACK_DAYS}' DAY
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    logger.info("Current window loaded: %d rows (Simulated time: %s, Lookback: %d days)", 
                len(df), max_ts, LOOKBACK_DAYS)
    return df


def _load_prediction_scores() -> pd.Series:
    """Load recent prediction probabilities from gold_model_predictions."""
    import trino
    host = os.environ.get("TRINO_HOST", "trino")
    port = int(os.environ.get("TRINO_PORT", "8080"))
    trino_catalog = os.environ.get("TRINO_CATALOG", "secom_catalog")

    conn = trino.dbapi.connect(
        host=host, port=port, user="admin", catalog=trino_catalog, schema="gold",
    )
    cursor = conn.cursor()
    
    cursor.execute("SELECT MAX(prediction_timestamp) FROM gold_model_predictions")
    max_ts_row = cursor.fetchone()
    
    if not max_ts_row or not max_ts_row[0]:
        return pd.Series(dtype=float)
        
    max_ts = max_ts_row[0]

    try:
        df = pd.read_sql_query(
            f"""
            SELECT defect_probability
            FROM gold_model_predictions
            WHERE prediction_timestamp >= TIMESTAMP '{max_ts}' - INTERVAL '{LOOKBACK_DAYS}' DAY
            """,
            conn
        )
        return df["defect_probability"]
    except Exception as e:
        logger.warning("Could not load prediction scores: %s", e)
        return pd.Series(dtype=float)
    finally:
        conn.close()


def _compute_psi(current: pd.Series, reference_mu: float, reference_sigma: float,
                 bins: int = 10) -> float:
    """PSI between current sensor distribution and the Phase I Gaussian baseline."""
    cur = current.dropna().values
    if len(cur) == 0 or reference_sigma == 0:
        return 0.0

    edges = np.linspace(
        reference_mu - 4 * reference_sigma,
        reference_mu + 4 * reference_sigma,
        bins + 1
    )
    cur_counts = np.histogram(cur, bins=edges)[0]

    from scipy.stats import norm
    ref_cdf = norm.cdf(edges, loc=reference_mu, scale=reference_sigma)
    ref_pct = np.diff(ref_cdf) + 1e-6
    ref_pct /= ref_pct.sum()

    cur_pct = (cur_counts + 1e-6) / (len(cur) + 1e-6 * bins)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def _run_evidently_report(current_df: pd.DataFrame,
                          reference_df: pd.DataFrame) -> tuple[str, dict]:
    try:
        report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
        my_report = report.run(reference_data=reference_df, current_data=current_df)

        with tempfile.NamedTemporaryFile(suffix=".html", mode="w+") as tmp:
            my_report.save_html(tmp.name)
            tmp.seek(0)
            html = tmp.read()

        summary = my_report.dict()
        
        drift_share = (
            summary.get("metrics", [{}])[0]
            .get("result", {})
            .get("share_of_drifted_columns", 0)
        )
        logger.info("Evidently report: %.1f%% of columns show drift", drift_share * 100)
        return html, {"drift_share": drift_share, "evidently_available": True}

    except ImportError:
        logger.warning("Evidently not installed — generating minimal drift report.")
        html = "<html><body><p>Evidently not available — PSI summary below.</p></body></html>"
        return html, {"drift_share": None, "evidently_available": False}


def _save_report_to_minio(html: str, report_key: str) -> str:
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY
    )

    path = f"{EVIDENTLY_BUCKET}/{report_key}"
    with fs.open(path, "w") as f:
        f.write(html)

    logger.info("Evidently report saved to s3://%s", path)
    return path


async def _publish_drift_alert(payload: dict) -> None:
    try:
        nc = await nats.connect(NATS_URL)
        js = nc.jetstream()
        try:
            await js.add_stream(name=NATS_STREAM, subjects=[NATS_SUBJECT])
        except Exception:
            pass 

        msg = json.dumps(payload).encode("utf-8")
        ack = await js.publish(NATS_SUBJECT, msg)
        logger.info("Drift alert published to NATS — seq=%d", ack.seq)
        await nc.close()
    except Exception as e:
        logger.error("Failed to publish NATS drift alert: %s", e)


def main():
    logger.info("Starting drift monitor (lookback=%d days, PSI threshold=%.2f)",
                LOOKBACK_DAYS, PSI_THRESHOLD)

    # ── Dynamically load features ─────────────────────────────────────────────
    active_sensors = _get_champion_active_sensors()
    if not active_sensors:
        logger.warning("No active sensors retrieved. Cannot perform drift check.")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    reference_stats = _load_reference_from_trino()
    current_df      = _load_current_window(active_sensors)

    if len(current_df) == 0:
        logger.warning("No current window data — skipping drift check.")
        return

    ref_lookup = {
        row["sensor_id"]: (row["mu"], row["sigma"])
        for _, row in reference_stats.iterrows()
    }

    # ── PSI per sensor ────────────────────────────────────────────────────────
    psi_results = []
    flagged_sensors = []

    for sensor in active_sensors:
        if sensor not in current_df.columns:
            continue
        mu, sigma = ref_lookup.get(sensor, (0.0, 1.0))
        psi = _compute_psi(current_df[sensor], mu, sigma)

        level = "red" if psi > PSI_THRESHOLD else ("amber" if psi > 0.10 else "green")
        psi_results.append({"sensor": sensor, "psi": round(psi, 4), "level": level})

        if psi > PSI_THRESHOLD:
            flagged_sensors.append(sensor)
            logger.warning("PSI alert: sensor %s PSI=%.4f > %.2f", sensor, psi, PSI_THRESHOLD)

    # ── Score distribution drift ───────────────────────────────────────────────
    pred_scores = _load_prediction_scores()
    score_psi   = 0.0
    score_drift = False

    if len(pred_scores) > 50:
        score_psi   = _compute_psi(pred_scores, reference_mu=0.07, reference_sigma=0.15)
        score_drift = score_psi > SCORE_PSI_THRESHOLD
        logger.info("Score distribution PSI=%.4f (drift=%s)", score_psi, score_drift)

    # ── Evidently report ──────────────────────────────────────────────────────
    mid = len(current_df) // 2
    html, evidently_summary = _run_evidently_report(
        current_df=current_df.iloc[mid:][active_sensors].copy(),
        reference_df=current_df.iloc[:mid][active_sensors].copy(),
    )

    report_key = f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html"
    report_path = _save_report_to_minio(html, report_key)

    # ── Summarise and decide ──────────────────────────────────────────────────
    drift_detected = bool(flagged_sensors) or score_drift
    summary = {
        "drift_detected":     drift_detected,
        "flagged_sensors":    flagged_sensors,
        "score_psi":          round(score_psi, 4),
        "score_drift":        score_drift,
        "psi_results":        psi_results,
        "evidently_summary":  evidently_summary,
        "report_path":        report_path,
        "checked_at":         datetime.now(timezone.utc).isoformat(),
        "lookback_days":      LOOKBACK_DAYS,
        "psi_threshold":      PSI_THRESHOLD,
        "current_window_rows": len(current_df),
    }

    if drift_detected:
        alert_payload = {
            "alert_type":      "DRIFT_DETECTED",
            "trigger":         "evidently_psi",
            "flagged_sensors": flagged_sensors,
            "score_drift":     score_drift,
            "report_path":     report_path,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
        asyncio.run(_publish_drift_alert(alert_payload))
        logger.warning("NATS drift alert published — retraining will be triggered.")
    else:
        logger.info("No significant drift detected — no alert published.")

    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()