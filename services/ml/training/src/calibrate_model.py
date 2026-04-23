"""
calibrate_model.py — Step 4b: Post-hoc Probability Calibration
────────────────────────────────────────────────────────────────
XGBoost and LightGBM push predicted probabilities toward 0 and 1
(overconfident). Raw `predict_proba()` values are NOT true probabilities —
this is well-documented and breaks:
  - Any threshold analysis beyond naive 0.5
  - Confidence-Based Performance Estimation (CBPE / NannyML)
  - Reliability of the confidence_band field in the serving layer

This step applies Platt scaling (sigmoid calibration) fit on a held-out
calibration fold from the TRAINING set (not the test set — that would
introduce leakage into the champion gate comparison).

The calibrated model wrapper is logged as a separate MLflow artifact
("calibrated_model") alongside the raw model so both are available.
Calibration quality is measured by Brier score and Expected Calibration
Error (ECE) and logged as metrics.

The calibration dataset (X_cal, y_cal) is also stored as a Parquet
artifact — this is the reference dataset for NannyML CBPE.

Insert between train_model and evaluate_model in the Airflow DAG.

"""

import os
import sys
import json
import logging
import argparse
import tempfile
import pickle

import numpy as np
import pandas as pd
import s3fs                          # CHANGE: was missing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from pyiceberg.catalog.sql import SqlCatalog

from config import ServiceConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "binary_label"
CAL_FRACTION = 0.15

# CHANGE: was referenced but never defined
META_FEATURES = ["missing_sensor_rate"]

config = ServiceConfig()

MINIO_ENDPOINT   = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key
CATALOG_URI      = config.catalog_uri
CATALOG_USER     = config.catalog_user
CATALOG_PASSWORD = config.catalog_password
S3_WAREHOUSE_PATH = config.minio_warehouse

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]      = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]  = MINIO_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"]     = "us-east-1"


def _get_catalog() -> SqlCatalog:
    return SqlCatalog(
        "secom_catalog",
        uri=CATALOG_URI,
        warehouse=S3_WAREHOUSE_PATH,
        **{
            "s3.endpoint":           MINIO_ENDPOINT,
            "s3.access-key-id":      MINIO_ACCESS_KEY,
            "s3.secret-access-key":  MINIO_SECRET_KEY,
            "s3.path-style-access":  "true",
            "s3.region":             "us-east-1",
        },
    )


def get_s3_filesys():
    """Returns an S3FileSystem configured for MinIO."""
    # CHANGE: function existed but s3fs was never imported — now fixed above
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": MINIO_ENDPOINT, "region_name": "us-east-1"},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
    )


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                 n_bins: int = 10) -> float:
    """
    ECE: weighted average of |accuracy - confidence| across probability bins.
    Lower is better. Perfectly calibrated model → ECE = 0.
    ECE > 0.05 is considered poorly calibrated for high-stakes decisions.
    """
    bins    = np.linspace(0, 1, n_bins + 1)
    ece     = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        mask     = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (n_in_bin / n_total) * abs(acc - conf)

    return float(ece)


def _load_model(run_id: str):
    """Load the logged model from the winning MLflow run."""
    client     = mlflow.tracking.MlflowClient()
    run        = client.get_run(run_id)
    model_type = run.data.tags.get("model_type", "xgboost")   # CHANGE: store as model_type

    model_uri = f"runs:/{run_id}/model"
    logger.info("Loading %s model from %s", model_type, model_uri)

    if model_type == "lightgbm":
        return mlflow.lightgbm.load_model(model_uri), model_type   # CHANGE: return model_type

    return mlflow.xgboost.load_model(model_uri), model_type        # CHANGE: return model_type


def _reliability_diagram(y_true: np.ndarray, y_prob_raw: np.ndarray,
                          y_prob_cal: np.ndarray, path: str) -> None:
    """Reliability diagram comparing raw vs calibrated probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, y_prob, title in [
        (axes[0], y_prob_raw, "Before calibration (raw XGB/LGBM)"),
        (axes[1], y_prob_cal, "After Platt scaling"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, "s-", color="#1D9E75", linewidth=2,
                label="Model")
        ax.fill_between(mean_pred, frac_pos, mean_pred,
                        alpha=0.15, color="#E24B4A")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    plt.suptitle("Reliability Diagram — Platt Scaling Calibration", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SECOM ML — Model Calibration")
    parser.add_argument("--train-table",   default="ml.train_features")
    parser.add_argument("--manifest-path", default="/tmp/feature_manifest.json")
    # CHANGE: removed --run-id arg — winner is always read from MinIO winner_run_id.txt
    #         consistent with evaluate_model.py and validate_model.py
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()
    s3     = get_s3_filesys()

    # ── Load winner run_id from MinIO ──────────────────────────────────────────
    # CHANGE: was `args.run_id` (undefined) — now reads from MinIO txt file
    #         identical pattern to evaluate_model.py and validate_model.py
    winner_path = args.manifest_path.replace("feature_manifest.json", "winner_run_id.txt")
    try:
        with s3.open(winner_path, "r") as f:
            winning_run_id = f.read().strip()
        logger.info("Retrieved winning run ID from MinIO: %s", winning_run_id)
    except FileNotFoundError:
        logger.error("Winner run_id file not found at %s. Calibration aborted.", winner_path)
        sys.exit(1)

    # ── Load manifest (dynamic feature contract) ───────────────────────────────
    # CHANGE: was missing META_FEATURES definition — now explicitly defined above
    try:
        with s3.open(args.manifest_path, "r") as f:
            manifest = json.load(f)

        active_sensors = manifest.get("active_features_list", [])
        cat_features   = manifest.get("categorical_features", [])
        cat_modes      = manifest.get("categorical_modes", {})
        lag_features   = manifest.get("lag_features", [])
        ALL_FEATURES   = active_sensors + cat_features + lag_features + META_FEATURES
        logger.info("Manifest loaded: %d active sensors, %d categorical, %d lag, %d meta",
                    len(active_sensors), len(cat_features), len(lag_features), len(META_FEATURES))
    except FileNotFoundError:
        logger.error("Manifest file not found at %s. Calibration aborted.", args.manifest_path)
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    # CHANGE: `model_type` is now correctly returned and used throughout
    model, model_type = _load_model(winning_run_id)

    # ── Load training data from Iceberg ───────────────────────────────────────
    catalog  = _get_catalog()
    train_df = catalog.load_table(args.train_table).scan().to_arrow().to_pandas()
    logger.info("Training set loaded: %d rows, %d cols", len(train_df), len(train_df.columns))

    # Build aligned feature matrix using manifest-defined ALL_FEATURES
    feature_cols = [c for c in ALL_FEATURES if c in train_df.columns and c != TARGET_COL]

    for c in cat_features:
        if c in train_df.columns:
            train_df[c] = train_df[c].astype("category")

    numeric_cols = [c for c in feature_cols if c not in cat_features]
    train_df[numeric_cols] = train_df[numeric_cols].astype(np.float32)

    X_all = train_df[feature_cols].copy()
    
    for c in cat_features:
        if c in X_all.columns:
            X_all[c] = X_all[c].astype("category")

    numeric_cols = [c for c in feature_cols if c not in cat_features]

    X_all[numeric_cols] = X_all[numeric_cols].astype(np.float32)
    y_all = train_df[TARGET_COL].values.astype(int)

    # ── Temporal calibration split ─────────────────────────────────────────────
    # Most recent CAL_FRACTION rows as calibration set (approximates production dist)
    n_cal = int(len(X_all) * CAL_FRACTION)
    X_fit, X_cal = X_all[:-n_cal], X_all[-n_cal:]
    y_fit, y_cal = y_all[:-n_cal], y_all[-n_cal:]

    logger.info("Calibration split: fit=%d rows, calibration=%d rows", len(X_fit), len(X_cal))

    # ── Raw model calibration metrics ─────────────────────────────────────────
    raw_prob  = model.predict_proba(X_cal)[:, 1]
    raw_brier = brier_score_loss(y_cal, raw_prob)
    raw_ece   = _expected_calibration_error(y_cal, raw_prob)
    logger.info("Raw model — Brier=%.5f ECE=%.5f", raw_brier, raw_ece)

    # ── Platt scaling (sigmoid calibration) ──────────────────────────────────
    frozen_model = FrozenEstimator(model)
    calibrated = CalibratedClassifierCV(estimator=frozen_model, method="sigmoid")
    calibrated.fit(X_cal, y_cal)

    cal_prob  = calibrated.predict_proba(X_cal)[:, 1]
    cal_brier = brier_score_loss(y_cal, cal_prob)
    cal_ece   = _expected_calibration_error(y_cal, cal_prob)

    improvement_brier = (raw_brier - cal_brier) / raw_brier * 100
    improvement_ece   = (raw_ece   - cal_ece)   / raw_ece   * 100
    logger.info("Calibrated model — Brier=%.5f ECE=%.5f (Brier Δ%.1f%% ECE Δ%.1f%%)",
                cal_brier, cal_ece, improvement_brier, improvement_ece)

    # ── Build CBPE reference dataset ──────────────────────────────────────────
    # feature_cols filtered to available columns to avoid KeyError on partial data
    cal_feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    cal_reference_df = X_cal.copy()
    cal_reference_df["binary_label"]    = y_cal
    cal_reference_df["calibrated_prob"] = cal_prob
    cal_reference_df["raw_prob"]        = raw_prob

    # ── Log all artifacts to the winning run in MLflow ─────────────────────────
    # CHANGE: was `with mlflow.start_run(run_id=args.run_id)` — args.run_id was never
    #         defined in argparse. Now correctly uses `winning_run_id`.
    with mlflow.start_run(run_id=winning_run_id):

        # Calibration metrics
        mlflow.log_metric("cal_brier_raw",             raw_brier)
        mlflow.log_metric("cal_brier_calibrated",      cal_brier)
        mlflow.log_metric("cal_ece_raw",               raw_ece)
        mlflow.log_metric("cal_ece_calibrated",        cal_ece)
        mlflow.log_metric("cal_brier_improvement_pct", improvement_brier)
        mlflow.log_metric("cal_ece_improvement_pct",   improvement_ece)
        mlflow.log_param("calibration_method",         "platt_sigmoid")
        mlflow.log_param("cal_fraction",               CAL_FRACTION)
        mlflow.log_param("n_cal_rows",                 len(X_cal))

        with tempfile.TemporaryDirectory() as tmp:

            # Reliability diagram
            rd_path = f"{tmp}/reliability_diagram.png"
            _reliability_diagram(y_cal, raw_prob, cal_prob, rd_path)
            mlflow.log_artifact(rd_path, "calibration")

            # Calibrated model pickle (sklearn wrapper around XGB/LGBM)
            # CHANGE: was `m_type` (undefined) — now correctly uses `model_type`
            cal_path = f"{tmp}/calibrated_model.pkl"
            with open(cal_path, "wb") as f:
                pickle.dump({
                    "calibrated_model":   calibrated,
                    "feature_names":      cal_feature_cols,
                    "model_type":         model_type,      # CHANGE: was m_type (undefined)
                    "calibration_method": "platt_sigmoid",
                }, f)
            mlflow.log_artifact(cal_path, "calibration")

            # CBPE reference dataset (consumed by performance_estimator.py)
            ref_path = f"{tmp}/cbpe_reference_dataset.parquet"
            cal_reference_df.to_parquet(ref_path, index=False)
            mlflow.log_artifact(ref_path, "calibration")

            # Calibration metadata JSON (consumed by serving_extensions.py)
            meta = {
                "brier_raw":          round(raw_brier, 6),
                "brier_calibrated":   round(cal_brier, 6),
                "ece_raw":            round(raw_ece, 6),
                "ece_calibrated":     round(cal_ece, 6),
                "n_cal_rows":         len(X_cal),
                "feature_names":      cal_feature_cols,
                "model_type":         model_type,          # CHANGE: was m_type
                "calibration_method": "platt_sigmoid",
                "well_calibrated":    cal_ece < 0.05,
            }
            meta_path = f"{tmp}/calibration_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            mlflow.log_artifact(meta_path, "calibration")

        # Tag the run
        client.set_tag(winning_run_id, "calibrated",          "true")
        client.set_tag(winning_run_id, "calibration_method",  "platt_sigmoid")
        client.set_tag(winning_run_id, "ece_calibrated",      str(round(cal_ece, 6)))

    if cal_ece > 0.05:
        logger.warning(
            "ECE=%.4f exceeds 0.05 threshold — model may still be poorly calibrated. "
            "Consider isotonic regression or more calibration data.", cal_ece
        )
    else:
        logger.info("Calibration successful — ECE=%.4f (target < 0.05)", cal_ece)

    # Print run_id for Airflow XCom passthrough
    print(winning_run_id)


if __name__ == "__main__":
    main()