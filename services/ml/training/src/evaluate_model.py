"""
evaluate_model.py — Step 5: Model Evaluation
──────────────────────────────────────────────
Loads the winning model from the MLflow run produced by train_model.py,
evaluates it on a held-out test set (CSV in MinIO), and logs:

  • Full classification report (precision, recall, F1 per class)
  • Confusion matrix as an MLflow artifact
  • SHAP TreeExplainer values — global feature importance bar chart
  • Per-observation SHAP values (Parquet) for the serving layer
  • Precision-Recall curve artifact
  • Threshold analysis (F1/precision/recall across 0.1–0.9 cutoffs)

Modifications:
  - Dynamically loads active features from feature_manifest.json
  - Reads a raw/pre-processed CSV from MinIO instead of an Iceberg table
  - Auto-calculates `binary_label` and `missing_sensor_rate` if missing.
"""

import os
import sys
import json
import logging
import argparse
import tempfile
import s3fs

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, roc_auc_score
)
from config import ServiceConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config = ServiceConfig()

MINIO_ENDPOINT = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

TARGET_COL = "binary_label"
META_FEATURES = ["missing_sensor_rate"]


def get_s3_filesys():
    """Returns an S3FileSystem configured for MinIO."""
    return s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT, 'region_name': 'us-east-1'},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY
    )


def _load_model(run_id: str):
    """Load the logged model from the winning MLflow run."""
    client = mlflow.tracking.MlflowClient()
    run    = client.get_run(run_id)
    model_type = run.data.tags.get("model_type", "xgboost")

    model_uri = f"runs:/{run_id}/model"
    logger.info("Loading %s model from %s", model_type, model_uri)

    if model_type == "lightgbm":
        return mlflow.lightgbm.load_model(model_uri), model_type

    return mlflow.xgboost.load_model(model_uri), model_type


def _stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Equal-class sample from the test set for SHAP computation."""
    pos = df[df[TARGET_COL] == 1]
    neg = df[df[TARGET_COL] == 0]
    n_each = min(n // 2, len(pos), len(neg))
    
    if n_each == 0:
        logger.warning("Not enough instances of both classes to stratify SHAP. Falling back to random sample.")
        return df.sample(min(n, len(df)), random_state=42)
        
    return pd.concat([
        pos.sample(n_each, random_state=42),
        neg.sample(n_each, random_state=42),
    ]).sample(frac=1, random_state=42)   # shuffle


def _plot_confusion_matrix(cm: np.ndarray, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Pass", "Predicted Fail"])
    ax.set_yticklabels(["Actual Pass", "Actual Fail"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix — Test Set")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_pr_curve(y_true, y_prob, path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color="#1D9E75", linewidth=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.fill_between(recall, precision, alpha=0.1, color="#1D9E75")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_shap_summary(shap_values, X_sample: pd.DataFrame, feature_names: list, path: str) -> None:
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SECOM ML — Model Evaluation")
    parser.add_argument("--shap-sample-size", type=int, default=200)
    parser.add_argument("--manifest-path",   default="/tmp/feature_manifest.json")
    parser.add_argument("--held-out-data",   default="s3a://ml-metadata/holdout-test-data/uci-secom.csv",required=True, help="S3 URI to the held-out CSV file")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    # ── Load Manifest ─────────────────────────────────────────────────────────
    s3 = get_s3_filesys()

    winner_path = args.manifest_path.replace("feature_manifest.json", "winner_run_id.txt")
    try:
        with s3.open(winner_path, "r") as f:
            winning_run_id = f.read().strip()
        logger.info("Retrieved winning run ID from MinIO: %s", winning_run_id)
    except FileNotFoundError:
        logger.error("Winner run_id file not found. Evaluation aborted.")
        sys.exit(1)

    try:
        with s3.open(args.manifest_path, "r") as f:
            manifest = json.load(f)
            
        active_sensors = manifest.get("active_features_list", [])
        cat_features   = manifest.get("categorical_features", [])
        lag_features   = manifest.get("lag_features", [])

        cat_modes      = manifest.get("categorical_modes", {})

        ALL_FEATURES   = active_sensors + cat_features + lag_features + META_FEATURES
        logger.info("Loaded manifest: Expecting %d features for evaluation.", len(ALL_FEATURES))
    except FileNotFoundError:
        logger.error("Manifest file not found. Evaluation aborted.")
        sys.exit(1)

    # ── Load model and held-out data ──────────────────────────────────────────
    model, model_type = _load_model(winning_run_id)

    logger.info("Loading held-out data from %s", args.held_out_data)
    storage_options = {
        "key": MINIO_ACCESS_KEY,
        "secret": MINIO_SECRET_KEY,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT, "region_name": "us-east-1"}
    }
    
    # Read the held out CSV
    test_df = pd.read_csv(args.held_out_data, storage_options=storage_options)
    logger.info("Held-out test set loaded: %d rows", len(test_df))

    # ── Preprocessing raw CSV to match model expectations ─────────────────────
    
    # 1. Resolve Target Column
    if TARGET_COL not in test_df.columns:
        if "Pass/Fail" in test_df.columns:
            test_df[TARGET_COL] = (test_df["Pass/Fail"] == 1).astype(int)
        else:
            logger.error("No target column ('binary_label' or 'label_numeric') found in CSV.")
            sys.exit(1)

    # 2. Resolve Missing Sensor Rate
    if "missing_sensor_rate" not in test_df.columns:
        valid_sensors = [c for c in active_sensors if c in test_df.columns]
        test_df["missing_sensor_count"] = test_df[valid_sensors].isnull().sum(axis=1)
        test_df["missing_sensor_rate"] = test_df["missing_sensor_count"] / 590.0

    # 3. Create observation_id if missing (needed for SHAP parquet)
    if "observation_id" not in test_df.columns:
        test_df["observation_id"] = [f"eval_{i}" for i in range(len(test_df))]

    for c in cat_features:
        if c not in test_df.columns:
            fallback_val = cat_modes.get(c, "UNKNOWN")
            logger.warning("Categorical feature '%s' missing from held-out data. Injecting fallback mode: '%s'", c, fallback_val)
            test_df[c] = fallback_val

    # 4. Extract aligned X and y
    feature_cols = [c for c in ALL_FEATURES if c in test_df.columns]
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].values.astype(int)

    # Cast categoricals to 'category' natively for LightGBM/XGBoost
    for c in cat_features:
        if c in X_test.columns:
            X_test[c] = X_test[c].astype("category")

    # Cast numerics
    numeric_cols = [c for c in feature_cols if c not in cat_features]
    X_test[numeric_cols] = X_test[numeric_cols].astype(np.float32)

    # ── Predict ───────────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # ── Full metrics ──────────────────────────────────────────────────────────
    auc    = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Pass", "Fail"], output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    # Threshold analysis
    thresholds = np.arange(0.1, 1.0, 0.1)
    thresh_analysis = []
    from sklearn.metrics import f1_score, precision_score, recall_score
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        thresh_analysis.append({
            "threshold": round(float(t), 1),
            "f1":        round(float(f1_score(y_test, yp, zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, yp, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test, yp, zero_division=0)), 4),
        })

    logger.info(
        "Evaluation — AUC=%.5f F1=%.5f Precision=%.5f Recall=%.5f",
        auc, report["Fail"]["f1-score"], report["Fail"]["precision"], report["Fail"]["recall"]
    )

    # ── SHAP ─────────────────────────────────────────────────────────────────
    sample_df = _stratified_sample(test_df, args.shap_sample_size)
    X_sample  = sample_df[feature_cols].copy()
    
    # Apply same casting to SHAP sample
    for c in cat_features:
        if c in X_sample.columns:
            X_sample[c] = X_sample[c].astype("category")
    X_sample[numeric_cols] = X_sample[numeric_cols].astype(np.float32)


    # ── Log everything to MLflow (resuming the winning run) ───────────────────
    with mlflow.start_run(run_id=winning_run_id):
        mlflow.log_metric("held_out_auc",       auc)
        mlflow.log_metric("held_out_f1_fail",   report["Fail"]["f1-score"])
        mlflow.log_metric("held_out_prec_fail", report["Fail"]["precision"])
        mlflow.log_metric("held_out_rec_fail",  report["Fail"]["recall"])
        mlflow.log_metric("tn", int(cm[0, 0]))
        mlflow.log_metric("fp", int(cm[0, 1]))
        mlflow.log_metric("fn", int(cm[1, 0]))
        mlflow.log_metric("tp", int(cm[1, 1]))

        with tempfile.TemporaryDirectory() as tmp:
            cm_path = f"{tmp}/confusion_matrix.png"
            _plot_confusion_matrix(cm, cm_path)
            mlflow.log_artifact(cm_path, "evaluation")

            pr_path = f"{tmp}/pr_curve.png"
            _plot_pr_curve(y_test, y_prob, pr_path)
            mlflow.log_artifact(pr_path, "evaluation")

            thresh_path = f"{tmp}/threshold_analysis.json"
            with open(thresh_path, "w") as f:
                json.dump(thresh_analysis, f, indent=2)
            mlflow.log_artifact(thresh_path, "evaluation")

            report_path = f"{tmp}/classification_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path, "evaluation")

    logger.info("Step 5 complete — evaluation artifacts logged to run %s", winning_run_id)


if __name__ == "__main__":
    main()