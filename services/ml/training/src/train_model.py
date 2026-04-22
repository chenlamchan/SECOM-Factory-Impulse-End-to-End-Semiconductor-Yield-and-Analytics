"""
train_model.py — Step 4: Model Training
─────────────────────────────────────────
Trains XGBoost (champion) and LightGBM (challenger) on the prepared
train_features Iceberg table. Both runs are logged to the same MLflow
experiment; the run with higher test-set AUC is tagged as the winner.

Class imbalance handling:
  SECOM is heavily imbalanced (~93% Pass / ~7% Fail).
  XGBoost:  scale_pos_weight = n_negative / n_positive
  LightGBM: is_unbalance = True

The winning run_id is printed to stdout (captured as Airflow XCom)
so evaluate_model.py and validate_model.py can reference it.
"""

import os
import sys
import s3fs
import json
import logging
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

import mlflow
import mlflow.xgboost
import mlflow.lightgbm

from mlflow.models.signature import infer_signature

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score
)
from pyiceberg.catalog.sql import SqlCatalog
from config import ServiceConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config = ServiceConfig()

MODEL_TRAIN_VERSION = "1.0.0"

# Constants
MINIO_ENDPOINT = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key
CATALOG_URI = config.catalog_uri
CATALOG_USER = config.catalog_user
CATALOG_PASSWORD = config.catalog_password
S3_WAREHOUSE_PATH = config.minio_warehouse

META_FEATURES = ["missing_sensor_rate"]
TARGET_COL    = "binary_label"

def get_s3_filesys():
    return s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT, 'region_name': 'us-east-1'},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY
    )


def _get_catalog() -> SqlCatalog:

    return SqlCatalog(
        "secom_catalog",
        uri=CATALOG_URI,
        warehouse=S3_WAREHOUSE_PATH,
        **{
            "s3.endpoint": MINIO_ENDPOINT,
            "s3.access-key-id": MINIO_ACCESS_KEY,
            "s3.secret-access-key": MINIO_SECRET_KEY,
            "s3.path-style-access": "true",
            "s3.region": "us-east-1"
        }
    )

def _load_table(catalog: SqlCatalog, table: str) -> pd.DataFrame:
    logger.info("Loading %s...", table)
    tbl = catalog.load_table(table)
    df  = tbl.scan().to_arrow().to_pandas()
    logger.info("  → %d rows, %d cols", len(df), len(df.columns))
    return df


def _get_xy(df: pd.DataFrame, feature_cols: list, cat_features: list) -> tuple:
    valid_features = [c for c in feature_cols if c in df.columns and c not in [TARGET_COL, "Pass/Fail"]]
    X = df[valid_features].copy()
    
    # Cast categorical features for native handling by XGB/LGBM
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].astype("category")
            
    # Cast numerics to float32 for performance and memory efficiency
    numeric_cols = [c for c in valid_features if c not in cat_features]
    X[numeric_cols] = X[numeric_cols].astype(np.float32)
    
    y = df[TARGET_COL].values.astype(int)
    return X, y, valid_features


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": round(float(roc_auc_score(y_true, y_prob)), 5),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 5),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 5),
        "avg_precision": round(float(average_precision_score(y_true, y_prob)), 5),
    }


def _train_xgboost(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray,
    feature_names: list, params: dict, run_name: str, manifest: str
) -> tuple:
    """Train XGBoost, log to MLflow, return (run_id, auc)."""
    neg   = int((y_train == 0).sum())
    pos   = int((y_train == 1).sum())
    spw   = round(neg / max(pos, 1), 2)
    logger.info("XGBoost — train class balance: neg=%d pos=%d spw=%.2f", neg, pos, spw)

    model_params = {
        "n_estimators":      params.get("n_estimators", 300),
        "max_depth":         params.get("max_depth", 6),
        "learning_rate":     params.get("learning_rate", 0.05),
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  5,
        "scale_pos_weight":  spw,
        "eval_metric":       "auc",
        "enable_categorical": True,
        "random_state":      42,
        "n_jobs":            -1,
    }

    model = xgb.XGBClassifier(**model_params)

    with mlflow.start_run(run_name=run_name) as run:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )
        y_prob = model.predict_proba(X_test)[:, 1]
        m = _metrics(y_test, y_prob)

        mlflow.log_params(model_params)
        mlflow.log_metrics(m)
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("train_rows", len(y_train))
        mlflow.log_param("test_rows", len(y_test))

        mlflow.log_dict(manifest, "metadata/feature_manifest.json")

        signature = infer_signature(X_train, y_train)

        mlflow.xgboost.log_model(
            model, "model",
            registered_model_name=None,
            signature=signature
        )

        run_id = run.info.run_id
        logger.info("XGBoost run %s — AUC=%.5f F1=%.5f", run_id, m["auc"], m["f1"])
        return run_id, m["auc"], m


def _train_lightgbm(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray,
    feature_names: list, cat_features: list, params: dict, run_name: str, manifest: str
) -> tuple:
    """Train LightGBM challenger, log to MLflow, return (run_id, auc)."""
    logger.info("LightGBM — training challenger model...")

    model_params = {
        "n_estimators":   params.get("n_estimators", 300),
        "max_depth":      params.get("max_depth", 6),
        "learning_rate":  params.get("learning_rate", 0.05),
        "num_leaves":     63,
        "subsample":      0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "is_unbalance":   True,
        "objective":      "binary",
        "metric":         "auc",
        "random_state":   42,
        "n_jobs":         -1,
        "verbose":        -1,
    }

    model = lgb.LGBMClassifier(**model_params)

    with mlflow.start_run(run_name=run_name) as run:
        valid_cats = [c for c in cat_features if c in feature_names]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=valid_cats,
            callbacks=[lgb.early_stopping(20, verbose=False),
                       lgb.log_evaluation(50)],
        )
        y_prob = model.predict_proba(X_test)[:, 1]
        m = _metrics(y_test, y_prob)

        mlflow.log_params(model_params)
        mlflow.log_metrics(m)
        mlflow.log_param("model_type", "lightgbm")
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_dict(manifest, "metadata/feature_manifest.json")

        signature = infer_signature(X_train, y_train)
        mlflow.lightgbm.log_model(model, "model", registered_model_name=None, signature=signature)

        run_id = run.info.run_id
        logger.info("LightGBM run %s — AUC=%.5f F1=%.5f", run_id, m["auc"], m["f1"])
        return run_id, m["auc"], m


def main():
    parser = argparse.ArgumentParser(description="SECOM ML — Model Training")
    parser.add_argument("--n-estimators",  type=int,   default=300)
    parser.add_argument("--max-depth",     type=int,   default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--train-table",   default="ml.train_features")
    parser.add_argument("--test-table",    default="ml.test_features")
    parser.add_argument("--manifest-path", default="/tmp/feature_manifest.json", help="Path to read/write the manifest file")
    args = parser.parse_args()

    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
    
    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    experiment_name = os.environ.get("EXPERIMENT_NAME", "secom_yield_prediction")
    mlflow.set_experiment(experiment_name)

    params = {
        "n_estimators":  args.n_estimators,
        "max_depth":     args.max_depth,
        "learning_rate": args.learning_rate,
    }

    # ── Load Manifest ─────────────────────────────────────────────────────────
    manifest_path = args.manifest_path
    s3 = get_s3_filesys()

    try:
        with s3.open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        active_sensors = manifest.get("active_features_list", [])
        cat_features   = manifest.get("categorical_features", [])
        lag_features   = manifest.get("lag_features", [])
        
        extract_version = manifest.get("extraction_pipeline_version", "unknown")
        prepare_version = manifest.get("preparation_pipeline_version", "unknown")

        # Combine all features requested for training
        ALL_FEATURES = active_sensors + cat_features + lag_features + META_FEATURES
        
        logger.info("Loaded manifest: %d active sensors, %d categorical features.", 
                    len(active_sensors), len(cat_features))
        
    except FileNotFoundError:
        logger.error("Manifest file not found. Training aborted.")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    catalog     = _get_catalog()
    train_df    = _load_table(catalog, args.train_table)
    test_df     = _load_table(catalog, args.test_table)

    X_train, y_train, feature_names = _get_xy(train_df, ALL_FEATURES, cat_features)
    X_test,  y_test,  _             = _get_xy(test_df, ALL_FEATURES, cat_features)

    logger.info(
        "Training set: %d rows, %d features | Test set: %d rows",
        len(X_train), len(feature_names), len(X_test)
    )

    # ── Train both models ────────────────────────────────────────────────────
    xgb_run_id,  xgb_auc,  xgb_metrics  = _train_xgboost(
        X_train, y_train, X_test, y_test, feature_names, params,
        run_name="xgboost_champion", manifest=manifest
    )
    lgb_run_id,  lgb_auc,  lgb_metrics  = _train_lightgbm(
        X_train, y_train, X_test, y_test, feature_names, cat_features, params,
        run_name="lightgbm_challenger", manifest=manifest
    )

    # ── Pick the winner ───────────────────────────────────────────────────────
    if xgb_auc >= lgb_auc:
        winner_run_id = xgb_run_id
        winner_type   = "xgboost"
        winner_metrics = xgb_metrics
        logger.info("Winner: XGBoost (AUC=%.5f vs LightGBM AUC=%.5f)", xgb_auc, lgb_auc)
    else:
        winner_run_id = lgb_run_id
        winner_type   = "lightgbm"
        winner_metrics = lgb_metrics
        logger.info("Winner: LightGBM (AUC=%.5f vs XGBoost AUC=%.5f)", lgb_auc, xgb_auc)

    mlflow_client = mlflow.tracking.MlflowClient()
    mlflow_client.set_tag(winner_run_id, "champion_candidate", "true")
    mlflow_client.set_tag(winner_run_id, "model_type", winner_type)
    mlflow_client.set_tag(winner_run_id, "extract_version", extract_version)
    mlflow_client.set_tag(winner_run_id, "prepare_version", prepare_version)

    # ── Output run_id to stdout → Airflow XCom ───────────────────────────────
    result = {
        "run_id":       winner_run_id,
        "model_type":   winner_type,
        "auc":          winner_metrics["auc"],
        "f1":           winner_metrics["f1"],
        "xgb_run_id":   xgb_run_id,
        "lgb_run_id":   lgb_run_id,
    }

    winner_path = args.manifest_path.replace("feature_manifest.json", "winner_run_id.txt")

    logger.info("Writing winner run_id to %s", winner_path)
    with s3.open(winner_path, "w") as f:
        f.write(winner_run_id)

    logger.info("Training complete — winner: %s", winner_run_id)


if __name__ == "__main__":
    main()