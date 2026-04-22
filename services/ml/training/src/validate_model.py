"""
validate_model.py — Step 6: Model Validation (Champion Gate)
──────────────────────────────────────────────────────────────
Compares the candidate model (from evaluate_model.py) against the currently
deployed Production model in the MLflow registry.

Decision logic:
  1. If no Production model exists → auto-promote (first deployment).
  2. If candidate AUC >= production AUC + --auc-min-delta
     AND candidate F1  >= production F1  + --f1-min-delta → promote.
  3. Otherwise → reject. 

On promotion:
  - Registers the model version in the MLflow registry.
  - Transitions it to "Staging" first, runs a smoke-test prediction using
    manifest-aligned features, then promotes to "Production".
  - Archives the previous Production version.
  - Tags the run with promotion metadata.
"""

import os
import sys
import json
import logging
import argparse
import time
import s3fs

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from config import ServiceConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config = ServiceConfig()

VALIDATE_PIPELINE_VERSION = "1.0.0"

MINIO_ENDPOINT = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key

META_FEATURES = ["missing_sensor_rate"]


def get_s3_filesys():
    """Returns an S3FileSystem configured for MinIO."""
    return s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT, 'region_name': 'us-east-1'},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY
    )


def _get_production_metrics(client: MlflowClient, model_name: str) -> dict | None:
    """Return metrics of the current Production model, or None if no Production version."""
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        prod_run_id = versions[0].run_id
        run = client.get_run(prod_run_id)
        return {
            "version": versions[0].version,
            "run_id":  prod_run_id,
            "auc":     float(run.data.metrics.get("held_out_auc", 0.0)),
            "f1":      float(run.data.metrics.get("held_out_f1_fail", 0.0)),
        }
    except MlflowException as e:
        logger.warning("Could not fetch Production metrics: %s", e)
        return None


def _smoke_test(model_uri: str, model_type: str, feature_names: list, cat_features: list, cat_modes: dict) -> bool:
    """
    Quick inference check.
    Dynamically builds a 1-row Pandas DataFrame matching the manifest 
    to properly test native categorical support and exact feature shapes.
    """
    try:
        if model_type == "lightgbm":
            model = mlflow.lightgbm.load_model(model_uri)
        else:
            model = mlflow.xgboost.load_model(model_uri)

        # Build a synthetic feature vector
        dummy_data = {}
        for c in feature_names:
            if c in cat_features:
                dummy_data[c] = [cat_modes.get(c, "UNKNOWN")]
            else:
                dummy_data[c] = [0.0]

        dummy_df = pd.DataFrame(dummy_data)

        # Cast to exact types expected by the model
        for c in cat_features:
            if c in dummy_df.columns:
                dummy_df[c] = dummy_df[c].astype("category")
                
        numeric_cols = [c for c in feature_names if c not in cat_features]
        dummy_df[numeric_cols] = dummy_df[numeric_cols].astype(np.float32)

        prob = model.predict_proba(dummy_df)
        assert 0.0 <= prob[0, 1] <= 1.0, f"Probability out of range: {prob}"
        logger.info("Smoke test passed — dummy probability=%.4f", prob[0, 1])
        return True
    except Exception as e:
        logger.error("Smoke test FAILED: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="SECOM ML — Champion Gate")
    parser.add_argument("--run-id",        required=True)
    parser.add_argument("--auc-min-delta", type=float, default=-0.005,
                        help="Min AUC improvement vs production (negative = allow regression).")
    parser.add_argument("--f1-min-delta",  type=float, default=-0.010)
    parser.add_argument("--model-name",    default=None)
    parser.add_argument("--manifest-path", default="/tmp/feature_manifest.json")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    model_name = args.model_name or os.environ.get("MODEL_NAME", "secom_yield_predictor")
    client     = MlflowClient()

    # ── Load Manifest for Smoke Test ──────────────────────────────────────────
    s3 = get_s3_filesys()
    try:
        with s3.open(args.manifest_path, "r") as f:
            manifest = json.load(f)
            
        active_sensors = manifest.get("active_features_list", [])
        cat_features   = manifest.get("categorical_features", [])
        cat_modes      = manifest.get("categorical_modes", {})
        lag_features   = manifest.get("lag_features", [])
        ALL_FEATURES   = active_sensors + cat_features + lag_features + META_FEATURES
    except FileNotFoundError:
        logger.error("Manifest file not found. Validation aborted.")
        sys.exit(1)

    # ── Fetch candidate metrics ───────────────────────────────────────────────
    candidate_run = client.get_run(args.run_id)

    cand_auc = float(candidate_run.data.metrics.get("held_out_auc", 0.0))
    cand_f1  = float(candidate_run.data.metrics.get("held_out_f1_fail", 0.0))
    cand_type = candidate_run.data.tags.get("model_type", "xgboost")

    logger.info(
        "Candidate — run_id=%s type=%s AUC=%.5f F1=%.5f",
        args.run_id, cand_type, cand_auc, cand_f1
    )

    # ── Fetch production metrics ──────────────────────────────────────────────
    prod_metrics = _get_production_metrics(client, model_name)

    result = {
        "promoted":    False,
        "version":     None,
        "new_auc":     cand_auc,
        "new_f1":      cand_f1,
        "prod_auc":    prod_metrics["auc"] if prod_metrics else None,
        "prod_f1":     prod_metrics["f1"]  if prod_metrics else None,
        "reason":      "",
    }

    # ── Champion decision ──────────────────────────────────────────────────────
    if prod_metrics is None:
        decision = "auto_promote"
        reason   = "No Production model exists — auto-promoting first version."
        logger.info(reason)
    else:
        auc_delta = cand_auc - prod_metrics["auc"]
        f1_delta  = cand_f1  - prod_metrics["f1"]
        logger.info(
            "AUC delta=%.5f (min=%.5f) | F1 delta=%.5f (min=%.5f)",
            auc_delta, args.auc_min_delta, f1_delta, args.f1_min_delta
        )
        if auc_delta >= args.auc_min_delta and f1_delta >= args.f1_min_delta:
            decision = "promote"
            reason   = (
                f"Candidate passes gate — AUC {cand_auc:.5f} (Δ{auc_delta:+.5f}), "
                f"F1 {cand_f1:.5f} (Δ{f1_delta:+.5f})."
            )
        else:
            decision = "reject"
            reason   = (
                f"Candidate did NOT pass gate — AUC Δ{auc_delta:+.5f} (min {args.auc_min_delta}), "
                f"F1 Δ{f1_delta:+.5f} (min {args.f1_min_delta}). "
                f"Keeping current Production v{prod_metrics['version']}."
            )

    result["reason"] = reason

    if decision == "reject":
        logger.warning("Champion gate REJECTED: %s", reason)
        print(json.dumps(result))
        sys.exit(0)

    # ── Register and promote ───────────────────────────────────────────────────
    logger.info("Registering model '%s' from run %s...", model_name, args.run_id)

    mv = mlflow.register_model(
        model_uri=f"runs:/{args.run_id}/model",
        name=model_name,
    )
    for _ in range(30):
        mv = client.get_model_version(model_name, mv.version)
        if mv.status == "READY":
            break
        time.sleep(2)

    # Transition to Staging and smoke-test
    client.transition_model_version_stage(
        name=model_name, version=mv.version, stage="Staging"
    )
    logger.info("Transitioned v%s to Staging — running smoke test...", mv.version)

    staging_uri = f"models:/{model_name}/Staging"
    
    smoke_ok    = _smoke_test(staging_uri, cand_type, ALL_FEATURES, cat_features, cat_modes)

    if not smoke_ok:
        result["reason"] = f"Smoke test FAILED for v{mv.version} — not promoting."
        client.transition_model_version_stage(
            name=model_name, version=mv.version, stage="Archived"
        )
        print(json.dumps(result))
        sys.exit(0)

    if prod_metrics:
        client.transition_model_version_stage(
            name=model_name, version=prod_metrics["version"], stage="Archived"
        )
        logger.info("Archived previous Production v%s", prod_metrics["version"])

    client.transition_model_version_stage(
        name=model_name, version=mv.version, stage="Production"
    )
    logger.info("Promoted v%s to Production.", mv.version)

    # Tag the winning run
    client.set_tag(args.run_id, "promoted_to_production", "true")
    client.set_tag(args.run_id, "model_version", mv.version)
    client.set_tag(args.run_id, "validate_version", VALIDATE_PIPELINE_VERSION)

    result["promoted"] = True
    result["version"]  = mv.version
    logger.info("Champion gate PROMOTED: %s", reason)
    print(json.dumps(result))


if __name__ == "__main__":
    main()