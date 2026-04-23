"""
serving_extensions.py — Additional FastAPI Endpoints for ml/serving/main.py
────────────────────────────────────────────────────────────────────────────
Append these router endpoints to the main FastAPI app in main.py.

Add to main.py:
    from serving_extensions import router as explain_router
    app.include_router(explain_router, prefix="/explain", tags=["explainability"])

Endpoints added:
    POST /explain/counterfactual   — DiCE + MILP counterfactuals for a failing wafer
    GET  /explain/performance      — Latest CBPE estimated performance metrics
    POST /explain/validate-input   — Mahalanobis distance gate for serving-time validation

"""

import os
import json
import logging
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import s3fs
import mlflow
from mlflow.tracking import MlflowClient

from config import ServiceConfig

logger = logging.getLogger(__name__)

config = ServiceConfig()

MINIO_ENDPOINT   = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key

# CHANGE: was os.environ.get only — now uses ServiceConfig for consistency
ML_PRED_BUCKET = os.environ.get("ML_PREDICTIONS_BUCKET", "ml-predictions")
MODEL_NAME     = os.environ.get("MODEL_NAME", "secom_yield_predictor")
MODEL_ALIAS    = os.environ.get("MODEL_ALIAS", "champion")

router = APIRouter()


# ─── Input validation via Mahalanobis distance ─────────────────────────────────
class InputValidationRequest(BaseModel):
    features:       dict
    observation_id: Optional[str] = None


class InputValidationResponse(BaseModel):
    observation_id:       Optional[str]
    mahalanobis_distance: float
    is_in_distribution:   bool
    p_value:              float
    warning:              Optional[str]


# Module-level cache for Mahalanobis training stats
_training_stats: dict = {}


def _get_training_stats() -> dict:
    """
    Load mean vector and inverse covariance from the calibration reference dataset.

    CHANGES:
      ✦ Was referencing undefined `SENSOR_COLS` global — now loads feature list
        dynamically from the champion model's manifest in MLflow.
      ✦ Switched from deprecated get_latest_versions(stages=["Production"])
        → get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS) — consistent
        with all other files (batch_inference, drift_monitor, main.py).
      ✦ Reference dataset is the CBPE parquet logged by calibrate_model.py,
        not a separate download via client.download_artifacts (deprecated path).
    """
    global _training_stats
    if _training_stats:
        return _training_stats

    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = MlflowClient()

        # CHANGE: use alias-based lookup, same as every other module
        v   = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run_id = v.run_id

        # CHANGE: load feature list from manifest, not from undefined SENSOR_COLS
        manifest_uri  = f"runs:/{run_id}/metadata/feature_manifest.json"
        local_manifest = mlflow.artifacts.download_artifacts(artifact_uri=manifest_uri)
        with open(local_manifest) as f:
            manifest = json.load(f)

        active_sensors = manifest.get("active_features_list", [])
        cat_features   = manifest.get("categorical_features", [])
        # Only numeric sensors are used for Mahalanobis distance
        feat_cols = [f for f in active_sensors if f not in cat_features]

        # CHANGE: load reference data from calibration artifact (written by calibrate_model.py)
        local_ref = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/calibration/cbpe_reference_dataset.parquet"
        )
        ref_df = pd.read_parquet(local_ref)
        available = [c for c in feat_cols if c in ref_df.columns]
        X_ref = ref_df[available].dropna().values.astype(np.float64)

        mu     = X_ref.mean(axis=0)
        cov    = np.cov(X_ref, rowvar=False)
        cov   += np.eye(len(available)) * 1e-6    # regularise for invertibility
        inv_cov = np.linalg.inv(cov)

        _training_stats = {
            "mu":            mu,
            "inv_cov":       inv_cov,
            "feature_names": available,
            "n_ref":         len(X_ref),
        }
        logger.info(
            "Mahalanobis stats loaded from calibration reference (%d rows, %d features)",
            len(X_ref), len(available)
        )
    except Exception as e:
        logger.warning("Could not load Mahalanobis stats: %s", e)

    return _training_stats


def _mahalanobis_pvalue(dist: float, n_features: int) -> float:
    """
    Under the null (x drawn from training distribution), D² ~ χ²(p).
    P-value = P(χ²(p) > D²). Low p-value → out-of-distribution.
    """
    from scipy.stats import chi2
    return float(1.0 - chi2.cdf(dist ** 2, df=n_features))


@router.post("/validate-input", response_model=InputValidationResponse)
async def validate_input(request: InputValidationRequest):
    """
    Mahalanobis distance gate: checks if the incoming sensor readings are
    within the training distribution before scoring.

    D > threshold (p < 0.001) → out-of-distribution → prediction unreliable.
    """
    stats = _get_training_stats()
    if not stats:
        raise HTTPException(status_code=503,
                            detail="Training distribution stats not loaded — calibration may not have run yet.")

    mu         = stats["mu"]
    inv_cov    = stats["inv_cov"]
    feat_names = stats["feature_names"]

    x    = np.array([request.features.get(f, float(mu[i]))
                     for i, f in enumerate(feat_names)], dtype=np.float64)
    diff = x - mu
    dist = float(np.sqrt(diff @ inv_cov @ diff))
    pval = _mahalanobis_pvalue(dist, len(feat_names))

    in_dist = pval >= 0.001   # 99.9% confidence in-distribution
    warning = None
    if not in_dist:
        warning = (
            f"Input is out-of-distribution (D={dist:.2f}, p={pval:.4f}). "
            "Possible sensor fault or process anomaly — treat prediction with caution."
        )

    return InputValidationResponse(
        observation_id       = request.observation_id,
        mahalanobis_distance = round(dist, 4),
        is_in_distribution   = in_dist,
        p_value              = round(pval, 6),
        warning              = warning,
    )


# ─── Counterfactual endpoint ───────────────────────────────────────────────────
class CounterfactualRequest(BaseModel):
    observation_id:    str
    n_counterfactuals: int = 3
    method:            str = "both"    # "dice", "milp", "both"
    max_changes:       int = 5


@router.post("/counterfactual")
async def counterfactual(request: CounterfactualRequest):
    """
    Generate actionable counterfactual explanations for a predicted-fail wafer.

    Returns the minimum sensor value changes that would flip the prediction
    from Fail → Pass, using DiCE (diverse options) and/or MILP (minimal changes).

    Changes are expressed in:
      - Absolute units (from / to values)
      - Delta sigma units (relative to Phase I process sigma)
    """
    if request.n_counterfactuals > 10:
        raise HTTPException(status_code=400, detail="Max 10 counterfactuals per request")

    try:
        # CHANGE: corrected import path from `ml.inference.counterfactual_engine`
        #         to local module `counterfactual_engine` (same container directory)
        from counterfactual_engine import generate_counterfactuals
        result = generate_counterfactuals(
            observation_id    = request.observation_id,
            n_counterfactuals = request.n_counterfactuals,
            method            = request.method,
            max_changes       = request.max_changes,
        )
        return result

    except Exception as e:
        logger.error("Counterfactual generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Counterfactual error: {e}")


# ─── Performance estimation endpoint ──────────────────────────────────────────
@router.get("/performance")
async def estimated_performance():
    """
    Return the latest CBPE-estimated performance metrics (no ground truth needed).
    Reads from the MinIO Parquet written by performance_estimator.py.
    """
    try:
        # CHANGE: use ServiceConfig values rather than raw os.environ.get
        fs = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url": MINIO_ENDPOINT},
            key=MINIO_ACCESS_KEY,
            secret=MINIO_SECRET_KEY,
        )
        files = sorted(
            fs.glob(f"{ML_PRED_BUCKET}/performance_estimates/**/*.parquet"),
            reverse=True,
        )
        if not files:
            return {
                "estimates_available": False,
                "message": "No estimates yet — run the monitoring DAG first",
            }

        with fs.open(files[0], "rb") as f:
            row = pd.read_parquet(f)

        r = row.iloc[0]
        return {
            "estimates_available": True,
            "estimated_at":        str(r.get("estimated_at")),
            "estimated_auc":       r.get("estimated_auc"),
            "estimated_f1":        r.get("estimated_f1"),
            "interval_width":      r.get("interval_width"),
            "width_inflation":     r.get("width_inflation"),
            "model_age_hours":     r.get("model_age_hours"),
            "model_version":       r.get("model_version"),
            "n_predictions_scored":r.get("n_predictions_scored"),
            "method":              r.get("method"),
            "note": (
                "These metrics are estimated without ground-truth labels (CBPE). "
                "True labels have a delayed arrival in semiconductor manufacturing."
            ),
        }
    except Exception as e:
        logger.warning("Could not load performance estimates: %s", e)
        return {"estimates_available": False, "error": str(e)}