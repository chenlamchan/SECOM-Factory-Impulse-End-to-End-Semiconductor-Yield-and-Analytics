"""
main.py — FastAPI ML Inference Server
───────────────────────────────────────
Production inference service for the SECOM yield prediction model.
Loads the Champion-aliased model from the MLflow registry on startup
and exposes:

  GET  /health            — liveness probe (used by Docker healthcheck)
  GET  /model-info        — current model name, version, alias, metrics
  POST /predict           — single wafer prediction + top-5 SHAP drivers
  POST /predict/batch     — batch prediction (list of wafers)
  POST /reload            — hot-reload Champion model without restart
  GET  /drift-report      — latest Evidently report URL from MinIO

Model is cached in module-level state and reloaded atomically on /reload,
so in-flight requests are never interrupted.
"""

import os
import logging
import json
import threading
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import shap
import s3fs
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from config import ServiceConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config = ServiceConfig()

MINIO_ENDPOINT = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key

# ─── Config from environment ─────────────────────────────────────────────────
MLFLOW_URI       = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME       = os.environ.get("MODEL_NAME",  "secom_yield_predictor")
MODEL_ALIAS      = os.environ.get("MODEL_ALIAS", "champion")
EVIDENTLY_BUCKET = os.environ.get("EVIDENTLY_BUCKET", "evidently-reports")

# 🟢 CRITICAL: Map custom MinIO vars to standard AWS vars for MLflow artifact downloading
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


# ─── Module-level model state (replaced atomically on /reload) ───────────────
class _ModelState:
    def __init__(self):
        self.model          = None
        self.model_type     = None
        self.explainer      = None
        self.feature_names  = []
        self.cat_features   = []
        self.cat_modes      = {}
        self.medians        = {}
        self.version        = None
        self.run_id         = None
        self.metrics        = {}
        self._lock          = threading.Lock()

    def load(self) -> None:
        """Load the Champion model and its exact manifest from MLflow."""
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()

        # 1. Identify the Champion Model
        try:
            v = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        except Exception as e:
            raise RuntimeError(
                f"No model found with alias '{MODEL_ALIAS}' for '{MODEL_NAME}'. "
                "Complete the training pipeline first to assign the champion alias."
            )
            
        run     = client.get_run(v.run_id)
        m_type  = run.data.tags.get("model_type", "xgboost")

        logger.info("Identified %s model v%s (run_id: %s) as %s", m_type, v.version, v.run_id, MODEL_ALIAS)

        # 2. Download the EXACT Manifest Artifact tied to this model version
        try:
            manifest_uri = f"runs:/{v.run_id}/metadata/feature_manifest.json"
            logger.info("Downloading manifest from MLflow artifact store: %s", manifest_uri)
            
            local_manifest_path = mlflow.artifacts.download_artifacts(
                artifact_uri=manifest_uri,
                tracking_uri=MLFLOW_URI
            )
            
            with open(local_manifest_path, "r") as f:
                manifest = json.load(f)
                
            active_sensors = manifest.get("active_features_list", [])
            cat_features   = manifest.get("categorical_features", [])
            lag_features   = manifest.get("lag_features", [])
            cat_modes      = manifest.get("categorical_modes", {})
            medians        = manifest.get("medians", {})
            
            feature_names  = active_sensors + cat_features + lag_features + ["missing_sensor_rate"]
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch feature_manifest.json for run {v.run_id}: {e}")

        # 3. Download the Model Weights
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        logger.info("Loading model weights from %s...", model_uri)

        if m_type == "lightgbm":
            model = mlflow.lightgbm.load_model(model_uri)
            # explainer = shap.TreeExplainer(model)
        else:
            model = mlflow.xgboost.load_model(model_uri)
            
            # Apply SHAP base_score patch for XGBoost >= 3.2
            booster = model.get_booster()
            config_dict = json.loads(booster.save_config())
            base_score_str = config_dict.get("learner", {}).get("learner_model_param", {}).get("base_score", "")
            
            if isinstance(base_score_str, str) and base_score_str.startswith('[') and base_score_str.endswith(']'):
                clean_score = base_score_str.strip('[]')
                config_dict["learner"]["learner_model_param"]["base_score"] = clean_score
                booster.load_config(json.dumps(config_dict))
                
            # explainer = shap.TreeExplainer(model)

        metrics = {k: round(v2, 5) for k, v2 in run.data.metrics.items()
                   if k.startswith("held_out_")}

        # 4. Safely swap state atomically
        with self._lock:
            self.model         = model
            self.model_type    = m_type
            # self.explainer     = explainer
            self.feature_names = feature_names
            self.cat_features  = cat_features
            self.cat_modes     = cat_modes
            self.medians       = medians
            self.version       = v.version
            self.run_id        = v.run_id
            self.metrics       = metrics

        logger.info("Model v%s successfully loaded (type=%s, features=%d).",
                    v.version, m_type, len(feature_names))


_state = _ModelState()


# ─── FastAPI lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state.load()
    except Exception as e:
        logger.error("Model load failed at startup: %s", e)
    yield


app = FastAPI(
    title="SECOM Yield Prediction API",
    version="1.0.0",
    description="Real-time wafer yield prediction with SHAP explanations.",
    lifespan=lifespan,
)


# ─── Request / Response schemas ───────────────────────────────────────────────
class WaferFeatures(BaseModel):
    """
    Sensor readings for a single wafer. Unobserved sensors can be omitted.
    Because we now have dynamic features (up to 590), we allow any arbitrary kwargs.
    """
    observation_id:       Optional[str]   = None
    missing_sensor_rate:  Optional[float] = Field(default=0.0, ge=0, le=1)
    
    # Allows any extra fields (e.g. "59": 1.2, "tester_id": "T_01")
    model_config = {"populate_by_name": True, "extra": "allow"}


class PredictionResponse(BaseModel):
    observation_id:    Optional[str]
    defect_probability: float
    yield_probability:  float
    prediction:         str   # "Pass" or "Fail"
    confidence_band:    str   # "High", "Medium", "Low"
    # top_drivers:        list  # [{feature, shap_value, direction}, ...]
    model_version:      str
    model_name:         str


def _batch_features_to_df(batch_features: list[dict]) -> pd.DataFrame:
    """
    Converts a list of request dicts into a Pandas DataFrame aligned with the model's 
    expected features. Injects manifest medians/modes for missing data.
    """
    feature_names = _state.feature_names
    data = {f: [] for f in feature_names}
    
    for features in batch_features:
        # Retrieve extra kwargs parsed by Pydantic
        extra_data = features.get("__pydantic_extra__") or {}
        merged_features = {**features, **extra_data}
        
        for f in feature_names:
            val = merged_features.get(f)
            if f in _state.cat_features:
                data[f].append(val if val is not None else _state.cat_modes.get(f, "UNKNOWN"))
            else:
                data[f].append(val if val is not None else _state.medians.get(f, 0.0))
                
    df = pd.DataFrame(data)
    
    # Cast Data Types
    for c in _state.cat_features:
        if c in df.columns:
            df[c] = df[c].astype("category")
            
    numeric_cols = [c for c in feature_names if c not in _state.cat_features]
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    
    return df


def _confidence_band(prob: float) -> str:
    if prob >= 0.8 or prob <= 0.2:
        return "High"
    if prob >= 0.65 or prob <= 0.35:
        return "Medium"
    return "Low"


# def _top_shap_drivers(shap_vals: np.ndarray, feature_names: list, n: int = 5) -> list:
#     """Return top-N features by |SHAP value| with direction."""
#     pairs = sorted(
#         zip(feature_names, shap_vals.tolist()),
#         key=lambda x: abs(x[1]), reverse=True
#     )[:n]
#     return [
#         {
#             "feature":    f"Sensor {name}" if name.isdigit() else name,
#             "shap_value": round(val, 6),
#             "direction":  "increases defect risk" if val > 0 else "reduces defect risk",
#         }
#         for name, val in pairs
#     ]


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "status":        "ok",
        "model_name":    MODEL_NAME,
        "model_version": _state.version,
        "model_alias":   MODEL_ALIAS,
    }


@app.get("/model-info")
async def model_info():
    if _state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_name":    MODEL_NAME,
        "model_version": _state.version,
        "model_alias":   MODEL_ALIAS,
        "model_type":    _state.model_type,
        "run_id":        _state.run_id,
        "n_features":    len(_state.feature_names),
        "test_metrics":  _state.metrics,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: WaferFeatures):
    if _state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    feat_dict = features.model_dump(by_alias=True, exclude_none=False)
    df = _batch_features_to_df([feat_dict])

    try:
        prob    = float(_state.model.predict_proba(df)[0, 1])
        sv      = _state.explainer.shap_values(df)
        if isinstance(sv, list):
            sv = sv[1]   # LightGBM positive class
        shap_vals = sv[0]
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictionResponse(
        observation_id    = features.observation_id,
        defect_probability= round(prob, 6),
        yield_probability = round(1.0 - prob, 6),
        prediction        = "Fail" if prob >= 0.5 else "Pass",
        confidence_band   = _confidence_band(prob),
        top_drivers       = _top_shap_drivers(shap_vals, _state.feature_names),
        model_version     = str(_state.version),
        model_name        = MODEL_NAME,
    )


@app.post("/predict/batch")
async def predict_batch(batch: list[WaferFeatures]):
    if _state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(batch) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1000 wafers.")

    feat_dicts = [f.model_dump(by_alias=True, exclude_none=False) for f in batch]
    df = _batch_features_to_df(feat_dicts)
    
    probs = _state.model.predict_proba(df)[:, 1]
    
    results = []
    for i, features in enumerate(batch):
        prob = float(probs[i])
        results.append({
            "observation_id":    features.observation_id,
            "defect_probability": round(prob, 6),
            "yield_probability":  round(1.0 - prob, 6),
            "prediction":         "Fail" if prob >= 0.5 else "Pass",
        })

    return {"count": len(results), "predictions": results}


@app.post("/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Hot-reload the Champion model from MLflow without restarting the container."""
    def _do_reload():
        try:
            _state.load()
            logger.info("Model reloaded successfully — v%s", _state.version)
        except Exception as e:
            logger.error("Model reload failed: %s", e)

    background_tasks.add_task(_do_reload)
    return {"status": "reload_initiated", "current_version": _state.version}


@app.get("/drift-report")
async def drift_report():
    """Return the URL and metadata of the latest Evidently drift report in MinIO."""
    try:
        fs = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url": MINIO_ENDPOINT},
            key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY
        )
        files = sorted(fs.glob(f"{EVIDENTLY_BUCKET}/drift_*.html"), reverse=True)
        if not files:
            return {"report_available": False}

        latest = files[0]
        return {
            "report_available": True,
            "latest_report":    f"s3://{latest}",
            "report_count":     len(files),
        }
    except Exception as e:
        logger.warning("Could not list drift reports: %s", e)
        return {"report_available": False, "error": str(e)}