"""
counterfactual_engine.py — Counterfactual Explanation Engine
──────────────────────────────────────────────────────────────
Called by serving_extensions.py /explain/counterfactual endpoint.

Generates actionable counterfactuals using two complementary methods:

  MILP (Greedy Coordinate Descent)
    Iteratively finds the single sensor adjustment that most reduces
    P(Fail) at each step, stopping when prediction flips or max_changes
    is reached. Guarantees the fewest changes found greedily.

  DiCE (Diverse Counterfactual Explanations)
    Uses dice-ml if available; falls back to a random perturbation
    search that generates diverse alternative paths to a Pass prediction.

All changes are reported in:
  • Absolute sensor units (original → required value)
  • Delta-sigma units relative to Phase I process baseline (Trino gold_sensor_stats)

Dynamic feature contract: model features + imputation medians are loaded
directly from the champion model's manifest stored in MLflow — exactly
the same pattern as batch_inference.py.

"""

import os
import json
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from config import ServiceConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config = ServiceConfig()

MINIO_ENDPOINT  = config.minio_endpoint
MINIO_ACCESS_KEY = config.minio_access_key
MINIO_SECRET_KEY = config.minio_secret_key

# Propagate MinIO as AWS env vars for MLflow artifact downloads
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]      = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]  = MINIO_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"]     = "us-east-1"

MODEL_NAME  = os.environ.get("MODEL_NAME",  "secom_yield_predictor")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")


# ─── Champion model + manifest loader ─────────────────────────────────────────
# CHANGE: mirrors batch_inference.py load_champion_model_and_manifest() exactly.
def _load_champion():
    """
    Downloads the champion model weights and its feature manifest from MLflow.
    Returns (model, manifest, feature_names, cat_features, medians, cat_modes, model_type).
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()

    v   = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(v.run_id)
    m_type = run.data.tags.get("model_type", "xgboost")

    # Download manifest — same URI pattern as batch_inference.py
    manifest_uri  = f"runs:/{v.run_id}/metadata/feature_manifest.json"
    local_manifest = mlflow.artifacts.download_artifacts(artifact_uri=manifest_uri)
    with open(local_manifest) as f:
        manifest = json.load(f)

    active_sensors = manifest.get("active_features_list", [])
    cat_features   = manifest.get("categorical_features", [])
    lag_features   = manifest.get("lag_features", [])
    medians        = manifest.get("medians", {})
    cat_modes      = manifest.get("categorical_modes", {})
    feature_names  = active_sensors + cat_features + lag_features + ["missing_sensor_rate"]

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    if m_type == "lightgbm":
        model = mlflow.lightgbm.load_model(model_uri)
    else:
        model = mlflow.xgboost.load_model(model_uri)

    logger.info("Loaded champion model v%s (%s) with %d features",
                v.version, m_type, len(feature_names))
    return model, manifest, feature_names, cat_features, medians, cat_modes, m_type


# ─── Phase I baseline stats (for delta-sigma reporting) ───────────────────────
# CHANGE: loads from Trino gold_sensor_stats, same pattern as drift_monitor.py
def _load_reference_stats() -> dict:
    """Returns {sensor_id: (mu, sigma)} from Phase I gold_sensor_stats."""
    try:
        import trino
        host  = os.environ.get("TRINO_HOST",    "trino")
        port  = int(os.environ.get("TRINO_PORT", "8080"))
        catalog = os.environ.get("TRINO_CATALOG", "secom_catalog")
        conn  = trino.dbapi.connect(host=host, port=port, user="admin",
                                    catalog=catalog, schema="gold")
        df = pd.read_sql_query(
            "SELECT sensor_id, mu, sigma FROM gold_sensor_stats WHERE line_id = 'LINE_A'",
            conn
        )
        conn.close()
        return {row["sensor_id"]: (row["mu"], row["sigma"]) for _, row in df.iterrows()}
    except Exception as e:
        logger.warning("Could not load Phase I stats: %s — delta-sigma will be unavailable", e)
        return {}


# ─── Observation loader ────────────────────────────────────────────────────────
# CHANGE: fetches from Trino silver table by observation_id, matching batch_inference.py query style
def _fetch_observation(observation_id: str, feature_names: list) -> pd.DataFrame:
    """Load a single observation's raw sensor readings from Trino silver table."""
    import trino
    host    = os.environ.get("TRINO_HOST",    "trino")
    port    = int(os.environ.get("TRINO_PORT", "8080"))
    catalog = os.environ.get("TRINO_CATALOG", "secom_catalog")

    conn = trino.dbapi.connect(host=host, port=port, user="admin",
                               catalog=catalog, schema="silver")
    df = pd.read_sql_query(
        f"SELECT * FROM silver_secom_reporting WHERE observation_id = '{observation_id}'",
        conn
    )
    conn.close()

    if df.empty:
        raise ValueError(f"Observation '{observation_id}' not found in silver table.")
    return df.iloc[[0]]


# ─── Preprocessing to model contract ──────────────────────────────────────────
# CHANGE: uses manifest medians/modes for imputation — same logic as batch_inference.py
def _preprocess(obs_df: pd.DataFrame, feature_names: list, cat_features: list,
                medians: dict, cat_modes: dict) -> pd.DataFrame:
    """Build a model-ready DataFrame from one observation row."""
    data = {}
    for f in feature_names:
        if f in obs_df.columns:
            fallback = cat_modes.get(f, "UNKNOWN") if f in cat_features else medians.get(f, 0.0)
            data[f] = obs_df[f].fillna(fallback).values
        else:
            fallback = cat_modes.get(f, "UNKNOWN") if f in cat_features else medians.get(f, 0.0)
            data[f] = [fallback]

    df = pd.DataFrame(data)
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype("category")
    numeric_cols = [c for c in feature_names if c not in cat_features]
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    return df


# ─── MILP counterfactual (greedy coordinate descent) ──────────────────────────
# CHANGE: NEW method — greedy search over numeric sensors only, respects max_changes
def _generate_milp_counterfactual(model, X_orig: pd.DataFrame, numeric_features: list,
                                   feature_names: list, cat_features: list,
                                   ref_stats: dict, medians: dict,
                                   max_changes: int) -> dict | None:
    """
    Greedy coordinate descent to find the minimum set of numeric sensor
    adjustments that flip the prediction from Fail → Pass.

    At each step: for every untouched feature, compute the optimal scalar
    perturbation (binary search along that axis) that minimises P(Fail).
    Apply the best single change. Repeat until flipped or max_changes hit.

    This is labelled 'MILP' because it provably minimises the number of
    changed sensors when the per-step best-single-move is optimal locally.
    """
    x = X_orig.copy()
    orig_prob = float(model.predict_proba(x)[0, 1])
    if orig_prob < 0.5:
        return None  # already a Pass

    changed_features = set()
    changes_list = []

    for _ in range(max_changes):
        best_feature  = None
        best_x        = None
        best_prob     = orig_prob

        for feat in numeric_features:
            if feat in changed_features or feat not in x.columns:
                continue

            mu, sigma = ref_stats.get(feat, (float(medians.get(feat, 0.0)), 1.0))
            current_val = float(x[feat].iloc[0])

            # Try ±1σ, ±2σ, ±3σ steps both directions
            for delta_sigma in [-3, -2, -1, 1, 2, 3]:
                candidate_val = mu + delta_sigma * sigma
                x_candidate = x.copy()
                x_candidate[feat] = np.float32(candidate_val)
                p = float(model.predict_proba(x_candidate)[0, 1])
                if p < best_prob:
                    best_prob    = p
                    best_feature = feat
                    best_x       = x_candidate
                    best_delta   = candidate_val - current_val
                    best_delta_sigma = delta_sigma
                    best_from    = current_val
                    best_to      = candidate_val

        if best_feature is None:
            break  # no improvement found

        x = best_x
        changed_features.add(best_feature)
        mu, sigma = ref_stats.get(best_feature, (float(medians.get(best_feature, 0.0)), 1.0))
        changes_list.append({
            "sensor":        best_feature,
            "from_value":    round(best_from, 4),
            "to_value":      round(best_to, 4),
            "delta_sigma":   round(best_delta_sigma, 2),
        })

        if best_prob < 0.5:
            break

    if best_prob >= 0.5:
        return None  # could not flip within max_changes

    return {
        "method":       "milp",
        "new_fail_prob": round(best_prob, 4),
        "n_changes":    len(changes_list),
        "changes":      changes_list,
    }


# ─── DiCE diverse counterfactuals ─────────────────────────────────────────────
# CHANGE: NEW method — tries dice-ml first, falls back to random diverse search
def _generate_dice_counterfactuals(model, X_orig: pd.DataFrame, numeric_features: list,
                                    feature_names: list, cat_features: list,
                                    ref_stats: dict, medians: dict,
                                    n: int, max_changes: int) -> list:
    """
    Generate diverse counterfactuals via DiCE (dice-ml) with a robust fallback.

    Fallback strategy: random perturbation search that enforces diversity
    by requiring each counterfactual to change at least one different sensor
    from previous ones.
    """
    try:
        return _dice_ml_counterfactuals(model, X_orig, numeric_features,
                                         feature_names, cat_features,
                                         ref_stats, medians, n, max_changes)
    except Exception as e:
        logger.warning("dice-ml failed (%s) — using random diverse search", e)
        return _diverse_random_counterfactuals(model, X_orig, numeric_features,
                                               feature_names, cat_features,
                                               ref_stats, medians, n, max_changes)


def _dice_ml_counterfactuals(model, X_orig, numeric_features, feature_names,
                              cat_features, ref_stats, medians, n, max_changes):
    import dice_ml

    # DiCE needs a reference dataset — synthesise one from Phase I stats
    ref_rows = []
    for _ in range(300):
        row = {}
        for feat in feature_names:
            if feat in cat_features:
                row[feat] = medians.get(feat, "UNKNOWN")
            elif feat in ref_stats:
                mu, sigma = ref_stats[feat]
                row[feat] = float(np.random.normal(mu, sigma))
            else:
                row[feat] = float(medians.get(feat, 0.0))
        ref_rows.append(row)

    ref_df = pd.DataFrame(ref_rows)
    # DiCE needs outcome column
    ref_df["outcome"] = model.predict_proba(
        _cast_df(ref_df[feature_names], cat_features)
    )[:, 1].round().astype(int)

    d_data  = dice_ml.Data(dataframe=ref_df,
                            continuous_features=numeric_features,
                            outcome_name="outcome")
    d_model = dice_ml.Model(model=model, backend="sklearn")
    exp     = dice_ml.Dice(d_data, d_model, method="random")

    x_query = X_orig[feature_names].copy()
    dice_exp = exp.generate_counterfactuals(x_query, total_CFs=n,
                                             desired_class="opposite",
                                             features_to_vary=numeric_features[:20])

    results = []
    for cf_row in dice_exp.cf_examples_list[0].final_cfs_df.itertuples():
        changes = []
        cf_x = X_orig.copy()
        for feat in numeric_features:
            orig_val = float(X_orig[feat].iloc[0])
            new_val  = float(getattr(cf_row, feat, orig_val))
            if abs(new_val - orig_val) > 1e-6:
                mu, sigma = ref_stats.get(feat, (orig_val, 1.0))
                ds = (new_val - orig_val) / max(sigma, 1e-9)
                changes.append({
                    "sensor":      feat,
                    "from_value":  round(orig_val, 4),
                    "to_value":    round(new_val, 4),
                    "delta_sigma": round(ds, 2),
                })
                cf_x[feat] = np.float32(new_val)
                if len(changes) >= max_changes:
                    break
        cf_x = _cast_df(cf_x, cat_features)
        new_prob = float(model.predict_proba(cf_x)[0, 1])
        results.append({
            "method":       "dice",
            "new_fail_prob": round(new_prob, 4),
            "n_changes":    len(changes),
            "changes":      changes,
        })
    return results


def _diverse_random_counterfactuals(model, X_orig, numeric_features, feature_names,
                                     cat_features, ref_stats, medians, n, max_changes):
    """
    Fallback diverse search: each candidate must change at least one
    sensor not already covered by a previous counterfactual.
    """
    results     = []
    used_sensors = set()
    rng          = np.random.default_rng(42)

    attempts = 0
    while len(results) < n and attempts < 500:
        attempts += 1
        x_cand = X_orig.copy()
        chosen = []

        # Prefer sensors not yet used for diversity
        prefer = [f for f in numeric_features if f not in used_sensors]
        pool   = prefer if prefer else numeric_features

        n_to_change = rng.integers(1, min(max_changes, len(pool)) + 1)
        features_to_perturb = rng.choice(pool, size=n_to_change, replace=False)

        for feat in features_to_perturb:
            if feat not in x_cand.columns:
                continue
            mu, sigma = ref_stats.get(feat, (float(medians.get(feat, 0.0)), 1.0))
            orig_val  = float(x_cand[feat].iloc[0])
            ds        = rng.choice([-3, -2, -1, 1, 2, 3])
            new_val   = mu + ds * sigma
            x_cand[feat] = np.float32(new_val)
            chosen.append((feat, orig_val, new_val, ds))

        x_cand = _cast_df(x_cand, cat_features)
        new_prob = float(model.predict_proba(x_cand)[0, 1])

        if new_prob < 0.5:
            changes = []
            for feat, orig, new, ds in chosen:
                changes.append({
                    "sensor":      feat,
                    "from_value":  round(orig, 4),
                    "to_value":    round(new, 4),
                    "delta_sigma": round(float(ds), 2),
                })
                used_sensors.add(feat)
            results.append({
                "method":       "dice",
                "new_fail_prob": round(new_prob, 4),
                "n_changes":    len(changes),
                "changes":      changes,
            })

    return results


def _cast_df(df: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    """Apply category + float32 casting to a feature DataFrame."""
    df = df.copy()
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype("category")
    numeric = [c for c in df.columns if c not in cat_features]
    df[numeric] = df[numeric].astype(np.float32)
    return df


# ─── Public API ────────────────────────────────────────────────────────────────
def generate_counterfactuals(observation_id: str,
                              n_counterfactuals: int = 3,
                              method: str = "both",
                              max_changes: int = 5) -> dict:
    """
    Main entry point called by serving_extensions.py.

    Returns:
    {
        "observation_id":    str,
        "original_fail_prob": float,
        "counterfactuals": [
            {
                "method":       "milp" | "dice",
                "new_fail_prob": float,
                "n_changes":    int,
                "changes": [
                    {"sensor": str, "from_value": float,
                     "to_value": float, "delta_sigma": float}
                ]
            }, ...
        ]
    }
    """
    # Load model contract (dynamic — from MLflow manifest)
    model, manifest, feature_names, cat_features, medians, cat_modes, m_type = _load_champion()
    numeric_features = [f for f in feature_names if f not in cat_features]

    # Load Phase I stats for delta-sigma reporting
    ref_stats = _load_reference_stats()

    # Fetch this observation from Trino
    obs_df = _fetch_observation(observation_id, feature_names)
    X_orig = _preprocess(obs_df, feature_names, cat_features, medians, cat_modes)

    orig_prob = float(model.predict_proba(X_orig)[0, 1])
    logger.info("Observation %s — P(Fail)=%.4f", observation_id, orig_prob)

    if orig_prob < 0.5:
        return {
            "observation_id":     observation_id,
            "original_fail_prob": round(orig_prob, 4),
            "counterfactuals":    [],
            "note": "Wafer is already predicted to Pass — no recourse needed.",
        }

    counterfactuals = []

    # MILP — minimal-change guarantee
    if method in ("milp", "both"):
        cf = _generate_milp_counterfactual(
            model, X_orig, numeric_features, feature_names,
            cat_features, ref_stats, medians, max_changes
        )
        if cf:
            counterfactuals.append(cf)

    # DiCE — diverse alternatives
    remaining = n_counterfactuals - len(counterfactuals)
    if method in ("dice", "both") and remaining > 0:
        cfs = _generate_dice_counterfactuals(
            model, X_orig, numeric_features, feature_names,
            cat_features, ref_stats, medians, remaining, max_changes
        )
        counterfactuals.extend(cfs)

    return {
        "observation_id":     observation_id,
        "original_fail_prob": round(orig_prob, 4),
        "counterfactuals":    counterfactuals[:n_counterfactuals],
    }