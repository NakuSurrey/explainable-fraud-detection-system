"""
Phase 8: Inference API (THE MESSENGER)
========================================
FastAPI microservice that wraps the trained fraud detection model,
SHAP explainer, and LIME explainer into a production-ready REST API.

Endpoints:
    GET  /health              - Health check and model status
    GET  /model/info          - Model metadata (metrics, features, comparison)
    GET  /features            - List of expected feature names
    GET  /sample              - Sample transaction template for testing
    POST /predict             - Single transaction prediction with SHAP explanation
    POST /predict/batch       - Batch predictions (up to 100 transactions)
    POST /predict/lime        - Single prediction with LIME explanation (slower)

Read -> Process -> Save Artifact Principle:
    READS:  models/best_model.pkl, models/shap_explainer.pkl,
            models/lime_explainer.pkl (via dill), models/scaler.pkl,
            data/processed/feature_names.json, models/metrics.json,
            models/model_comparison.json
    SERVES: JSON responses with risk scores, SHAP values, explanations

Why it's independent:
    This API has zero knowledge of the dashboard UI (Phase 9).
    Any system -- Streamlit, mobile app, curl, Postman -- can call it.
    If the UI crashes, this API keeps serving predictions.

Usage:
    python -m src.api.inference_api          # Start the API server
    uvicorn src.api.inference_api:app --reload  # Dev mode with auto-reload
"""

import os
import sys
import json
import pickle
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup -- ensure project root is importable
# ---------------------------------------------------------------------------
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logger import (
    get_logger,
    load_config,
    resolve_path,
    log_phase_start,
    log_phase_end,
)

logger = get_logger("Phase8.InferenceAPI")
config = load_config()

# ---------------------------------------------------------------------------
# Lazy imports -- FastAPI and uvicorn
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as e:
    logger.error(f"Missing dependency: {e}. Run: pip install fastapi uvicorn")
    raise

try:
    import dill
except ImportError as e:
    logger.error(f"Missing dependency: {e}. Run: pip install dill")
    raise


# ============================================================
# 1. PYDANTIC MODELS (Request / Response Schemas)
# ============================================================

class TransactionInput(BaseModel):
    """
    A single transaction to score.

    Accepts a dictionary of feature_name -> value pairs.
    All 37 features from the trained model must be provided.

    Example:
        {
            "features": {
                "Time": 406.0,
                "V1": -1.359807,
                "V2": -0.072781,
                ...
                "Amount": 149.62,
                "amount_log": 5.01,
                ...
            }
        }
    """
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature_name -> float value. All 37 model features required."
    )


class BatchTransactionInput(BaseModel):
    """Batch of transactions (max 100)."""
    transactions: List[TransactionInput] = Field(
        ...,
        description="List of transactions to score (max 100 per batch)."
    )


class ShapExplanation(BaseModel):
    """SHAP-based explanation for a single prediction."""
    feature_name: str
    shap_value: float
    feature_value: float
    direction: str  # "increases_risk" or "decreases_risk"


class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    transaction_id: str
    fraud_probability: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    risk_score: int  # 0-100 integer score
    is_flagged: bool
    threshold_used: float
    prediction_label: str  # "LEGITIMATE" or "FRAUDULENT"
    shap_explanation: List[ShapExplanation]
    plain_english_summary: str
    inference_time_ms: float
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    total_transactions: int
    flagged_count: int
    predictions: List[PredictionResponse]
    batch_inference_time_ms: float


class LimeExplanation(BaseModel):
    """Single feature contribution from LIME."""
    feature_rule: str
    weight: float
    direction: str


class LimePredictionResponse(BaseModel):
    """Response for a LIME-explained prediction."""
    transaction_id: str
    fraud_probability: float
    risk_level: str
    risk_score: int
    is_flagged: bool
    lime_explanation: List[LimeExplanation]
    plain_english_summary: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    shap_loaded: bool
    lime_loaded: bool
    scaler_loaded: bool
    feature_count: int
    uptime_seconds: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_type: str
    feature_count: int
    feature_names: List[str]
    metrics: Dict[str, Any]
    model_comparison: Dict[str, Any]
    config: Dict[str, Any]


# ============================================================
# 2. MODEL LOADER (loads all artifacts at startup)
# ============================================================

class ModelService:
    """
    Loads and manages all ML artifacts.

    Centralizes all model/explainer loading so the API endpoints
    stay clean and focused on request/response logic.
    """

    def __init__(self):
        self.model = None
        self.shap_explainer = None
        self.lime_explainer = None
        self.scaler = None
        self.feature_names = None
        self.metrics = None
        self.model_comparison = None
        self.model_type = "unknown"
        self.model_version = "v1"
        self.startup_time = datetime.now()
        self.prediction_count = 0
        self._loaded = False

    def load_all(self):
        """Load all artifacts from disk. Called once at API startup."""
        api_cfg = config.get("api", {})
        logger.info("=" * 60)
        logger.info("Loading ML artifacts for inference...")
        logger.info("=" * 60)

        # --- 1. Load trained model ---
        model_path = resolve_path(api_cfg.get("model_path", "models/best_model.pkl"))
        logger.info(f"  [1/6] Loading model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.model_type = type(self.model).__name__
        logger.info(f"         Model type: {self.model_type}")

        # --- 2. Load SHAP explainer (standard pickle) ---
        shap_path = resolve_path(
            api_cfg.get("explainer_path", "models/shap_explainer.pkl")
        )
        logger.info(f"  [2/6] Loading SHAP explainer from: {shap_path}")
        if not shap_path.exists():
            logger.warning(f"         SHAP explainer not found: {shap_path}")
            self.shap_explainer = None
        else:
            with open(shap_path, "rb") as f:
                self.shap_explainer = pickle.load(f)
            logger.info("         SHAP explainer loaded.")

        # --- 3. Load LIME explainer (uses dill, NOT pickle) ---
        lime_path = resolve_path(
            config.get("explainability", {}).get(
                "lime_explainer_path", "models/lime_explainer.pkl"
            )
        )
        logger.info(f"  [3/6] Loading LIME explainer from: {lime_path}")
        if not lime_path.exists():
            logger.warning(f"         LIME explainer not found: {lime_path}")
            self.lime_explainer = None
        else:
            with open(lime_path, "rb") as f:
                self.lime_explainer = dill.load(f)
            logger.info("         LIME explainer loaded (via dill).")

        # --- 4. Load scaler ---
        scaler_path = resolve_path(
            api_cfg.get("scaler_path", "models/scaler.pkl")
        )
        logger.info(f"  [4/6] Loading scaler from: {scaler_path}")
        if not scaler_path.exists():
            logger.warning(f"         Scaler not found: {scaler_path}")
            self.scaler = None
        else:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info(f"         Scaler type: {type(self.scaler).__name__}")

        # --- 5. Load feature names ---
        feature_path = resolve_path("data/processed/feature_names.json")
        logger.info(f"  [5/6] Loading feature names from: {feature_path}")
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature names not found: {feature_path}")
        with open(feature_path, "r") as f:
            fn_data = json.load(f)
        # Handle dict format: {"features": [...], "target": "Class", ...}
        if isinstance(fn_data, dict):
            for key in ("features", "feature_names", "columns", "names"):
                if key in fn_data and isinstance(fn_data[key], list):
                    self.feature_names = fn_data[key]
                    break
            if self.feature_names is None:
                raise ValueError(
                    f"feature_names.json is a dict but no list found under "
                    f"known keys. Keys present: {list(fn_data.keys())}"
                )
        elif isinstance(fn_data, list):
            self.feature_names = fn_data
        else:
            raise ValueError(f"Unexpected feature_names.json format: {type(fn_data)}")
        logger.info(f"         Loaded {len(self.feature_names)} feature names.")

        # --- 6. Load metrics and model comparison ---
        metrics_path = resolve_path(
            config.get("model", {}).get("metrics_path", "models/metrics.json")
        )
        logger.info(f"  [6/6] Loading metrics from: {metrics_path}")
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
            logger.info("         Metrics loaded.")
        else:
            self.metrics = {}
            logger.warning("         Metrics file not found.")

        comparison_path = resolve_path(
            config.get("model", {}).get("comparison_path", "models/model_comparison.json")
        )
        if comparison_path.exists():
            with open(comparison_path, "r") as f:
                self.model_comparison = json.load(f)
        else:
            self.model_comparison = {}

        self._loaded = True
        logger.info("=" * 60)
        logger.info("All ML artifacts loaded successfully.")
        logger.info("=" * 60)

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def get_uptime(self) -> float:
        return (datetime.now() - self.startup_time).total_seconds()


# ============================================================
# 3. PREDICTION ENGINE
# ============================================================

def classify_risk(probability: float) -> tuple:
    """
    Convert raw fraud probability into human-friendly risk level.

    Returns:
        (risk_level, risk_score, is_flagged)
    """
    risk_score = int(round(probability * 100))

    if probability >= 0.80:
        return "CRITICAL", risk_score, True
    elif probability >= 0.50:
        return "HIGH", risk_score, True
    elif probability >= 0.20:
        return "MEDIUM", risk_score, True
    else:
        return "LOW", risk_score, False


def generate_plain_english(
    shap_explanations: List[dict],
    risk_level: str,
    fraud_probability: float,
) -> str:
    """
    Translate SHAP values into a plain-English explanation
    suitable for a non-technical fraud investigator.
    """
    if not shap_explanations:
        return f"Risk Level: {risk_level} (probability: {fraud_probability:.1%})."

    # Get top 3 risk-increasing factors
    risk_factors = [
        e for e in shap_explanations if e["direction"] == "increases_risk"
    ][:3]

    # Get top 2 risk-decreasing factors
    safe_factors = [
        e for e in shap_explanations if e["direction"] == "decreases_risk"
    ][:2]

    parts = []

    if risk_level in ("CRITICAL", "HIGH"):
        parts.append(
            f"ALERT: This transaction has a {fraud_probability:.1%} probability "
            f"of being fraudulent ({risk_level} risk)."
        )
    elif risk_level == "MEDIUM":
        parts.append(
            f"REVIEW: This transaction has a {fraud_probability:.1%} probability "
            f"of being fraudulent ({risk_level} risk)."
        )
    else:
        parts.append(
            f"This transaction appears legitimate with a {fraud_probability:.1%} "
            f"fraud probability ({risk_level} risk)."
        )

    if risk_factors:
        factors_text = []
        for rf in risk_factors:
            name = rf["feature_name"]
            val = rf["feature_value"]
            sv = abs(rf["shap_value"])
            factors_text.append(
                f"{name} = {val:.4f} (+{sv:.4f} risk contribution)"
            )
        parts.append(
            "Key risk drivers: " + "; ".join(factors_text) + "."
        )

    if safe_factors:
        safe_text = []
        for sf in safe_factors:
            name = sf["feature_name"]
            val = sf["feature_value"]
            sv = abs(sf["shap_value"])
            safe_text.append(
                f"{name} = {val:.4f} (-{sv:.4f} risk reduction)"
            )
        parts.append(
            "Mitigating factors: " + "; ".join(safe_text) + "."
        )

    return " ".join(parts)


def predict_single(
    service: ModelService,
    features_dict: Dict[str, float],
    transaction_id: str = "txn_001",
) -> dict:
    """
    Score a single transaction and generate SHAP explanation.

    Args:
        service: Loaded ModelService instance
        features_dict: {feature_name: value} dictionary
        transaction_id: Identifier for this transaction

    Returns:
        Dictionary matching PredictionResponse schema
    """
    start_time = time.time()

    # --- Validate features ---
    missing = [f for f in service.feature_names if f not in features_dict]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} feature(s): {missing[:5]}... "
            f"Expected {len(service.feature_names)} features."
        )

    # --- Build input array in correct column order ---
    values = [features_dict[f] for f in service.feature_names]
    X_input = np.array(values).reshape(1, -1)

    # --- Get fraud probability ---
    fraud_prob = float(service.model.predict_proba(X_input)[0, 1])
    risk_level, risk_score, is_flagged = classify_risk(fraud_prob)

    # --- Determine threshold ---
    threshold = 0.50  # default
    if service.metrics:
        # Try to get optimal threshold from metrics
        for model_key in ("xgboost", "lightgbm", "best"):
            m = service.metrics.get(model_key, {})
            if "optimal_threshold" in m:
                # threshold = m["optimal_threshold"]
                threshold = float(m["optimal_threshold"]["threshold"])
                break

    prediction_label = "FRAUDULENT" if fraud_prob >= threshold else "LEGITIMATE"

    # --- Compute SHAP values ---
    shap_explanations = []
    if service.shap_explainer is not None:
        try:
            shap_values = service.shap_explainer.shap_values(X_input)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classifier: shap_values[1] = fraud class
                sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    sv = shap_values[0, :, 1]  # (samples, features, classes)
                elif shap_values.ndim == 2:
                    sv = shap_values[0]
                else:
                    sv = shap_values
            else:
                sv = np.zeros(len(service.feature_names))

            # Build explanation list sorted by absolute importance
            explanations = []
            for i, fname in enumerate(service.feature_names):
                explanations.append({
                    "feature_name": fname,
                    "shap_value": float(sv[i]),
                    "feature_value": float(values[i]),
                    "direction": "increases_risk" if sv[i] > 0 else "decreases_risk",
                })
            # Sort by absolute SHAP value (most important first)
            explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            # Keep top N features (from config)
            max_display = config.get("explainability", {}).get("max_display_features", 15)
            shap_explanations = explanations[:max_display]

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            shap_explanations = []

    # --- Generate plain-English summary ---
    plain_english = generate_plain_english(shap_explanations, risk_level, fraud_prob)

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "transaction_id": transaction_id,
        "fraud_probability": round(fraud_prob, 6),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "is_flagged": is_flagged,
        "threshold_used": threshold,
        "prediction_label": prediction_label,
        "shap_explanation": shap_explanations,
        "plain_english_summary": plain_english,
        "inference_time_ms": round(elapsed_ms, 2),
        "model_version": service.model_version,
        "timestamp": datetime.now().isoformat(),
    }


def predict_lime(
    service: ModelService,
    features_dict: Dict[str, float],
    transaction_id: str = "txn_001",
) -> dict:
    """
    Score a single transaction and generate LIME explanation.

    LIME is slower than SHAP but provides rule-based explanations
    that are easier for auditors to understand.
    """
    start_time = time.time()

    if service.lime_explainer is None:
        raise RuntimeError("LIME explainer is not loaded.")

    # --- Validate features ---
    missing = [f for f in service.feature_names if f not in features_dict]
    if missing:
        raise ValueError(f"Missing features: {missing[:5]}...")

    # --- Build input array ---
    values = [features_dict[f] for f in service.feature_names]
    X_input = np.array(values).reshape(1, -1)

    # --- Get fraud probability ---
    fraud_prob = float(service.model.predict_proba(X_input)[0, 1])
    risk_level, risk_score, is_flagged = classify_risk(fraud_prob)

    # --- Generate LIME explanation ---
    lime_explanations = []
    try:
        explanation = service.lime_explainer.explain_instance(
            X_input[0],
            service.model.predict_proba,
            num_features=10,
            top_labels=1,
        )

        # Get the explanation for the fraud class (label 1)
        label = 1
        lime_list = explanation.as_list(label=label)

        for rule, weight in lime_list:
            lime_explanations.append({
                "feature_rule": rule,
                "weight": round(float(weight), 6),
                "direction": "increases_risk" if weight > 0 else "decreases_risk",
            })
    except Exception as e:
        logger.warning(f"LIME explanation failed: {e}")

    # --- Generate plain-English summary ---
    if lime_explanations:
        risk_drivers = [
            le for le in lime_explanations if le["direction"] == "increases_risk"
        ][:3]
        if risk_level in ("CRITICAL", "HIGH"):
            summary = (
                f"ALERT: {fraud_prob:.1%} fraud probability ({risk_level}). "
            )
        else:
            summary = f"Risk Level: {risk_level} ({fraud_prob:.1%}). "

        if risk_drivers:
            rules = [r["feature_rule"] for r in risk_drivers]
            summary += "Key factors: " + "; ".join(rules) + "."
        else:
            summary += "No strong risk-increasing factors detected."
    else:
        summary = f"Risk Level: {risk_level} ({fraud_prob:.1%}). LIME explanation unavailable."

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "transaction_id": transaction_id,
        "fraud_probability": round(fraud_prob, 6),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "is_flagged": is_flagged,
        "lime_explanation": lime_explanations,
        "plain_english_summary": summary,
        "inference_time_ms": round(elapsed_ms, 2),
    }


# ============================================================
# 4. FASTAPI APPLICATION
# ============================================================

# --- Initialize the service (models loaded at startup event) ---
model_service = ModelService()

# --- Create FastAPI app ---
app = FastAPI(
    title="Fraud Detection Inference API",
    description=(
        "Enterprise-grade REST API for real-time fraud detection. "
        "Scores transactions using XGBoost/LightGBM and explains "
        "every decision using SHAP and LIME. Part of the Explainable "
        "Fraud Detection System."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS middleware (allows Streamlit dashboard to call us) ---
api_cfg = config.get("api", {})
cors_origins = api_cfg.get("cors_origins", ["http://localhost:8501"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins + ["*"],  # Permissive for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Startup event: load all models ---
@app.on_event("startup")
async def startup_load_models():
    """Load all ML artifacts when the API starts."""
    try:
        log_phase_start("Phase 8: Inference API")
        model_service.load_all()
        log_phase_end("Phase 8: Inference API", status="SUCCESS")
    except Exception as e:
        logger.error(f"Failed to load models at startup: {e}")
        logger.error(traceback.format_exc())
        log_phase_end("Phase 8: Inference API", status="FAILED", error=str(e))
        # Don't raise -- let the API start so /health can report the problem


# ============================================================
# 5. API ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns the status of all loaded components. Use this to verify
    the API is ready before sending predictions.
    """
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        model_loaded=model_service.model is not None,
        shap_loaded=model_service.shap_explainer is not None,
        lime_loaded=model_service.lime_explainer is not None,
        scaler_loaded=model_service.scaler is not None,
        feature_count=len(model_service.feature_names) if model_service.feature_names else 0,
        uptime_seconds=round(model_service.get_uptime(), 2),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """
    Returns model metadata: type, features, performance metrics,
    and comparison between XGBoost and LightGBM.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    return ModelInfoResponse(
        model_type=model_service.model_type,
        feature_count=len(model_service.feature_names),
        feature_names=model_service.feature_names,
        metrics=model_service.metrics or {},
        model_comparison=model_service.model_comparison or {},
        config={
            "api_host": api_cfg.get("host", "0.0.0.0"),
            "api_port": api_cfg.get("port", 8000),
            "cors_origins": cors_origins,
            "max_display_features": config.get("explainability", {}).get(
                "max_display_features", 15
            ),
        },
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: TransactionInput):
    """
    Score a single transaction and return SHAP explanation.

    This is the primary endpoint. It returns:
    - Fraud probability (0.0 to 1.0)
    - Risk level (LOW / MEDIUM / HIGH / CRITICAL)
    - Risk score (0-100)
    - Whether it's flagged for review
    - SHAP feature contributions (sorted by importance)
    - Plain-English explanation for non-technical investigators
    - Inference time in milliseconds
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    try:
        model_service.prediction_count += 1
        txn_id = f"txn_{model_service.prediction_count:06d}"

        result = predict_single(
            service=model_service,
            features_dict=transaction.features,
            transaction_id=txn_id,
        )

        logger.info(
            f"Prediction #{model_service.prediction_count}: "
            f"{result['risk_level']} "
            f"(prob={result['fraud_probability']:.4f}, "
            f"time={result['inference_time_ms']:.1f}ms)"
        )

        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchTransactionInput):
    """
    Score a batch of transactions (max 100).

    Returns predictions with SHAP explanations for each transaction.
    Useful for processing historical data or running batch audits.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    if len(batch.transactions) > 100:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(batch.transactions)} exceeds max of 100."
        )

    batch_start = time.time()
    predictions = []

    for i, txn in enumerate(batch.transactions):
        try:
            model_service.prediction_count += 1
            txn_id = f"batch_{model_service.prediction_count:06d}"
            result = predict_single(
                service=model_service,
                features_dict=txn.features,
                transaction_id=txn_id,
            )
            predictions.append(result)
        except Exception as e:
            logger.warning(f"Batch item {i} failed: {e}")
            predictions.append({
                "transaction_id": f"batch_{i:06d}_ERROR",
                "fraud_probability": -1.0,
                "risk_level": "ERROR",
                "risk_score": -1,
                "is_flagged": False,
                "threshold_used": 0.5,
                "prediction_label": "ERROR",
                "shap_explanation": [],
                "plain_english_summary": f"Prediction failed: {str(e)}",
                "inference_time_ms": 0.0,
                "model_version": model_service.model_version,
                "timestamp": datetime.now().isoformat(),
            })

    batch_elapsed = (time.time() - batch_start) * 1000
    flagged = sum(1 for p in predictions if p.get("is_flagged", False))

    logger.info(
        f"Batch prediction: {len(predictions)} transactions, "
        f"{flagged} flagged, {batch_elapsed:.1f}ms total"
    )

    return BatchPredictionResponse(
        total_transactions=len(predictions),
        flagged_count=flagged,
        predictions=[PredictionResponse(**p) for p in predictions],
        batch_inference_time_ms=round(batch_elapsed, 2),
    )


@app.post("/predict/lime", response_model=LimePredictionResponse, tags=["Prediction"])
async def predict_with_lime(transaction: TransactionInput):
    """
    Score a single transaction with LIME explanation.

    LIME is slower than SHAP but provides rule-based explanations
    (e.g., "V14 <= -5.23") that are easier for compliance auditors
    to verify against raw data.

    NOTE: This endpoint is ~100x slower than /predict due to LIME's
    perturbation-based approach. Use /predict for real-time scoring.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    if model_service.lime_explainer is None:
        raise HTTPException(
            status_code=503,
            detail="LIME explainer not loaded. Check /health."
        )

    try:
        result = predict_lime(
            service=model_service,
            features_dict=transaction.features,
            transaction_id=f"lime_{model_service.prediction_count + 1:06d}",
        )

        logger.info(
            f"LIME prediction: {result['risk_level']} "
            f"(prob={result['fraud_probability']:.4f}, "
            f"time={result['inference_time_ms']:.1f}ms)"
        )

        return LimePredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"LIME prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"LIME prediction failed: {str(e)}")


@app.get("/features", tags=["System"])
async def get_features():
    """
    Returns the list of feature names expected by the model.

    Useful for building client applications that need to know
    which features to send.
    """
    if not model_service.feature_names:
        raise HTTPException(status_code=503, detail="Features not loaded.")

    return {
        "feature_count": len(model_service.feature_names),
        "feature_names": model_service.feature_names,
    }


@app.get("/sample", tags=["System"])
async def get_sample_transaction():
    """
    Returns a sample transaction with all required features set to 0.0.

    Useful for testing the API -- copy this, modify values, and POST to /predict.
    """
    if not model_service.feature_names:
        raise HTTPException(status_code=503, detail="Features not loaded.")

    sample = {f: 0.0 for f in model_service.feature_names}
    return {
        "description": "Sample transaction template. Modify values and POST to /predict.",
        "features": sample,
    }


# ============================================================
# 6. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    host = api_cfg.get("host", "0.0.0.0")
    port = api_cfg.get("port", 8000)

    logger.info(f"Starting Fraud Detection API on {host}:{port}")
    logger.info(f"Docs available at: http://localhost:{port}/docs")
    logger.info(f"Health check at:   http://localhost:{port}/health")

    uvicorn.run(
        "src.api.inference_api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
