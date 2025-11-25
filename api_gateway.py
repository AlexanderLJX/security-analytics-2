"""
Unified FastAPI Gateway for Phishing Detection
Combines Random Forest, XGBoost, and LLM-GRPO models into a single API

Endpoints:
    POST /predict                - Predict using all models (ensemble)
    POST /predict/rf             - Random Forest prediction only
    POST /predict/xgboost        - XGBoost prediction only
    POST /predict/llm            - LLM-GRPO prediction only
    POST /predict/batch          - Batch prediction with all models
    POST /predict/csv            - CSV file batch prediction
    GET  /health                 - Health check for all models
    GET  /models/info            - Get information about all models
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import numpy as np
import pandas as pd
import joblib
import io
import os
import sys
from datetime import datetime
import hashlib
import socket
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add model directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Random-Forest'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'XgBoost'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LLM-GRPO'))

# ============================================================================
# CONFIGURATION
# ============================================================================

RF_MODEL_PATH = "Random-Forest/checkpoints/phishing_detector/rf_phishing_detector.joblib"
XGBOOST_MODEL_PATH = "XgBoost/phishing_text_model.joblib"
LLM_MODEL_NAME = "AlexanderLJX/phishing-detection-qwen3-grpo"

# Model weights for ensemble (sum to 1.0)
ENSEMBLE_WEIGHTS = {
    "rf": 0.25,
    "xgboost": 0.35,
    "llm": 0.40
}

# Splunk HEC Configuration (from environment or defaults)
SPLUNK_HEC_URL = os.getenv("SPLUNK_HEC_URL", "")
SPLUNK_HEC_TOKEN = os.getenv("SPLUNK_HEC_TOKEN", "")
SPLUNK_INDEX = os.getenv("SPLUNK_INDEX", "security")
SPLUNK_ENABLED = bool(SPLUNK_HEC_URL and SPLUNK_HEC_TOKEN)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Models will be loaded lazily
rf_model = None
rf_scaler = None
rf_feature_names = None
rf_metrics = None

xgb_pipeline = None
xgb_threshold = None
xgb_metrics = None

llm_model = None
llm_tokenizer = None

models_loaded = {
    "rf": False,
    "xgboost": False,
    "llm": False
}

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class EmailRequest(BaseModel):
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")

class EmailBatchRequest(BaseModel):
    emails: List[EmailRequest] = Field(..., description="List of emails to analyze")

class ModelPrediction(BaseModel):
    is_phishing: bool
    phishing_probability: float
    confidence: float
    label: str

class RFPredictionResponse(ModelPrediction):
    risk_score: int
    recommended_action: str

class LLMPredictionResponse(ModelPrediction):
    reasoning: Optional[str] = None

class EnsemblePredictionResponse(BaseModel):
    """Combined prediction from all models"""
    # Ensemble result
    ensemble_prediction: bool
    ensemble_probability: float
    ensemble_label: str
    recommended_action: str
    risk_score: int

    # Individual model results
    rf_prediction: Optional[ModelPrediction] = None
    xgboost_prediction: Optional[ModelPrediction] = None
    llm_prediction: Optional[LLMPredictionResponse] = None

    # Metadata
    models_used: List[str]
    agreement_score: float  # How much models agree (0-1)

class HealthResponse(BaseModel):
    status: str
    models: dict
    timestamp: str

class ModelInfoResponse(BaseModel):
    models: dict
    ensemble_weights: dict

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_rf_model():
    """Load Random Forest model"""
    global rf_model, rf_scaler, rf_feature_names, rf_metrics, models_loaded

    try:
        if not os.path.exists(RF_MODEL_PATH):
            print(f"RF model not found at {RF_MODEL_PATH}")
            return False

        model_data = joblib.load(RF_MODEL_PATH)
        rf_model = model_data["model"]
        rf_scaler = model_data["scaler"]
        rf_feature_names = model_data["feature_names"]
        rf_metrics = model_data.get("metrics", {})
        models_loaded["rf"] = True
        print(f"[OK] Random Forest model loaded from {RF_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to load RF model: {e}")
        return False

def load_xgboost_model():
    """Load XGBoost model"""
    global xgb_pipeline, xgb_threshold, xgb_metrics, models_loaded

    try:
        if not os.path.exists(XGBOOST_MODEL_PATH):
            print(f"XGBoost model not found at {XGBOOST_MODEL_PATH}")
            return False

        model_data = joblib.load(XGBOOST_MODEL_PATH)
        xgb_pipeline = model_data["pipeline"]
        xgb_threshold = model_data.get("threshold", 0.5)
        xgb_metrics = model_data.get("metrics", {})
        models_loaded["xgboost"] = True
        print(f"[OK] XGBoost model loaded from {XGBOOST_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to load XGBoost model: {e}")
        return False

def load_llm_model():
    """Load LLM-GRPO model (requires GPU)"""
    global llm_model, llm_tokenizer, models_loaded

    try:
        import torch
        if not torch.cuda.is_available():
            print("[SKIP] LLM model requires GPU - CUDA not available")
            return False

        from unsloth import FastLanguageModel

        llm_model, llm_tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-4B-Base",
            max_seq_length=2048,
            load_in_4bit=False,
            fast_inference=True,
            max_lora_rank=32,
            gpu_memory_utilization=0.8,
        )

        # Setup chat template
        SYSTEM_PROMPT = """You are an expert cybersecurity analyst specializing in phishing email detection.
Analyze the given email carefully and provide your reasoning.
Place your analysis between <start_analysis> and <end_analysis>.
Then, provide your classification between <CLASSIFICATION></CLASSIFICATION>.
Respond with either "PHISHING" or "LEGITIMATE"."""

        chat_template = \
            "{% if messages[0]['role'] == 'system' %}"\
                "{{ messages[0]['content'] + eos_token }}"\
                "{% set loop_messages = messages[1:] %}"\
            "{% else %}"\
                "{{ '{system_prompt}' + eos_token }}"\
                "{% set loop_messages = messages %}"\
            "{% endif %}"\
            "{% for message in loop_messages %}"\
                "{% if message['role'] == 'user' %}"\
                    "{{ message['content'] }}"\
                "{% elif message['role'] == 'assistant' %}"\
                    "{{ message['content'] + eos_token }}"\
                "{% endif %}"\
            "{% endfor %}"\
            "{% if add_generation_prompt %}{{ '<start_analysis>' }}"\
            "{% endif %}"

        chat_template = chat_template.replace("'{system_prompt}'", f"'{SYSTEM_PROMPT}'")
        llm_tokenizer.chat_template = chat_template

        models_loaded["llm"] = True
        print(f"[OK] LLM-GRPO model loaded from {LLM_MODEL_NAME}")
        return True
    except ImportError:
        print("[SKIP] LLM dependencies not installed (unsloth, vllm)")
        return False
    except Exception as e:
        print(f"[FAIL] Failed to load LLM model: {e}")
        return False

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_rf_features(email_text: str) -> dict:
    """Extract features for Random Forest model"""
    try:
        from feature_extraction_rf import features_from_text
        return features_from_text(email_text)
    except ImportError:
        # Fallback: minimal feature extraction
        return extract_basic_features(email_text)

def extract_xgb_features(subject: str, body: str) -> pd.DataFrame:
    """Extract features for XGBoost model"""
    try:
        from feature_extraction_text import features_from_dataframe
        df = pd.DataFrame([{"subject": subject, "body": body}])
        return features_from_dataframe(df)
    except ImportError:
        # Fallback
        return pd.DataFrame([extract_basic_features(f"Subject: {subject}\n\n{body}")])

def extract_basic_features(text: str) -> dict:
    """Basic feature extraction fallback"""
    import re

    text_lower = text.lower()

    return {
        "body_length": len(text),
        "word_count": len(text.split()),
        "url_count": len(re.findall(r'https?://\S+', text)),
        "exclamation_count": text.count('!'),
        "question_count": text.count('?'),
        "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "urgency_keyword_count": sum(1 for w in ['urgent', 'immediately', 'now', 'act'] if w in text_lower),
        "financial_keyword_count": sum(1 for w in ['bank', 'account', 'payment', 'money'] if w in text_lower),
        "credential_keyword_count": sum(1 for w in ['password', 'login', 'verify', 'confirm'] if w in text_lower),
    }

# ============================================================================
# SPLUNK INTEGRATION
# ============================================================================

def send_to_splunk(email_data: dict, prediction_result: dict, source: str = "ensemble"):
    """Send prediction event to Splunk HTTP Event Collector"""
    if not SPLUNK_ENABLED:
        return False

    try:
        # Create email hash for deduplication
        email_hash = hashlib.sha256(
            f"{email_data.get('subject', '')}{email_data.get('body', '')}".encode()
        ).hexdigest()

        # Determine severity based on risk score
        risk_score = prediction_result.get("risk_score", prediction_result.get("ensemble_probability", 0) * 100)
        if isinstance(risk_score, float) and risk_score <= 1:
            risk_score = int(risk_score * 100)

        if risk_score > 90:
            severity = "CRITICAL"
        elif risk_score > 70:
            severity = "HIGH"
        elif risk_score > 50:
            severity = "MEDIUM"
        elif risk_score > 30:
            severity = "LOW"
        else:
            severity = "INFO"

        # Build Splunk HEC event payload
        event = {
            "time": datetime.utcnow().timestamp(),
            "host": socket.gethostname(),
            "source": f"phishing-gateway-{source}",
            "sourcetype": "phishing_detection",
            "index": SPLUNK_INDEX,
            "event": {
                "alert_id": f"phish-{source}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "event_type": "email_analysis",
                "severity": severity,
                "email": {
                    "subject": email_data.get("subject", "")[:100],
                    "body_preview": email_data.get("body", "")[:200],
                    "hash": email_hash,
                    "size_bytes": len(email_data.get("body", ""))
                },
                "detection": {
                    "classification": prediction_result.get("ensemble_label", prediction_result.get("label", "Unknown")),
                    "is_phishing": prediction_result.get("ensemble_prediction", prediction_result.get("is_phishing", False)),
                    "probability": prediction_result.get("ensemble_probability", prediction_result.get("phishing_probability", 0)),
                    "risk_score": risk_score,
                    "recommended_action": prediction_result.get("recommended_action", "REVIEW"),
                    "models_used": prediction_result.get("models_used", [source]),
                    "agreement_score": prediction_result.get("agreement_score", 1.0)
                },
                "actions_taken": {
                    "email_quarantined": risk_score > 70,
                    "sender_blocked": prediction_result.get("recommended_action") == "BLOCK",
                    "soc_alerted": risk_score > 80
                }
            }
        }

        # Send to Splunk HEC
        headers = {
            "Authorization": f"Splunk {SPLUNK_HEC_TOKEN}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            SPLUNK_HEC_URL,
            json=event,
            headers=headers,
            timeout=5,
            verify=False  # Set to True in production with proper SSL certs
        )

        if response.status_code == 200:
            print(f"[SPLUNK] {event['event']['alert_id']} (severity: {severity})")
            return True
        else:
            print(f"[ERROR] Splunk error: {response.status_code}")
            return False

    except Exception as e:
        print(f"[ERROR] Splunk send failed: {e}")
        return False

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_rf(subject: str, body: str) -> dict:
    """Make prediction using Random Forest model"""
    if not models_loaded["rf"]:
        raise HTTPException(status_code=503, detail="Random Forest model not loaded")

    email_text = f"Subject: {subject}\n\n{body}"
    features_dict = extract_rf_features(email_text)
    feature_vector = np.array([features_dict.get(name, 0.0) for name in rf_feature_names])
    X = feature_vector.reshape(1, -1)
    X_scaled = rf_scaler.transform(X)

    prediction = rf_model.predict(X_scaled)[0]
    proba = rf_model.predict_proba(X_scaled)[0]

    # Determine action
    if proba[1] > 0.9:
        action = "BLOCK"
    elif proba[1] > 0.7:
        action = "QUARANTINE"
    elif proba[1] > 0.5:
        action = "REVIEW"
    else:
        action = "ALLOW"

    return {
        "is_phishing": bool(prediction == 1),
        "phishing_probability": float(proba[1]),
        "confidence": float(proba[prediction]),
        "label": "Phishing" if prediction == 1 else "Legitimate",
        "risk_score": int(proba[1] * 100),
        "recommended_action": action
    }

def predict_xgboost(subject: str, body: str) -> dict:
    """Make prediction using XGBoost model"""
    if not models_loaded["xgboost"]:
        raise HTTPException(status_code=503, detail="XGBoost model not loaded")

    X = extract_xgb_features(subject, body)
    proba = xgb_pipeline.predict_proba(X)[0, 1]
    prediction = 1 if proba >= xgb_threshold else 0

    return {
        "is_phishing": bool(prediction == 1),
        "phishing_probability": float(proba),
        "confidence": float(abs(proba - 0.5) * 2),
        "label": "Phishing" if prediction == 1 else "Legitimate"
    }

def predict_llm(subject: str, body: str) -> dict:
    """Make prediction using LLM-GRPO model"""
    if not models_loaded["llm"]:
        raise HTTPException(status_code=503, detail="LLM model not loaded")

    import re
    from vllm import SamplingParams

    email_text = f"Subject: {subject}\n\n{body}"

    SYSTEM_PROMPT = """You are an expert cybersecurity analyst specializing in phishing email detection.
Analyze the given email carefully and provide your reasoning.
Place your analysis between <start_analysis> and <end_analysis>.
Then, provide your classification between <CLASSIFICATION></CLASSIFICATION>.
Respond with either "PHISHING" or "LEGITIMATE"."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this email:\n\n{email_text}"},
    ]

    text = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        max_tokens=1024,
    )

    output = llm_model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=llm_model.load_lora(LLM_MODEL_NAME),
    )[0].outputs[0].text

    # Parse output
    result = {
        "is_phishing": False,
        "phishing_probability": 0.5,
        "confidence": 0.5,
        "label": "Unknown",
        "reasoning": ""
    }

    # Extract reasoning
    reasoning_match = re.search(r"<start_analysis>(.*?)<end_analysis>", output, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract classification
    class_match = re.search(r"<CLASSIFICATION>(.*?)</CLASSIFICATION>", output, re.DOTALL)
    if class_match:
        prediction = class_match.group(1).strip().upper()
        if "PHISHING" in prediction:
            result["is_phishing"] = True
            result["phishing_probability"] = 0.9
            result["confidence"] = 0.9
            result["label"] = "Phishing"
        elif "LEGITIMATE" in prediction:
            result["is_phishing"] = False
            result["phishing_probability"] = 0.1
            result["confidence"] = 0.9
            result["label"] = "Legitimate"

    return result

def predict_ensemble(subject: str, body: str, models: List[str] = None) -> dict:
    """Make ensemble prediction using available models"""

    if models is None:
        models = [m for m, loaded in models_loaded.items() if loaded]

    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")

    predictions = {}
    probabilities = []
    weights = []

    # Get predictions from each model
    if "rf" in models and models_loaded["rf"]:
        try:
            predictions["rf"] = predict_rf(subject, body)
            probabilities.append(predictions["rf"]["phishing_probability"])
            weights.append(ENSEMBLE_WEIGHTS["rf"])
        except Exception as e:
            print(f"RF prediction failed: {e}")

    if "xgboost" in models and models_loaded["xgboost"]:
        try:
            predictions["xgboost"] = predict_xgboost(subject, body)
            probabilities.append(predictions["xgboost"]["phishing_probability"])
            weights.append(ENSEMBLE_WEIGHTS["xgboost"])
        except Exception as e:
            print(f"XGBoost prediction failed: {e}")

    if "llm" in models and models_loaded["llm"]:
        try:
            predictions["llm"] = predict_llm(subject, body)
            probabilities.append(predictions["llm"]["phishing_probability"])
            weights.append(ENSEMBLE_WEIGHTS["llm"])
        except Exception as e:
            print(f"LLM prediction failed: {e}")

    if not probabilities:
        raise HTTPException(status_code=500, detail="All model predictions failed")

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Calculate weighted ensemble probability
    ensemble_prob = sum(p * w for p, w in zip(probabilities, weights))
    ensemble_prediction = ensemble_prob >= 0.5

    # Calculate agreement score (how much models agree)
    if len(probabilities) > 1:
        binary_preds = [p >= 0.5 for p in probabilities]
        agreement = sum(binary_preds) / len(binary_preds)
        agreement_score = max(agreement, 1 - agreement)  # Normalize to 0.5-1.0
    else:
        agreement_score = 1.0

    # Determine action
    if ensemble_prob > 0.9:
        action = "BLOCK"
    elif ensemble_prob > 0.7:
        action = "QUARANTINE"
    elif ensemble_prob > 0.5:
        action = "REVIEW"
    else:
        action = "ALLOW"

    return {
        "ensemble_prediction": ensemble_prediction,
        "ensemble_probability": float(ensemble_prob),
        "ensemble_label": "Phishing" if ensemble_prediction else "Legitimate",
        "recommended_action": action,
        "risk_score": int(ensemble_prob * 100),
        "rf_prediction": predictions.get("rf"),
        "xgboost_prediction": predictions.get("xgboost"),
        "llm_prediction": predictions.get("llm"),
        "models_used": list(predictions.keys()),
        "agreement_score": float(agreement_score)
    }

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Phishing Detection Gateway API",
    description="Unified API for phishing detection using Random Forest, XGBoost, and LLM-GRPO models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("\n" + "="*60)
    print("PHISHING DETECTION GATEWAY")
    print("="*60)
    print("\nLoading models...")

    load_rf_model()
    load_xgboost_model()
    # LLM is loaded lazily due to GPU memory requirements
    # load_llm_model()  # Uncomment if GPU available

    loaded = [m for m, l in models_loaded.items() if l]
    print(f"\n[OK] Models loaded: {loaded}")
    print("="*60 + "\n")

@app.get("/", tags=["General"])
def root():
    """API root - service info"""
    return {
        "service": "Phishing Detection Gateway API",
        "version": "1.0.0",
        "models_available": [m for m, l in models_loaded.items() if l],
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Health check for all models"""
    return {
        "status": "healthy" if any(models_loaded.values()) else "degraded",
        "models": {
            "random_forest": {"loaded": models_loaded["rf"], "path": RF_MODEL_PATH},
            "xgboost": {"loaded": models_loaded["xgboost"], "path": XGBOOST_MODEL_PATH},
            "llm_grpo": {"loaded": models_loaded["llm"], "model": LLM_MODEL_NAME}
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/models/info", response_model=ModelInfoResponse, tags=["General"])
def model_info():
    """Get information about all models"""
    info = {
        "models": {},
        "ensemble_weights": ENSEMBLE_WEIGHTS
    }

    if models_loaded["rf"]:
        info["models"]["random_forest"] = {
            "type": "RandomForestClassifier",
            "n_features": len(rf_feature_names) if rf_feature_names else 0,
            "metrics": rf_metrics
        }

    if models_loaded["xgboost"]:
        info["models"]["xgboost"] = {
            "type": "XGBClassifier with PolynomialFeatures",
            "threshold": xgb_threshold,
            "metrics": xgb_metrics
        }

    if models_loaded["llm"]:
        info["models"]["llm_grpo"] = {
            "type": "Qwen3-4B with LoRA (GRPO trained)",
            "model": LLM_MODEL_NAME
        }

    return info

@app.post("/predict", response_model=EnsemblePredictionResponse, tags=["Prediction"])
def predict(email: EmailRequest, send_splunk: bool = Query(True, description="Send result to Splunk if configured")):
    """
    Predict using ensemble of all available models.
    Returns combined prediction with individual model results.
    Optionally sends to Splunk HEC if configured.
    """
    result = predict_ensemble(email.subject, email.body)

    # Send to Splunk if enabled and phishing detected or high risk
    if send_splunk and SPLUNK_ENABLED:
        if result["ensemble_prediction"] or result["risk_score"] > 50:
            send_to_splunk(
                email_data={"subject": email.subject, "body": email.body},
                prediction_result=result,
                source="ensemble"
            )

    return result

@app.post("/predict/rf", response_model=RFPredictionResponse, tags=["Prediction"])
def predict_random_forest(email: EmailRequest):
    """Predict using Random Forest model only"""
    return predict_rf(email.subject, email.body)

@app.post("/predict/xgboost", response_model=ModelPrediction, tags=["Prediction"])
def predict_xgb(email: EmailRequest):
    """Predict using XGBoost model only"""
    return predict_xgboost(email.subject, email.body)

@app.post("/predict/llm", response_model=LLMPredictionResponse, tags=["Prediction"])
def predict_language_model(email: EmailRequest):
    """Predict using LLM-GRPO model only (requires GPU)"""
    return predict_llm(email.subject, email.body)

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(
    batch: EmailBatchRequest,
    models: Optional[str] = Query(None, description="Comma-separated list of models to use: rf,xgboost,llm")
):
    """
    Batch prediction for multiple emails.
    Optionally specify which models to use.
    """
    model_list = models.split(",") if models else None

    results = []
    for email in batch.emails:
        try:
            result = predict_ensemble(email.subject, email.body, model_list)
            results.append({
                "subject": email.subject[:50] + "..." if len(email.subject) > 50 else email.subject,
                **result
            })
        except Exception as e:
            results.append({
                "subject": email.subject[:50],
                "error": str(e)
            })

    phishing_count = sum(1 for r in results if r.get("ensemble_prediction", False))

    return {
        "status": "success",
        "total_emails": len(results),
        "phishing_detected": phishing_count,
        "legitimate_detected": len(results) - phishing_count,
        "predictions": results
    }

@app.post("/predict/csv", tags=["Prediction"])
async def predict_csv(
    file: UploadFile = File(..., description="CSV file with 'subject' and 'body' columns"),
    models: Optional[str] = Query(None, description="Comma-separated list of models to use")
):
    """
    Upload CSV file for batch prediction.
    CSV must have 'subject' and 'body' columns.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        if 'subject' not in df.columns or 'body' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'subject' and 'body' columns"
            )

        model_list = models.split(",") if models else None
        results = []

        for idx, row in df.iterrows():
            subject = str(row.get('subject', ''))
            body = str(row.get('body', ''))

            try:
                result = predict_ensemble(subject, body, model_list)
                results.append({
                    "row_index": int(idx),
                    "subject": subject[:50] + "..." if len(subject) > 50 else subject,
                    **result
                })
            except Exception as e:
                results.append({
                    "row_index": int(idx),
                    "subject": subject[:50],
                    "error": str(e)
                })

        phishing_count = sum(1 for r in results if r.get("ensemble_prediction", False))

        return {
            "status": "success",
            "filename": file.filename,
            "total_emails": len(results),
            "phishing_detected": phishing_count,
            "legitimate_detected": len(results) - phishing_count,
            "predictions": results
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.post("/load/llm", tags=["Model Management"])
def load_llm_endpoint():
    """
    Manually load the LLM model.
    Use this if LLM wasn't loaded at startup due to GPU constraints.
    """
    success = load_llm_model()
    if success:
        return {"status": "success", "message": "LLM model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load LLM model")

# ============================================================================
# SPLUNK ENDPOINTS
# ============================================================================

@app.get("/splunk/status", tags=["Splunk"])
def splunk_status():
    """Check Splunk HEC connection status"""
    return {
        "enabled": SPLUNK_ENABLED,
        "hec_url": SPLUNK_HEC_URL[:50] + "..." if len(SPLUNK_HEC_URL) > 50 else SPLUNK_HEC_URL if SPLUNK_HEC_URL else None,
        "index": SPLUNK_INDEX if SPLUNK_ENABLED else None,
        "token_configured": bool(SPLUNK_HEC_TOKEN)
    }

@app.post("/splunk/configure", tags=["Splunk"])
def configure_splunk(
    hec_url: str = Query(..., description="Splunk HEC URL (e.g., https://splunk:8088/services/collector)"),
    token: str = Query(..., description="Splunk HEC Token"),
    index: str = Query("security", description="Splunk index name")
):
    """
    Configure Splunk HEC at runtime.
    This allows setting Splunk credentials without environment variables.
    """
    global SPLUNK_HEC_URL, SPLUNK_HEC_TOKEN, SPLUNK_INDEX, SPLUNK_ENABLED

    SPLUNK_HEC_URL = hec_url
    SPLUNK_HEC_TOKEN = token
    SPLUNK_INDEX = index
    SPLUNK_ENABLED = True

    return {
        "status": "success",
        "message": "Splunk HEC configured",
        "hec_url": hec_url[:50] + "..." if len(hec_url) > 50 else hec_url,
        "index": index
    }

@app.post("/splunk/test", tags=["Splunk"])
def test_splunk():
    """Send a test event to Splunk to verify connection"""
    if not SPLUNK_ENABLED:
        raise HTTPException(status_code=400, detail="Splunk not configured. Use /splunk/configure first.")

    test_result = {
        "ensemble_prediction": True,
        "ensemble_probability": 0.85,
        "ensemble_label": "Phishing",
        "risk_score": 85,
        "recommended_action": "QUARANTINE",
        "models_used": ["test"],
        "agreement_score": 1.0
    }

    success = send_to_splunk(
        email_data={"subject": "TEST: Splunk Connection Test", "body": "This is a test event from the Phishing Detection API."},
        prediction_result=test_result,
        source="test"
    )

    if success:
        return {"status": "success", "message": "Test event sent to Splunk successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send test event to Splunk")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

    print("="*60)
    print("PHISHING DETECTION GATEWAY API")
    print("="*60)
    print(f"\nStarting server on port {port}...")
    print(f"API Documentation: http://localhost:{port}/docs")
    print(f"Health Check: http://localhost:{port}/health")
    print("\nPress CTRL+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
