from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime, timezone
import pandas as pd
import joblib
import shap
import os
import logging
import hashlib
import time
import io
import base64
import matplotlib
matplotlib.use('Agg') # Safe for server environments
import matplotlib.pyplot as plt

# --- Setup Logging (Requirement 6.3) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("api_audit.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hospital Readmission Risk API",
    version="1.0.0"
)

# --- Security & Rate Limiting (Requirement 6.3) ---
API_KEY = "hospital-secure-key-2026"
api_key_header = APIKeyHeader(name="X-API-Key")

# Simple in-memory rate limiter (max 5 requests per second per IP)
rate_limit_cache = {}

def check_rate_limit(request: Request):
    ip = request.client.host
    current_time = time.time()
    
    if ip not in rate_limit_cache:
        rate_limit_cache[ip] = []
        
    # Clean up old requests
    rate_limit_cache[ip] = [t for t in rate_limit_cache[ip] if current_time - t < 1.0]
    
    if len(rate_limit_cache[ip]) >= 5:
        logger.warning(f"Rate limit exceeded for IP: {ip}")
        raise HTTPException(status_code=429, detail="Too many requests. Limit 5 per second.")
        
    rate_limit_cache[ip].append(current_time)

def get_api_key(request: Request, api_key_header: str = Security(api_key_header)):
    check_rate_limit(request) # Apply rate limit
    if api_key_header != API_KEY:
        logger.warning(f"Unauthorized access attempt from {request.client.host}")
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key_header

# --- Global Variables ---
CALIB_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'calibrated_xgb_pipeline.pkl')
UNCALIB_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_pipeline.pkl')

calib_pipeline = None
preprocessor = None
explainer = None
all_feature_names = None
last_prediction_time = None  # Requirement 6.4

@app.on_event("startup")
def load_models():
    global calib_pipeline, preprocessor, explainer, all_feature_names
    logger.info("Starting up API, loading models into memory...")
    try:
        calib_pipeline = joblib.load(CALIB_MODEL_PATH)
        uncalib_pipeline = joblib.load(UNCALIB_MODEL_PATH)
        
        preprocessor = uncalib_pipeline.named_steps['preprocessor']
        xgb_model = uncalib_pipeline.named_steps['classifier']
        
        explainer = shap.TreeExplainer(xgb_model)
        
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
        all_feature_names = list(num_features) + list(cat_features)
        
        logger.info("Models and SHAP Explainer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

# --- Pydantic Schema (Requirement 6.2) ---
DiagEnum = Literal["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"]
MedEnum = Literal["Up", "Down", "Steady", "No"]

class PatientProfile(BaseModel):
    # Reject unknown fields
    class Config:
        extra = "forbid"

    age: int = Field(..., ge=0, le=100)
    time_in_hospital: int = Field(..., ge=1, le=14)
    num_lab_procedures: int = Field(..., ge=1)
    num_procedures: int = Field(..., ge=0)
    num_medications: int = Field(..., ge=1)
    number_outpatient: int = Field(..., ge=0)
    number_emergency: int = Field(..., ge=0)
    number_inpatient: int = Field(..., ge=0)
    number_diagnoses: int = Field(..., ge=1)
    
    admission_type_id: str
    discharge_disposition_id: str
    admission_source_id: str
    
    # Enforce valid Enums for clinical categories
    diag_1: DiagEnum
    diag_2: DiagEnum
    diag_3: DiagEnum
    
    # Sensible defaults for optional fields
    max_glu_serum: Literal[">300", "Norm", ">200", "Not_Measured"] = "Not_Measured"
    A1Cresult: Literal[">7", ">8", "Norm", "Not_Measured"] = "Not_Measured"
    
    change: Literal["Ch", "No"] = "No"
    diabetesMed: Literal["Yes", "No"] = "No"
    
    # Medications
    metformin: MedEnum = "No"
    repaglinide: MedEnum = "No"
    nateglinide: MedEnum = "No"
    chlorpropamide: MedEnum = "No"
    glimepiride: MedEnum = "No"
    acetohexamide: MedEnum = "No"
    glipizide: MedEnum = "No"
    glyburide: MedEnum = "No"
    tolbutamide: MedEnum = "No"
    pioglitazone: MedEnum = "No"
    rosiglitazone: MedEnum = "No"
    acarbose: MedEnum = "No"
    miglitol: MedEnum = "No"
    troglitazone: MedEnum = "No"
    tolazamide: MedEnum = "No"
    insulin: MedEnum = "No"
    glyburide_metformin: MedEnum = Field(default="No", alias="glyburide-metformin")
    glipizide_metformin: MedEnum = Field(default="No", alias="glipizide-metformin")
    glimepiride_pioglitazone: MedEnum = Field(default="No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: MedEnum = Field(default="No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: MedEnum = Field(default="No", alias="metformin-pioglitazone")
    
    # Note: 'race' and 'gender' are intentionally omitted due to Phase 5 Fairness Audit constraints.

# --- Endpoints ---

@app.get("/health")
def health_check():
    # Requirement 6.4
    return {
        "status": "online",
        "model_loaded": calib_pipeline is not None,
        "version": app.version,
        "last_prediction": last_prediction_time
    }

@app.post("/predict")
def predict_readmission(patient: PatientProfile, api_key: str = Depends(get_api_key)):
    global last_prediction_time
    if calib_pipeline is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")
        
    try:
        # Convert to dict and hash for audit logging
        patient_data = patient.dict(by_alias=True)
        input_hash = hashlib.sha256(str(patient_data).encode()).hexdigest()[:16]
        
        # 1. Get Score
        df = pd.DataFrame([patient_data])
        risk_score = calib_pipeline.predict_proba(df)[0, 1]
        
        # 2. Get SHAP
        transformed_data = preprocessor.transform(df)
        sv = explainer(transformed_data)
        
        feature_impacts = pd.DataFrame({'feature': all_feature_names, 'shap_value': sv[0].values})
        top_drivers_df = feature_impacts.reindex(feature_impacts['shap_value'].abs().sort_values(ascending=False).index).head(3)
        
        top_drivers = [
            {
                "feature": row['feature'],
                "impact": f"{'+' if row['shap_value'] > 0 else ''}{round(float(row['shap_value']), 3)}",
                "direction": "increases risk" if row['shap_value'] > 0 else "decreases risk"
            } for _, row in top_drivers_df.iterrows()
        ]
        
        # Requirement 7: Waterfall Plot Generation
        plt.figure(figsize=(8, 5))
        shap.plots.waterfall(sv[0], max_display=10, show=False)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        response_payload = {
            "risk_score": round(float(risk_score), 3),
            "risk_tier": "HIGH" if risk_score >= 0.20 else ("MEDIUM" if risk_score >= 0.10 else "LOW"),
            "top_drivers": top_drivers,
            "waterfall_plot_base64": plot_base64,
            "model_version": app.version,
            "calibrated": True
        }
        
        # Requirement 6.3 - Audit Logging
        last_prediction_time = datetime.now(timezone.utc).isoformat()
        logger.info(f"PREDICT | Hash: {input_hash} | Risk: {response_payload['risk_score']} | Drivers: {[d['feature'] for d in top_drivers]}")
        
        return response_payload
        
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
