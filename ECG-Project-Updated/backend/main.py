import os
import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd

# Make `preprocessing` importable and model paths independent of the working dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.ecg_validator import (   # noqa: E402
    ECGValidator,
    ValidationConfig,
    MESSAGES,
)

LOW_CONFIDENCE = "LOW_CONFIDENCE"
MESSAGES[LOW_CONFIDENCE] = (
    "A 12-lead ECG was detected, but the model is not confident about it, so no class is shown. "
    "Please upload a clearer ECG image."
)

app = FastAPI(
    title="ECG Heart Disease Classification API",
    description="ECG image classification using PCA and SVM, with input validation",
    version="2.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load trained models + validator
# --------------------------------------------------

MODELS = ROOT / "models"

pca = joblib.load(MODELS / "pca.pkl")
model = joblib.load(MODELS / "ecg_model.pkl")

config = ValidationConfig.load(str(MODELS / "validation_thresholds.json"))
validator = ECGValidator(pca, config)

if config.max_recon_error is None:
    print("WARNING: models/validation_thresholds.json not found - the feature-"
          "distribution gate is OFF. Run: python training/calibrate_validation.py")

LABELS = {
    0: "NORMAL",
    1: "AHB",
    2: "MI",
    3: "PMI"
}

FEATURE_COLUMNS = [f"feature_{i}" for i in range(3060)]


def reject(result):
    """422 response. `detail` is a plain string so the existing frontend shows it."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": result.message,
            "code": result.code,
            "stage": result.stage,
            "twelve_lead": result.twelve_lead,
            "checks": [c.__dict__ for c in result.checks],
        },
    )


async def read_upload(file: UploadFile):
    # read at most limit+1 bytes so huge uploads are rejected cheaply
    limit = validator.cfg.max_file_bytes
    return await file.read(limit + 1)


@app.get("/")
def home():
    return {"message": "ECG Heart Disease Classification API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "validation": "full" if config.max_recon_error is not None else "image-checks-only",
    }


@app.post("/validate")
async def validate(file: UploadFile = File(...), rotation: int = Query(0, ge=0, le=180, multiple_of=180)):
    """Validation only (no prediction). Always 200; look at `valid`."""
    data = await read_upload(file)
    result = await run_in_threadpool(validator.validate_bytes, data, rotation)
    return result.to_dict()


@app.post("/predict")
async def predict(file: UploadFile = File(...), rotation: int = Query(0, ge=0, le=180, multiple_of=180)):

    data = await read_upload(file)

    # Gates 1-4: file, ECG-ness, signal quality, similarity to training data
    result = await run_in_threadpool(validator.validate_bytes, data, rotation)

    if not result.valid:
        return reject(result)

    # Prediction (features were already extracted during validation)
    features_df = pd.DataFrame([result.features], columns=FEATURE_COLUMNS)
    features_pca = pca.transform(features_df)

    prediction = int(model.predict(features_pca)[0])
    probabilities = model.predict_proba(features_pca)[0]

    # confidence of the class we actually report
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(prediction)])

    probs = {LABELS[int(c)]: round(float(p), 4) for c, p in zip(classes, probabilities)}

    # Gate 5: low confidence -> do not show a class
    if confidence < config.min_confidence:
        return JSONResponse(
            status_code=422,
            content={
                "detail": MESSAGES[LOW_CONFIDENCE],
                "code": LOW_CONFIDENCE,
                "stage": "confidence",
                "twelve_lead": True,
                "best_guess": LABELS[prediction],
                "confidence": round(confidence, 4),
                "probabilities": probs,
                "checks": [c.__dict__ for c in result.checks],
            },
        )

    return {
        "status": "ok",
        "prediction": LABELS[prediction],
        "confidence": round(confidence, 4),
        "probabilities": probs,
        "warnings": result.warnings,
    }
