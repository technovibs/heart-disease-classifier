from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model import FEATURE_COLUMNS, load_or_train_model
from app.schemas import PatientFeatures
from fetch_dataset import fetch_heart_disease_dataset

app = FastAPI(
    title="Heart Disease Classifier API",
    description="Predict heart disease risk from patient health data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    predictor = load_or_train_model()
except FileNotFoundError:
    fetch_heart_disease_dataset()
    predictor = load_or_train_model()


@app.get("/health")
def healthcheck() -> dict[str, Any]:
    return {
        "status": "ok",
        "app_env": settings.app_env,
        "model": "heart_disease_classifier",
        "features": FEATURE_COLUMNS,
    }


@app.post("/predict")
def predict(patient: PatientFeatures) -> dict[str, Any]:
    record = pd.DataFrame([patient.model_dump()])
    probability = float(predictor.predict_proba(record)["probability_disease"].iloc[0])
    prediction = int(probability >= 0.5)
    risk_level = "high" if prediction == 1 else "low"

    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "probability": round(probability, 4),
    }


@app.post("/predict/batch")
def predict_batch(patients: list[PatientFeatures]) -> list[dict[str, Any]]:
    records = pd.DataFrame([patient.model_dump() for patient in patients])
    probabilities = predictor.predict_proba(records)["probability_disease"]

    results: list[dict[str, Any]] = []
    for probability in probabilities:
        prediction = int(probability >= 0.5)
        results.append(
            {
                "prediction": prediction,
                "risk_level": "high" if prediction == 1 else "low",
                "probability": round(float(probability), 4),
            }
        )
    return results


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Heart Disease Classifier API is running."}
