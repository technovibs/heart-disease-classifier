from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "heart_disease.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "heart_disease_model.joblib"


class HeartDiseasePredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.pipeline: Pipeline | None = None

    def train(self, data: pd.DataFrame) -> Pipeline:
        if "target" not in data.columns:
            raise ValueError("Training data must include a 'target' column.")

        features = data[FEATURE_COLUMNS]
        target = data["target"]

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        pipeline.fit(features, target)
        self.pipeline = pipeline
        self.save()
        return pipeline

    def save(self) -> None:
        if self.pipeline is None:
            raise ValueError("No trained pipeline available to save.")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)

    def load(self) -> Pipeline:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        return self.pipeline

    def predict(self, patient_data: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            self.load()
        return pd.Series(self.pipeline.predict(patient_data[FEATURE_COLUMNS]), index=patient_data.index)

    def predict_proba(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            self.load()
        probabilities = self.pipeline.predict_proba(patient_data[FEATURE_COLUMNS])
        return pd.DataFrame(probabilities, columns=["probability_no_disease", "probability_disease"])


def load_or_train_model(dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH) -> HeartDiseasePredictor:
    predictor = HeartDiseasePredictor(model_path=model_path)

    if model_path.exists():
        predictor.load()
        return predictor

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    data = pd.read_csv(dataset_path)
    predictor.train(data)
    return predictor
