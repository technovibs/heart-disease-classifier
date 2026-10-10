from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.config import settings
from app.model import DATASET_PATH, FEATURE_COLUMNS, MODEL_PATH, HeartDiseasePredictor


def train_and_log_model() -> None:
    Path(MODEL_PATH.parent).mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}. Fetching it now...")
        from fetch_dataset import fetch_heart_disease_dataset
        fetch_heart_disease_dataset()

    data = pd.read_csv(DATASET_PATH)
    print(f"Loading dataset from {DATASET_PATH}")
    print(f"Dataset shape: {data.shape}")

    X = data[FEATURE_COLUMNS]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    predictor = HeartDiseasePredictor(model_path=MODEL_PATH)

    train_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    predictor.train(train_data)

    y_pred = predictor.predict(X_test)
    y_pred_proba = predictor.predict_proba(X_test)["probability_disease"]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    with mlflow.start_run(run_name="heart_disease_classifier"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.log_artifact(str(MODEL_PATH))

    print(f"\nModel trained and saved to {predictor.model_path}")
    print("MLflow run completed. Use `mlflow ui` to view the dashboard.")


if __name__ == "__main__":
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    train_and_log_model()
