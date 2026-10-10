from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from app.model import DATASET_PATH, FEATURE_COLUMNS, MODEL_PATH, HeartDiseasePredictor


def build_models() -> dict:
    data = pd.read_csv(DATASET_PATH)
    X = data[FEATURE_COLUMNS]
    y = data["target"]

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        results[name] = {
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std()),
        }
    return results


def train_selected_model() -> None:
    data = pd.read_csv(DATASET_PATH)
    X = data[FEATURE_COLUMNS]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    print(metrics)

    Path(MODEL_PATH.parent).mkdir(parents=True, exist_ok=True)
    joblib = __import__("joblib")
    joblib.dump(model, MODEL_PATH)


if __name__ == "__main__":
    print(build_models())
    train_selected_model()
