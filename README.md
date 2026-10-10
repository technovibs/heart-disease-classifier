# Heart Disease Classifier

A production-ready machine learning project that predicts heart disease risk from patient health data and exposes the model through a monitored FastAPI API.

## Overview

This project includes:
- a heart disease prediction model trained on the UCI ML Repository Heart Disease dataset,
- a FastAPI service for single and batch inference,
- MLflow experiment tracking,
- Docker and Docker Compose support,
- a clean separation between model logic, API schemas, and application entrypoints.

## Project structure

```text
heart-disease-classifier/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── model.py
│   └── schemas.py
├── data/
│   └── .gitkeep
├── model/
│   └── .gitkeep
├── notebooks/
│   └── eda.ipynb
├── scripts/
│   └── train_models.py
├── tests/
│   └── test_api.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── fetch_dataset.py
├── README.md
├── requirements.txt
├── train.py
└── .venv/
```

## Features

- Binary classification for heart disease risk
- preprocessing with median imputation + standardization
- FastAPI endpoints for single and batch predictions
- MLflow experiment tracking and model artifact logging
- containerized deployment using Docker
- dataset fetch from UCI ML Repository

## Local setup

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Fetch dataset:

```bash
python fetch_dataset.py
```

4. Train the model:

```bash
python train.py
```

5. Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API endpoints

### Health check

```bash
curl http://localhost:8000/health
```

### Predict single patient

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
  }'
```

### Predict batch

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "age": 52,
      "sex": 1,
      "cp": 0,
      "trestbps": 125,
      "chol": 212,
      "fbs": 0,
      "restecg": 1,
      "thalach": 168,
      "exang": 0,
      "oldpeak": 1.0,
      "slope": 2,
      "ca": 2,
      "thal": 3
    }
  ]'
```

## Docker

Build image:

```bash
docker build -t heart-disease-classifier .
```

Run container:

```bash
docker run -p 8000:8000 heart-disease-classifier
```

## Docker Compose

```bash
docker compose up --build
```

Then:
- API: http://localhost:8000
- MLflow UI: http://localhost:5000

## Dataset

The dataset is fetched from the UCI ML Repository:
- Heart Disease dataset
- Dataset ID: 45
- Source: https://archive.ics.uci.edu/dataset/45/heart+disease

This project uses:

```python
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets
```

and saves the combined dataset to:

```text
data/heart_disease.csv
```

## Model details

A logistic regression model is trained with:
- `SimpleImputer(strategy="median")`
- `StandardScaler()`
- `LogisticRegression(...)`

The API returns:
- `prediction`: 0 or 1
- `risk_level`: "low" or "high"
- `probability`: probability of disease

## Monitoring and tracking

MLflow is used to track:
- model hyperparameters
- evaluation metrics
- model artifacts

Run:

```bash
mlflow ui
```

Then open http://localhost:5000

## Testing

```bash
pytest
```

## Notes

- For production, add train/test validation and monitor drift over time.
- Replace this baseline logistic model with a more advanced algorithm if required.
- Keep the model artifact and dataset versioning controlled in MLflow or a registry.
