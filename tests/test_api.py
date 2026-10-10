from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "heart_disease_classifier"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_predict_endpoint():
    payload = {
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
        "thal": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "risk_level" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert data["risk_level"] in ["low", "high"]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_batch_endpoint():
    payloads = [
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
            "thal": 3,
        },
        {
            "age": 45,
            "sex": 0,
            "cp": 2,
            "trestbps": 110,
            "chol": 264,
            "fbs": 0,
            "restecg": 0,
            "thalach": 132,
            "exang": 0,
            "oldpeak": 1.2,
            "slope": 1,
            "ca": 0,
            "thal": 3,
        },
    ]
    response = client.post("/predict/batch", json=payloads)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    for result in data:
        assert "prediction" in result
        assert "risk_level" in result
        assert "probability" in result
        assert result["prediction"] in [0, 1]
        assert result["risk_level"] in ["low", "high"]
        assert 0.0 <= result["probability"] <= 1.0
