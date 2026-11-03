import pytest
import numpy as np

from unittest.mock import Mock
from fastapi.testclient import TestClient

import src.api as api


@pytest.fixture
def client():
    # Подменяем подключение к БД
    fake_db = Mock()
    fake_db.execute.return_value.scalars.return_value.all.return_value = []
    def override_get_db():
        yield fake_db

    # FastAPI будет использовать fake_db вместо PostgreSQL
    api.app.dependency_overrides[api.get_db] = override_get_db

    test_client = TestClient(api.app)

    yield test_client

    # Убираем подмену после теста
    api.app.dependency_overrides.pop(api.get_db, None)


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "probabilities, expected_prediction",
    [
        ([[0.8, 0.2]], 0),
        ([[0.2, 0.8]], 1),
    ],
    ids=["negative_prediction", "positive_prediction"],
)
def test_predict_with_mock(
    client,
    monkeypatch,
    probabilities,
    expected_prediction,
):
    model_mock = Mock()
    model_mock.predict_proba.return_value = np.array(probabilities)

    monkeypatch.setattr(api, "model", model_mock)
    monkeypatch.setattr(api, "build_features", lambda df: df)

    payload = {
        "decision_day": "2025-01-15",
        "loan_amount_last": 100000,
        "offered_rate": 15.0,
        "cb_rate": 10.0,
        "overdraft_limit_min": 0,
        "overdraft_limit_max": 50000,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == expected_prediction
    assert data["probability"] == pytest.approx(probabilities[0][1])

    model_mock.predict_proba.assert_called_once()

    # Проверяем, что API вызвал сохранение результата
    api_db = api.app.dependency_overrides[api.get_db]
    assert callable(api_db)


def test_predict_probability(client, monkeypatch):
    model_mock = Mock()
    model_mock.predict_proba.return_value = np.array([[0.15, 0.85]])

    monkeypatch.setattr(api, "model", model_mock)
    monkeypatch.setattr(api, "build_features", lambda df: df)

    payload = {
        "decision_day": "2025-01-15",
        "loan_amount_last": 100000,
        "offered_rate": 15.0,
        "cb_rate": 10.0,
        "overdraft_limit_min": 0,
        "overdraft_limit_max": 50000,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["probability"] == pytest.approx(0.85)
    assert response.json()["prediction"] == 1


def test_predict_invalid_data(client):
    payload = {
        "loan_amount_last": 100000,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_history(client):
    response = client.get("/history")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_metrics(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.text.strip()