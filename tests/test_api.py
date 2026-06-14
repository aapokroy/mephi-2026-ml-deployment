from __future__ import annotations

from pathlib import Path

from credit_default_service.api import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXAMPLE_PAYLOAD = {
    "limit_bal": 120000,
    "sex": 2,
    "education": 2,
    "marriage": 2,
    "age": 26,
    "pay_0": -1,
    "pay_2": 2,
    "pay_3": 0,
    "pay_4": 0,
    "pay_5": 0,
    "pay_6": 2,
    "bill_amt1": 2682,
    "bill_amt2": 1725,
    "bill_amt3": 2682,
    "bill_amt4": 3272,
    "bill_amt5": 3455,
    "bill_amt6": 3261,
    "pay_amt1": 0,
    "pay_amt2": 1000,
    "pay_amt3": 1000,
    "pay_amt4": 1000,
    "pay_amt5": 0,
    "pay_amt6": 2000,
}


def test_health() -> None:
    app = create_app(PROJECT_ROOT / "models")
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_v1() -> None:
    app = create_app(PROJECT_ROOT / "models")
    client = app.test_client()
    response = client.post(
        "/predict",
        json={"features": EXAMPLE_PAYLOAD, "model_version": "v1"},
    )
    data = response.get_json()

    assert response.status_code == 200
    assert data["model_version"] == "v1"
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["default_probability"] <= 1.0


def test_predict_v2() -> None:
    app = create_app(PROJECT_ROOT / "models")
    client = app.test_client()
    response = client.post(
        "/predict",
        json={"features": EXAMPLE_PAYLOAD, "model_version": "v2"},
    )
    data = response.get_json()

    assert response.status_code == 200
    assert data["model_version"] == "v2"
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["default_probability"] <= 1.0
