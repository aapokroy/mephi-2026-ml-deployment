from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

from credit_default_service.model_io import load_model_artifact


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"


def create_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("credit_default_service")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(log_dir / "api_requests.jsonl", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def load_models(models_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "v1": load_model_artifact(models_dir / "model_v1.joblib"),
        "v2": load_model_artifact(models_dir / "model_v2.joblib"),
    }


def validate_payload(payload: dict[str, Any], feature_columns: list[str]) -> dict[str, float]:
    missing = [feature for feature in feature_columns if feature not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    return {feature: float(payload[feature]) for feature in feature_columns}


def predict_default(payload: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    feature_columns = artifact["feature_columns"]
    features = validate_payload(payload, feature_columns)
    frame = pd.DataFrame([features], columns=feature_columns)

    model = artifact["model"]
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "model_version": artifact["version"],
        "prediction": prediction,
        "default_probability": probability,
    }


def create_app(models_dir: Path | str | None = None) -> Flask:
    app = Flask(__name__)
    resolved_models_dir = Path(models_dir or os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR))
    models = load_models(resolved_models_dir)
    logger = create_logger(LOG_DIR)

    @app.get("/health")
    def health() -> tuple[Any, int]:
        return jsonify({"status": "ok", "models": sorted(models)}), 200

    @app.post("/predict")
    def predict() -> tuple[Any, int]:
        payload = request.get_json(silent=True) or {}
        model_version = str(payload.get("model_version", "v1"))
        features = payload.get("features")

        if model_version not in models:
            return jsonify({"error": f"Unknown model_version: {model_version}"}), 400

        if not isinstance(features, dict):
            return jsonify({"error": "Field 'features' must be an object"}), 400

        try:
            result = predict_default(features, models[model_version])
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        logger.info(json.dumps({"request": payload, "response": result}, ensure_ascii=False))
        return jsonify(result), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
