from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_default_service.data import FEATURE_COLUMNS, load_credit_default_data
from credit_default_service.model_io import save_model_artifact


RANDOM_STATE = 42


def make_logistic_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_random_forest_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight="balanced_subsample",
        max_depth=8,
        min_samples_leaf=20,
        n_estimators=150,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def calculate_metrics(model: object, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "f1": f1_score(target, predictions),
        "precision": precision_score(target, predictions),
        "recall": recall_score(target, predictions),
        "roc_auc": roc_auc_score(target, probabilities),
    }


def train_and_save_models(data_dir: Path, models_dir: Path) -> dict[str, dict[str, float]]:
    features, target = load_credit_default_data(data_dir)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    models = {
        "v1": make_logistic_model(),
        "v2": make_random_forest_model(),
    }

    metrics: dict[str, dict[str, float]] = {}
    for version, model in models.items():
        model.fit(x_train, y_train)
        metrics[version] = calculate_metrics(model, x_test, y_test)
        artifact = {
            "version": version,
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics[version],
            "positive_class": "default",
        }
        save_model_artifact(artifact, models_dir / f"model_{version}.joblib")

    return metrics

