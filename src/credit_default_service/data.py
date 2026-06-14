from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


DATASET_ID = 42477
TARGET_COLUMN = "default_payment_next_month"

FEATURE_COLUMNS = [
    "limit_bal",
    "sex",
    "education",
    "marriage",
    "age",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
]


def load_credit_default_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(
        data_id=DATASET_ID,
        as_frame=True,
        data_home=str(data_dir),
        parser="auto",
    )
    frame = dataset.frame.copy()
    frame.columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    features = frame[FEATURE_COLUMNS].astype(float)
    target = frame[TARGET_COLUMN].astype(int)
    return features, target

