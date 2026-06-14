from __future__ import annotations

from pathlib import Path

from credit_default_service.train import train_and_save_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metrics = train_and_save_models(
        data_dir=PROJECT_ROOT / "data" / "raw" / "openml",
        models_dir=PROJECT_ROOT / "models",
    )
    for version, values in metrics.items():
        print(version)
        for metric, value in values.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()

