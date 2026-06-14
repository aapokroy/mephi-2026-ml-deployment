# mephi-2026-ml-deployment

Учебный проект по внедрению ML-модели. Сервис прогнозирует дефолт по кредитной
карте на датасете Default of Credit Card Clients.

## Структура

```text
mephi-2026-ml-deployment/
  src/credit_default_service/
    api.py
    data.py
    model_io.py
    train.py
  scripts/train_models.py
  tests/test_api.py
  notebooks/
  models/
  data/
  .dockerignore
  Dockerfile
  docker-compose.yml
  ARCHITECTURE.md
  AB_TEST_PLAN.md
  RESULTS.md
  requirements.txt
```

## Локальный запуск

Создание виртуального окружения:

```bash
cd mephi-2026-ml-deployment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
cd mephi-2026-ml-deployment
PYTHONPATH=src python scripts/train_models.py
PYTHONPATH=src flask --app credit_default_service.api run --host 0.0.0.0 --port 5000
```

## Запуск тестов

```bash
cd mephi-2026-ml-deployment
PYTHONPATH=src pytest
```

## Docker

```bash
cd mephi-2026-ml-deployment
docker build -t mephi-2026-ml-deployment:latest .
docker run -p 5000:5000 mephi-2026-ml-deployment:latest
```

Через Docker Compose:

```bash
docker compose up --build
```

Docker Hub:

```text
docker.io/aapokroy/mephi-2026-ml-deployment:latest
```

Образ опубликован в Docker Hub.

## API

### GET /health

```bash
curl http://localhost:5000/health
```

Пример ответа:

```json
{"models":["v1","v2"],"status":"ok"}
```

### POST /predict

Поле `model_version` необязательное. По умолчанию используется `v1`.

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_version": "v2",
    "features": {
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
      "pay_amt6": 2000
    }
  }'
```

Пример ответа:

```json
{
  "default_probability": 0.619592211928824,
  "model_version": "v2",
  "prediction": 1
}
```

## Модели

`v1` - LogisticRegression.

`v2` - RandomForestClassifier.

Модели сохраняются в `models/model_v1.joblib` и `models/model_v2.joblib`.

## Документация

Архитектура описана в `ARCHITECTURE.md`.

План A/B-теста описан в `AB_TEST_PLAN.md`.
