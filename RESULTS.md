# Результаты запуска

## Обучение

Было обучено две версии модели.

`v1` - LogisticRegression:

```text
f1: 0.4613
precision: 0.3672
recall: 0.6202
roc_auc: 0.7081
```

`v2` - RandomForestClassifier:

```text
f1: 0.5409
precision: 0.4956
recall: 0.5953
roc_auc: 0.7743
```

## Тесты

```text
3 passed in 1.42s
```

## Проверка API

`GET /health`:

```json
{
  "models": [
    "v1",
    "v2"
  ],
  "status": "ok"
}
```

`POST /predict`, модель `v1`:

```json
{
  "default_probability": 0.7754788294950143,
  "model_version": "v1",
  "prediction": 1
}
```

`POST /predict`, модель `v2`:

```json
{
  "default_probability": 0.7722370096332974,
  "model_version": "v2",
  "prediction": 1
}
```

Файл логов создаётся по пути `logs/api_requests.jsonl` после первого запроса.

## Docker

Образ был собран локально:

```text
docker build -t mephi-2026-ml-deployment:latest .
```

Образ был опубликован в Docker Hub:

```text
docker.io/aapokroy/mephi-2026-ml-deployment:latest
sha256:2ad86c6cd7d71762e7b20a70289514172694f324e9bc07132861d9cd905eebce
```

Контейнер был запущен на порту `5050`. Проверка `/health` вернула:

```json
{"models":["v1","v2"],"status":"ok"}
```
