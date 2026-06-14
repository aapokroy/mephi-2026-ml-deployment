FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models

ENV PYTHONPATH=/app/src
ENV FLASK_APP=credit_default_service.api:app
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "credit_default_service.api:app"]

