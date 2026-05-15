FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

COPY requirements.base.txt .
COPY requirements.prod.txt .

RUN pip install --no-cache-dir -r requirements.prod.txt

COPY . .

EXPOSE 7860

CMD python scripts/download_artifacts.py && \
    gunicorn --bind 0.0.0.0:7860 "app:create_app()"