FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY app /app/app
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

COPY alembic.ini /app/alembic.ini
COPY alembic /app/alembic
COPY data /app/data

RUN addgroup --system signflow && \
    adduser --system --ingroup signflow signflow && \
    chown -R signflow:signflow /app

USER signflow

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
