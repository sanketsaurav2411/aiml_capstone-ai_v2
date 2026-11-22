FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001 \
    CHROMA_PERSIST_DIR=/tmp/chroma_db \
    SQLITE_DB_PATH=/tmp/ai_sessions.sqlite

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts

RUN sed -i 's/\r$//' ./scripts/entrypoint.sh && \
    chmod +x ./scripts/entrypoint.sh && \
    useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app /tmp

USER appuser

EXPOSE 8001

ENTRYPOINT ["./scripts/entrypoint.sh"]
