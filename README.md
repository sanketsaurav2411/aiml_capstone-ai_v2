# Capstone AI RAG API

A modular FastAPI service that answers user questions by querying a vector database running at `http://localhost:8000` and synthesizing a response from the most relevant retrieved documents.

## Features

- FastAPI-based REST API
- Vector DB client with fallback search endpoint detection
- Modular RAG pipeline with separate services for vector search and answer generation
- Configurable collection and result count
- Health endpoint and query endpoint
- Easy local development with `.env` overrides

## Quick start

1. Create and activate a virtual environment:

```bash
cd /Users/kshitiz.saxenayahoo.com/Data/work/learning/capstone/capstone-ai
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment example and adjust values if needed:

```bash
cp .env.example .env
```

4. Start Chroma DB:

```bash
chroma run --path ./data/chroma_db_large
```

5. Start the API server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

The service listens on port `8001` by default so it does not clash with the vector database running on `localhost:8000`.

## API usage

### Health check

```bash
curl http://localhost:8001/api/v1/health
```

### Ask a question

```bash
curl -X POST http://localhost:8001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main ideas in this document?",
    "collection": "pdf_rag",
    "top_k": 5
  }'
```

## Project structure

```text
capstone-ai/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── errors.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── answer_generator.py
│   │   ├── rag_service.py
│   │   └── vector_db_client.py
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   └── schemas.py
├── tests/
│   └── test_rag_service.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Notes

- If your vector database exposes a different endpoint structure, update the candidate URLs in `app/services/vector_db_client.py`.
- The service supports a local fallback answer generator when no external LLM API key is configured.
- Add your Gemini API key as `GEMINI_API_KEY` in `.env` if you want to enable Gemini-backed answer generation.
