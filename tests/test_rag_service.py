from app.main import app
from app.api.routes import get_rag_service
from app.services.rag_service import RAGService
from fastapi.testclient import TestClient


class FakeVectorDBClient:
    def search(self, query, collection=None, top_k=5, filters=None):
        return [
            {
                "id": "doc-1",
                "content": "Retrieval-Augmented Generation combines search and generation to answer questions using relevant documents.",
                "score": 0.98,
                "metadata": {"source": "knowledge-base"},
            }
        ]


class FakeAnswerGenerator:
    def generate(self, query, documents):
        return f"Using the retrieved context, the answer is: {documents[0]['content']}"


app.dependency_overrides[get_rag_service] = lambda: RAGService(
    vector_db_client=FakeVectorDBClient(),
    answer_generator=FakeAnswerGenerator(),
)

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint():
    response = client.post(
        "/api/v1/query",
        json={"query": "What is RAG?", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "RAG" in payload["answer"] or "retrieved context" in payload["answer"]
    assert len(payload["sources"]) == 1
