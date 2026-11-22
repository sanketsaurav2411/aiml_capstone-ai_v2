from fastapi import APIRouter, Depends, HTTPException

from app.config import APP_TITLE, VECTOR_DB_BASE_URL
from app.core.errors import VectorDBConnectionError
from app.schemas import QueryRequest, QueryResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["rag"])


def get_rag_service() -> RAGService:
    return RAGService()


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": APP_TITLE,
        "vector_db_url": VECTOR_DB_BASE_URL,
    }


@router.post("/query", response_model=QueryResponse)
def ask_question(
    payload: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    try:
        return rag_service.answer_query(
            query=payload.query,
            collection=payload.collection,
            top_k=payload.top_k,
            filters=payload.filters,
        )
    except VectorDBConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
