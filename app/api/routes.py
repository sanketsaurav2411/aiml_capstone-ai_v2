from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from app.config import APP_TITLE, CHROMA_PERSIST_DIR, INTERNAL_API_KEY, VECTOR_DB_BASE_URL
from app.core.errors import VectorDBConnectionError
from app.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    DocumentParseResponse,
    MetricsSummaryResponse,
    QueryRequest,
    QueryResponse,
    SessionStateResponse,
)
from app.services.customs_agent import CustomsAgentService
from app.services.document_parser import parse_document_bytes
from app.services.rag_service import RAGService
from app.services.vector_db_client import VectorDBClient

router = APIRouter(prefix="/api/v1", tags=["rag"])


def verify_internal_auth(x_internal_api_key: str | None = Header(default=None)) -> None:
    if INTERNAL_API_KEY and x_internal_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing internal service API key")


@lru_cache(maxsize=1)
def get_vector_db_client() -> VectorDBClient:
    return VectorDBClient()


def get_rag_service() -> RAGService:
    return RAGService(vector_db_client=get_vector_db_client())


@lru_cache(maxsize=1)
def get_customs_agent_service() -> CustomsAgentService:
    return CustomsAgentService(vector_db_client=get_vector_db_client())


@router.get("/health")
def health() -> dict[str, object]:
    vdb = get_vector_db_client()
    return {
        "status": "ok",
        "service": APP_TITLE,
        "vector_db_url": VECTOR_DB_BASE_URL or "local-sqlite-chroma",
        "chroma_dir": CHROMA_PERSIST_DIR,
        "pipeline_version": vdb.pipeline_version,
        "embed_model": vdb.embed_model_name,
        "collections": vdb.collection_counts,
    }


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_internal_auth)])
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


@router.post("/agent/chat", response_model=AgentChatResponse, dependencies=[Depends(verify_internal_auth)])
def agent_chat(
    payload: AgentChatRequest,
    agent_service: CustomsAgentService = Depends(get_customs_agent_service),
) -> AgentChatResponse:
    return agent_service.handle_chat(payload)


@router.get(
    "/agent/session/{thread_id}",
    response_model=SessionStateResponse,
    dependencies=[Depends(verify_internal_auth)],
)
def get_agent_session(
    thread_id: str,
    agent_service: CustomsAgentService = Depends(get_customs_agent_service),
) -> SessionStateResponse:
    return agent_service.get_session(thread_id)


@router.post(
    "/documents/parse",
    response_model=DocumentParseResponse,
    dependencies=[Depends(verify_internal_auth)],
)
async def parse_uploaded_document(file: UploadFile = File(...)) -> DocumentParseResponse:
    raw = await file.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Document exceeds 5 MB limit")
    result = parse_document_bytes(
        filename=file.filename or "document",
        raw_bytes=raw,
        mimetype=file.content_type or "application/octet-stream",
    )
    return DocumentParseResponse(
        filename=result["filename"],
        mimetype=result["mimetype"],
        pages_extracted=result["pages_extracted"],
        sanitized_text=result["sanitized_text"],
        warnings=result["warnings"],
    )


@router.get(
    "/metrics/summary",
    response_model=MetricsSummaryResponse,
    dependencies=[Depends(verify_internal_auth)],
)
def metrics_summary(
    agent_service: CustomsAgentService = Depends(get_customs_agent_service),
) -> MetricsSummaryResponse:
    return agent_service.get_metrics_summary()
