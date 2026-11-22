from __future__ import annotations

from typing import Any

from app.config import DEFAULT_COLLECTION
from app.schemas import QueryResponse, SourceDocument
from app.services.answer_generator import AnswerGenerator
from app.services.vector_db_client import VectorDBClient


class RAGService:
    """Orchestrates vector search and answer generation."""

    def __init__(
        self,
        vector_db_client: VectorDBClient | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self.vector_db_client = vector_db_client or VectorDBClient(default_collection=DEFAULT_COLLECTION)
        self.answer_generator = answer_generator or AnswerGenerator()

    def answer_query(
        self,
        query: str,
        collection: str | None = None,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> QueryResponse:
        documents = self.vector_db_client.search(
            query=query,
            collection=collection,
            top_k=top_k,
            filters=filters,
        )

        answer = self.answer_generator.generate(query, documents)
        source_documents = [
            SourceDocument(
                id=document.get("id"),
                content=document.get("content"),
                score=document.get("score"),
                metadata=document.get("metadata"),
            )
            for document in documents
        ]

        resolved_collection = collection or getattr(self.vector_db_client, "default_collection", None)
        return QueryResponse(
            answer=answer,
            query=query,
            collection=resolved_collection,
            sources=source_documents,
        )
