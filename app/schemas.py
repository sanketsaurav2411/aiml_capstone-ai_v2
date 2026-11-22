from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] | None = None


class SourceDocument(BaseModel):
    id: str | None = None
    content: str | None = None
    score: float | None = None
    metadata: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    answer: str
    query: str
    collection: str | None = None
    sources: list[SourceDocument] = Field(default_factory=list)
