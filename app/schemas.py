from typing import Any, Literal

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


class DocumentAttachmentInput(BaseModel):
    name: str
    mimetype: str = "application/octet-stream"
    size: int = 0
    extracted_text: str | None = None
    content_base64: str | None = None


class ChatTurnMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str | None = None


class AgentChatRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=128)
    user_id: str | None = None
    message: str = Field(default="", max_length=8000)
    attachments: list[DocumentAttachmentInput] = Field(default_factory=list)
    history: list[ChatTurnMessage] = Field(default_factory=list)


class ClassifiedItem(BaseModel):
    line: int = 1
    product_name: str
    chapter_2_digit: str = ""
    heading_4_digit: str = ""
    subheading_6_digit: str = ""
    code_8d: str
    lineage_path: str = ""
    schedule_duty_rate: str = ""
    unit: str = ""
    gri_applied: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ClassificationDetail(BaseModel):
    items: list[ClassifiedItem] = Field(default_factory=list)
    section_chapter_notes_reasoning: str = ""
    precedents_reasoning: str = ""
    step_by_step_reasoning: list[str] = Field(default_factory=list)
    conflict_resolution_note: str | None = None


class CitationItem(BaseModel):
    source_type: str
    id: str
    title: str
    snippet: str
    authority: str | None = None
    authority_rank: int | None = None
    score: float | None = None


class TelemetryMetrics(BaseModel):
    latency_ms: int
    retrieval_ms: int = 0
    llm_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_inr: float = 0.0
    embed_model: str = ""
    llm_model: str = ""
    pipeline_version: str = "v0_3"


class AgentChatResponse(BaseModel):
    thread_id: str
    action: Literal["classify", "classify_each_line", "clarify", "abstain", "followup_qa"]
    reply: str
    tariff_version: str
    needs_clarification: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    active_product_profile: dict[str, Any] = Field(default_factory=dict)
    classification: ClassificationDetail | None = None
    citations: list[CitationItem] = Field(default_factory=list)
    telemetry: TelemetryMetrics


class SessionStateResponse(BaseModel):
    thread_id: str
    messages: list[ChatTurnMessage] = Field(default_factory=list)
    active_product_profile: dict[str, Any] = Field(default_factory=dict)
    last_classification: ClassificationDetail | None = None
    updated_at: str | None = None


class DocumentParseResponse(BaseModel):
    filename: str
    mimetype: str
    pages_extracted: int
    sanitized_text: str
    warnings: list[str] = Field(default_factory=list)


class MetricsSummaryResponse(BaseModel):
    total_queries: int
    action_counts: dict[str, int] = Field(default_factory=dict)
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_cost_inr: float = 0.0
    total_cost_inr: float = 0.0
    abstention_rate_pct: float = 0.0
    clarification_rate_pct: float = 0.0
    chroma_dir: str = ""
    pipeline_version: str = "v0_3"
    embed_model: str = ""
    collection_counts: dict[str, int] = Field(default_factory=dict)
