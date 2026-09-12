"""
Pydantic API request and response schemas for Hiver Support Agent REST API.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


ALLOWED_POLICY_TYPES = {"risk_aware", "auto_handle_first", "escalate_all"}


class SupportProcessRequest(BaseModel):
    """
    Request payload for end-to-end customer message processing.
    """
    customer_message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The customer inquiry message text."
    )
    case_id: Optional[str] = Field(
        default=None,
        description="Optional unique identifier for the customer case."
    )
    policy_type: str = Field(
        default="risk_aware",
        description="Decision engine policy type. Defaults to 'risk_aware'."
    )

    @field_validator("customer_message")
    @classmethod
    def validate_message_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("customer_message cannot be empty or whitespace-only.")
        return stripped

    @field_validator("policy_type")
    @classmethod
    def validate_policy_type(cls, v: str) -> str:
        if v not in ALLOWED_POLICY_TYPES:
            raise ValueError(f"Invalid policy_type '{v}'. Allowed values: {sorted(list(ALLOWED_POLICY_TYPES))}")
        return v


class SupportProcessResponse(BaseModel):
    """
    Unified response payload for end-to-end customer support processing.
    """
    case_id: Optional[str] = None
    customer_message: str
    predicted_intent: str
    intent_confidence: float
    decision: str = Field(..., description="'AUTO_HANDLE' or 'ESCALATE_TO_HUMAN'")
    reason_code: str = Field(..., description="Escalation trigger reason code")
    decision_reason: str = Field(..., description="Natural language decision rationale")
    draft_response: str = Field(..., description="Generated customer support response text")
    risk_score: float = Field(..., description="Composite risk score (0.0 to 1.0)")
    safety_check_passed: bool
    grounding_status: str
    selected_evidence_count: int
    retrieved_case_ids: List[str] = Field(default_factory=list)
    retrieved_scores: List[float] = Field(default_factory=list)
    signals: Dict[str, Any] = Field(default_factory=dict)
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="LLM provider metadata (real vs mock/template)")
    latency_ms: float = Field(..., description="API request processing latency in milliseconds")
    request_id: Optional[str] = Field(default=None, description="Unique correlation request ID")


class HealthResponse(BaseModel):
    """
    Response schema for GET /api/v1/health (Liveness).
    """
    status: str = Field(..., description="'ok' or 'error'")
    service: str = Field(default="hiver-ai-support-agent")
    pipeline_ready: bool = Field(..., description="True if pipeline singleton is loaded and ready")
    version: str = Field(default="0.1.0")
    request_id: Optional[str] = Field(default=None, description="Unique correlation request ID")


class ReadyResponse(BaseModel):
    """
    Response schema for GET /api/v1/ready (Readiness).
    """
    status: str = Field(..., description="'ready' or 'not_ready'")
    service: str = Field(default="hiver-ai-support-agent")
    pipeline_ready: bool = Field(..., description="True if pipeline singleton is initialized")
    vector_store_documents: int = Field(default=0, description="Total indexed cases in vector store")
    version: str = Field(default="0.1.0")
    request_id: Optional[str] = Field(default=None, description="Unique correlation request ID")


class MetricDetail(BaseModel):
    value: float
    evaluation_set: str
    n_examples: int
    note: Optional[str] = None


class StatsResponse(BaseModel):
    """
    Response schema for GET /api/v1/stats exposing validated project metrics.
    """
    service: str = "hiver-ai-support-agent"
    version: str = "0.1.0"
    target_brand: str = "AmazonHelp"
    canonical_corpus_size: int = 15316
    raw_conversation_turns: int = 81186
    intent_taxonomy_count: int = 17
    active_intent_classifier: str = "CalibratedRulePrototypeHybrid"
    active_retrieval_mode: str = "Hybrid Dense + BM25 (RRF k=60 + Intent Compatibility)"
    active_llm_model: str = "gemini-1.5-flash"
    active_risk_threshold: float = 0.35
    metrics: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = Field(default=None, description="Unique correlation request ID")


class RetrievalSearchRequest(BaseModel):
    """
    Request schema for POST /api/v1/retrieval/search.
    """
    query: str = Field(..., min_length=1, max_length=1000, description="Search query string.")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of historical cases to retrieve.")

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query cannot be empty or whitespace-only.")
        return stripped


class RetrievalSearchResult(BaseModel):
    case_id: str
    customer_problem: str
    score: float
    intent: Optional[str] = None


class RetrievalSearchResponse(BaseModel):
    query: str
    total_retrieved: int
    results: List[RetrievalSearchResult]
    latency_ms: float
    request_id: Optional[str] = Field(default=None, description="Unique correlation request ID")


class APIErrorDetail(BaseModel):
    code: str
    message: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


class APIErrorResponse(BaseModel):
    error: APIErrorDetail
