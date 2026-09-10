"""
Structured generation schemas and interface definitions for Phase 6.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """
    Structured representation of a single retrieved historical support case evidence.
    """
    case_id: str = Field(..., description="Unique historical case ID")
    customer_problem: str = Field(..., description="Customer initial problem description")
    support_interaction: str = Field(..., description="Support interaction body")
    resolution: str = Field(..., description="Final support resolution response text")
    intent: str = Field(..., description="Historical case intent label")
    score: float = Field(0.0, description="Retrieval similarity score")


class GenerationResult(BaseModel):
    """
    Strict response-generation contract for Phase 6 grounded generation.
    """
    case_id: Optional[str] = Field(None, description="Optional evaluation case ID")
    customer_message: str = Field(..., description="Target customer query message")
    predicted_intent: str = Field(..., description="Predicted customer intent label")
    intent_confidence: float = Field(..., description="Confidence score of predicted intent [0.0, 1.0]")
    retrieval_mode: str = Field(..., description="Retrieval mode used: 'intent_aware' or 'semantic_fallback'")
    retrieved_case_ids: List[str] = Field(default_factory=list, description="IDs of cases selected as evidence")
    retrieved_evidence: List[EvidenceItem] = Field(default_factory=list, description="Formatted evidence items")
    draft_response: str = Field(..., description="Final drafted customer support response")
    grounding_status: str = Field(..., description="Grounding status: 'grounded' or 'insufficient_evidence'")
    safety_check_passed: bool = Field(True, description="Whether the response passed grounding safety validation")
    safety_violations: List[str] = Field(default_factory=list, description="List of safety violation messages if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional generation metadata")


class BaseResponseGenerator(ABC):
    """
    Abstract interface for grounded response generation.
    """
    @abstractmethod
    def generate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieval_mode: str,
        evidence: List[EvidenceItem],
        case_id: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate a grounded customer support response given the customer message and selected evidence.
        """
        pass
