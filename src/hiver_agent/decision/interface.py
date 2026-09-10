"""
Interface definitions and Pydantic contracts for Phase 7 Decision & Escalation Layer.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from hiver_agent.generation.interface import EvidenceItem, GenerationResult


class DecisionInput(BaseModel):
    """
    Structured input payload for the decision engine, aggregating all upstream signals.
    """
    case_id: Optional[str] = Field(None, description="Optional target evaluation case ID")
    customer_message: str = Field(..., description="Original customer query text")
    predicted_intent: str = Field(..., description="Predicted intent label")
    intent_confidence: float = Field(..., description="Confidence of predicted intent [0.0, 1.0]")
    retrieval_mode: str = Field(..., description="Retrieval mode: 'intent_aware' or 'semantic_fallback'")
    retrieved_case_ids: List[str] = Field(default_factory=list, description="IDs of retrieved historical evidence cases")
    retrieved_evidence: List[EvidenceItem] = Field(default_factory=list, description="List of evidence items used for generation")
    top_retrieval_score: float = Field(0.0, description="Similarity score of top-1 evidence item")
    avg_retrieval_score: float = Field(0.0, description="Average similarity score of retrieved evidence")
    draft_response: str = Field(..., description="Drafted customer support response")
    grounding_status: str = Field(..., description="Grounding status ('grounded' or 'insufficient_evidence')")
    safety_check_passed: bool = Field(True, description="Whether the response passed grounding safety validation")
    safety_violations: List[str] = Field(default_factory=list, description="List of safety violations if any")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Upstream metadata (provider, model, etc.)")

    @classmethod
    def from_generation_result(cls, gen_res: GenerationResult) -> "DecisionInput":
        """
        Factory method to construct DecisionInput directly from a GenerationResult object.
        """
        evidence = gen_res.retrieved_evidence
        top_score = evidence[0].score if evidence else 0.0
        avg_score = sum(e.score for e in evidence) / len(evidence) if evidence else 0.0

        return cls(
            case_id=gen_res.case_id,
            customer_message=gen_res.customer_message,
            predicted_intent=gen_res.predicted_intent,
            intent_confidence=gen_res.intent_confidence,
            retrieval_mode=gen_res.retrieval_mode,
            retrieved_case_ids=gen_res.retrieved_case_ids,
            retrieved_evidence=gen_res.retrieved_evidence,
            top_retrieval_score=round(top_score, 4),
            avg_retrieval_score=round(avg_score, 4),
            draft_response=gen_res.draft_response,
            grounding_status=gen_res.grounding_status,
            safety_check_passed=gen_res.safety_check_passed,
            safety_violations=gen_res.safety_violations,
            metadata=gen_res.metadata
        )


class RiskSignalBreakdown(BaseModel):
    """
    Sub-scores breaking down individual risk components.
    """
    intent_uncertainty_score: float = Field(..., description="1.0 - intent_confidence [0.0, 1.0]")
    retrieval_quality_score: float = Field(..., description="Risk derived from retrieval similarity & evidence consistency [0.0, 1.0]")
    evidence_sufficiency_score: float = Field(..., description="Risk derived from evidence count and grounding status [0.0, 1.0]")
    intent_risk_score: float = Field(..., description="Inherent category risk score [0.0, 1.0]")
    composite_risk_score: float = Field(..., description="Weighted composite risk score [0.0, 1.0]")


class DecisionResult(BaseModel):
    """
    Explainable decision outcome contract for auto-handle vs escalation.
    """
    case_id: Optional[str] = Field(None, description="Optional evaluation case ID")
    decision: str = Field(..., description="Decision outcome: 'AUTO_HANDLE' or 'ESCALATE_TO_HUMAN'")
    risk_score: float = Field(..., description="Overall composite risk score [0.0, 1.0]")
    reason_code: str = Field(..., description="Machine-readable reason code")
    reason: str = Field(..., description="Human-readable justification for the decision")
    signals: Dict[str, Any] = Field(default_factory=dict, description="Detailed dictionary of risk signals and sub-scores")
    recommended_action: str = Field(..., description="Recommended action: 'auto_send' or 'human_review'")
