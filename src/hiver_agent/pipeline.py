"""
End-to-end customer support agent pipeline.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from hiver_agent.classification.interface import IntentClassifier
from hiver_agent.retrieval.interfaces import Retriever
from hiver_agent.retrieval.retriever import FAISSRetriever, IntentAwareRetriever, SearchResult
from hiver_agent.generation.interface import BaseResponseGenerator, EvidenceItem, GenerationResult
from hiver_agent.generation.evidence import EvidenceFormatter
from hiver_agent.generation.evidence_selector import EvidenceSelector
from hiver_agent.generation.safety import GroundingSafetyChecker
from hiver_agent.generation.generator import TemplateResponseGenerator, MockLLMGenerator

from hiver_agent.decision.decision_engine import RiskAwareDecisionEngine
from hiver_agent.decision.interface import DecisionResult, DecisionInput


class EndToEndResult(BaseModel):
    """
    Unified end-to-end result contract for evaluation, debugging, and customer response.
    """
    case_id: Optional[str] = None
    customer_message: str
    predicted_intent: str
    intent_confidence: float
    retrieval_mode: str
    retrieved_case_ids: List[str] = Field(default_factory=list)
    retrieval_scores: List[float] = Field(default_factory=list)
    selected_evidence_count: int = 0
    draft_response: str
    grounding_status: str
    safety_check_passed: bool
    safety_violations: List[str] = Field(default_factory=list)
    risk_score: float
    decision: str
    reason_code: str
    decision_reason: str
    recommended_action: str
    signals: Dict[str, Any] = Field(default_factory=dict)
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_generation_and_decision(
        cls,
        gen_res: GenerationResult,
        dec_res: DecisionResult
    ) -> "EndToEndResult":
        retrieved_ids = [ev.case_id for ev in gen_res.retrieved_evidence] if gen_res.retrieved_evidence else []
        retrieved_scores = [round(ev.score, 4) for ev in gen_res.retrieved_evidence] if gen_res.retrieved_evidence else []

        return cls(
            case_id=gen_res.case_id,
            customer_message=gen_res.customer_message,
            predicted_intent=gen_res.predicted_intent,
            intent_confidence=round(gen_res.intent_confidence, 4),
            retrieval_mode=gen_res.retrieval_mode,
            retrieved_case_ids=retrieved_ids,
            retrieval_scores=retrieved_scores,
            selected_evidence_count=len(gen_res.retrieved_evidence) if gen_res.retrieved_evidence else 0,
            draft_response=gen_res.draft_response,
            grounding_status=gen_res.grounding_status,
            safety_check_passed=gen_res.safety_check_passed,
            safety_violations=gen_res.safety_violations,
            risk_score=dec_res.risk_score,
            decision=dec_res.decision,
            reason_code=dec_res.reason_code,
            decision_reason=dec_res.reason,
            recommended_action=dec_res.recommended_action,
            signals=dec_res.signals.model_dump() if hasattr(dec_res.signals, "model_dump") else dict(dec_res.signals),
            generation_metadata=getattr(gen_res, "metadata", {})
        )


class CustomerSupportPipeline:
    """
    End-to-end customer support RAG pipeline.
    Combines Intent Classification -> Confidence Gating -> Retrieval -> Evidence Selection -> Response Generation -> Safety Validation -> Decision Engine.
    """
    def __init__(
        self,
        classifier: IntentClassifier,
        base_retriever: FAISSRetriever,
        generator: BaseResponseGenerator,
        confidence_threshold: float = 0.60,
        top_k_retrieve: int = 10,
        top_k_evidence: int = 3,
        safety_checker: Optional[GroundingSafetyChecker] = None,
        decision_engine: Optional[RiskAwareDecisionEngine] = None
    ):
        self.classifier = classifier
        self.base_retriever = base_retriever
        self.generator = generator
        self.confidence_threshold = confidence_threshold
        self.top_k_retrieve = top_k_retrieve
        self.top_k_evidence = top_k_evidence
        self.safety_checker = safety_checker or GroundingSafetyChecker()
        self.decision_engine = decision_engine or RiskAwareDecisionEngine()

        self.intent_aware_retriever = IntentAwareRetriever(
            embedding_provider=base_retriever.embedding_provider,
            vector_store=base_retriever.vector_store,
            classifier=classifier,
            confidence_threshold=confidence_threshold
        )
        self.evidence_selector = EvidenceSelector(
            max_evidence=top_k_evidence,
            confidence_threshold=confidence_threshold
        )

    def process_message(
        self,
        customer_message: str,
        case_id: Optional[str] = None,
        exclude_case_id: Optional[str] = None
    ) -> GenerationResult:
        """
        Process a single customer message through generation and safety validation.
        """
        # 1. Intent Classification
        pred = self.classifier.predict(customer_message)

        # 2. Confidence Gating & Retrieval Mode Selection
        if pred.confidence >= self.confidence_threshold:
            retrieval_mode = "intent_aware"
            raw_results = self.intent_aware_retriever.retrieve(customer_message, top_k=self.top_k_retrieve)
        else:
            retrieval_mode = "semantic_fallback"
            raw_results = self.base_retriever.retrieve(customer_message, top_k=self.top_k_retrieve)

        # 3. Exclude target self-case from evidence to prevent evaluation data leakage
        filtered_results: List[SearchResult] = []
        target_exclude = exclude_case_id or case_id

        for r in raw_results:
            if target_exclude and str(r.case_id).strip() == str(target_exclude).strip():
                continue
            # Also exclude if customer problem text is identical
            if customer_message.strip().lower() == r.customer_problem.strip().lower():
                continue
            filtered_results.append(r)

        # 4. Format & Select Evidence
        all_evidence = EvidenceFormatter.format_search_results(filtered_results)
        selected_evidence = self.evidence_selector.select_evidence(
            all_evidence,
            predicted_intent=pred.intent,
            intent_confidence=pred.confidence
        )

        # 5. Generate Response
        result = self.generator.generate(
            customer_message=customer_message,
            predicted_intent=pred.intent,
            intent_confidence=pred.confidence,
            retrieval_mode=retrieval_mode,
            evidence=selected_evidence,
            case_id=case_id
        )

        return result

    def process_message_with_decision(
        self,
        customer_message: str,
        case_id: Optional[str] = None,
        exclude_case_id: Optional[str] = None,
        policy_type: str = "risk_aware"
    ) -> tuple[GenerationResult, DecisionResult]:
        """
        Process a customer message through generation and execute Phase 7 decision engine.
        """
        gen_res = self.process_message(customer_message, case_id=case_id, exclude_case_id=exclude_case_id)
        dec_res = self.decision_engine.decide_from_generation_result(gen_res, policy_type=policy_type)
        return gen_res, dec_res

    def process_end_to_end(
        self,
        customer_message: str,
        case_id: Optional[str] = None,
        exclude_case_id: Optional[str] = None,
        policy_type: str = "risk_aware"
    ) -> EndToEndResult:
        """
        Public End-to-End API returning unified EndToEndResult.
        """
        gen_res, dec_res = self.process_message_with_decision(
            customer_message=customer_message,
            case_id=case_id,
            exclude_case_id=exclude_case_id,
            policy_type=policy_type
        )
        return EndToEndResult.from_generation_and_decision(gen_res, dec_res)
