"""
Decision Policies for Phase 7 Auto-Handle vs Escalation Layer.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from hiver_agent.decision.interface import DecisionInput, DecisionResult, RiskSignalBreakdown
from hiver_agent.decision.risk_signals import RiskSignalCalculator


class BaseDecisionPolicy(ABC):
    """
    Abstract interface for decision policies.
    """
    @abstractmethod
    def evaluate(self, inp: DecisionInput) -> DecisionResult:
        """
        Evaluate a DecisionInput and return a DecisionResult.
        """
        pass


class HardRulesPolicy(BaseDecisionPolicy):
    """
    Hard Escalation Rules Evaluator.
    Checks deterministic safety, evidence, and response validity constraints.
    """
    @staticmethod
    def check_hard_rules(inp: DecisionInput) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Returns (triggered: bool, reason_code: Optional[str], reason: Optional[str])
        """
        if not inp.safety_check_passed or inp.safety_violations:
            violations_str = "; ".join(inp.safety_violations) if inp.safety_violations else "Safety check failed"
            return True, "SAFETY_CHECK_FAILED", f"Hard safety check failed: {violations_str}"

        if inp.grounding_status == "insufficient_evidence" or inp.draft_response.strip() == "INSUFFICIENT_EVIDENCE":
            return True, "INSUFFICIENT_EVIDENCE", "Retrieved evidence is insufficient to safely draft a grounded resolution."

        if not inp.retrieved_evidence:
            return True, "NO_RETRIEVABLE_EVIDENCE", "No historical support evidence was retrieved for the customer query."

        if not inp.draft_response or len(inp.draft_response.strip()) == 0:
            return True, "INVALID_RESPONSE", "Generated draft response is empty or invalid."

        return False, None, None

    def evaluate(self, inp: DecisionInput) -> DecisionResult:
        triggered, code, reason = self.check_hard_rules(inp)
        if triggered:
            return DecisionResult(
                case_id=inp.case_id,
                decision="ESCALATE_TO_HUMAN",
                risk_score=1.0,
                reason_code=code,  # type: ignore
                reason=reason,  # type: ignore
                signals={"hard_rule_triggered": True, "reason_code": code},
                recommended_action="human_review"
            )

        return DecisionResult(
            case_id=inp.case_id,
            decision="AUTO_HANDLE",
            risk_score=0.0,
            reason_code="HARD_RULES_PASSED",
            reason="Hard safety and evidence rules passed.",
            signals={"hard_rule_triggered": False},
            recommended_action="auto_send"
        )


class ConfidenceOnlyPolicy(BaseDecisionPolicy):
    """
    Baseline 1: Naive Intent Confidence Threshold Policy.
    Escalates if intent confidence is below confidence_threshold.
    """
    def __init__(self, confidence_threshold: float = 0.60):
        self.confidence_threshold = confidence_threshold

    def evaluate(self, inp: DecisionInput) -> DecisionResult:
        if inp.intent_confidence < self.confidence_threshold:
            return DecisionResult(
                case_id=inp.case_id,
                decision="ESCALATE_TO_HUMAN",
                risk_score=round(1.0 - inp.intent_confidence, 4),
                reason_code="LOW_INTENT_CONFIDENCE",
                reason=f"Intent confidence ({inp.intent_confidence:.2f}) is below threshold ({self.confidence_threshold:.2f}).",
                signals={"intent_confidence": inp.intent_confidence, "threshold": self.confidence_threshold},
                recommended_action="human_review"
            )
        
        return DecisionResult(
            case_id=inp.case_id,
            decision="AUTO_HANDLE",
            risk_score=round(1.0 - inp.intent_confidence, 4),
            reason_code="HIGH_INTENT_CONFIDENCE",
            reason=f"Intent confidence ({inp.intent_confidence:.2f}) meets or exceeds threshold ({self.confidence_threshold:.2f}).",
            signals={"intent_confidence": inp.intent_confidence, "threshold": self.confidence_threshold},
            recommended_action="auto_send"
        )


class RiskAwarePolicy(BaseDecisionPolicy):
    """
    Proposed Risk-Aware Composite Policy.
    Combines hard safety/evidence rules + multi-signal composite risk score evaluation.
    """
    def __init__(
        self,
        risk_threshold: float = 0.35,
        min_retrieval_score: float = 0.70,
        high_risk_intents: Optional[list] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.risk_threshold = risk_threshold
        self.calculator = RiskSignalCalculator(
            min_retrieval_score=min_retrieval_score,
            high_risk_intents=high_risk_intents,
            weights=weights
        )

    def evaluate(self, inp: DecisionInput) -> DecisionResult:
        # 1. Evaluate Hard Rules First
        hard_triggered, hard_code, hard_reason = HardRulesPolicy.check_hard_rules(inp)
        signals_breakdown = self.calculator.compute_signals(inp)

        if hard_triggered:
            return DecisionResult(
                case_id=inp.case_id,
                decision="ESCALATE_TO_HUMAN",
                risk_score=1.0,
                reason_code=hard_code,  # type: ignore
                reason=hard_reason,  # type: ignore
                signals={
                    "hard_rule_triggered": True,
                    "reason_code": hard_code,
                    "breakdown": signals_breakdown.model_dump()
                },
                recommended_action="human_review"
            )

        # 2. Evaluate Composite Risk Score against Risk Threshold
        composite_risk = signals_breakdown.composite_risk_score
        signals_dict = {
            "hard_rule_triggered": False,
            "intent_confidence": inp.intent_confidence,
            "top_retrieval_score": inp.top_retrieval_score,
            "evidence_count": len(inp.retrieved_evidence),
            "predicted_intent": inp.predicted_intent,
            "breakdown": signals_breakdown.model_dump()
        }

        if composite_risk > self.risk_threshold:
            # Generate explicit, interpretable reason
            primary_reason = []
            if signals_breakdown.intent_uncertainty_score > 0.40:
                primary_reason.append(f"low intent confidence ({inp.intent_confidence:.2f})")
            if signals_breakdown.retrieval_quality_score > 0.30:
                primary_reason.append(f"weak historical evidence match ({inp.top_retrieval_score:.2f})")
            if signals_breakdown.intent_risk_score > 0.40:
                primary_reason.append(f"high-risk category '{inp.predicted_intent}'")

            reason_str = f"Composite risk score ({composite_risk:.2f}) exceeds threshold ({self.risk_threshold:.2f}). Signals: " + ", ".join(primary_reason if primary_reason else ["multi-signal risk accumulation"]) + "."
            
            return DecisionResult(
                case_id=inp.case_id,
                decision="ESCALATE_TO_HUMAN",
                risk_score=composite_risk,
                reason_code="COMPOSITE_RISK_EXCEEDED",
                reason=reason_str,
                signals=signals_dict,
                recommended_action="human_review"
            )

        # 3. Safe Auto-Handle
        return DecisionResult(
            case_id=inp.case_id,
            decision="AUTO_HANDLE",
            risk_score=composite_risk,
            reason_code="SAFE_AUTO_HANDLE",
            reason=f"High-confidence intent ({inp.intent_confidence:.2f}), strong historical evidence ({inp.top_retrieval_score:.2f}), sufficient grounding, and no safety violations.",
            signals=signals_dict,
            recommended_action="auto_send"
        )
