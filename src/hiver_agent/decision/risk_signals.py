"""
Interpretable Risk Signal Calculators for Phase 7 Decision Layer.
"""
from typing import List, Dict, Any, Optional
from hiver_agent.decision.interface import DecisionInput, RiskSignalBreakdown


class RiskSignalCalculator:
    """
    Computes individual and composite risk signals from a DecisionInput payload.
    """
    def __init__(
        self,
        min_retrieval_score: float = 0.70,
        high_risk_intents: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.min_retrieval_score = min_retrieval_score
        self.high_risk_intents = high_risk_intents or [
            "payment_billing_issue",
            "account_access_login",
            "third_party_seller_issue",
            "refund_status_request",
            "damaged_defective_item"
        ]
        self.weights = weights or {
            "intent_uncertainty": 0.40,
            "retrieval_quality": 0.30,
            "category_risk": 0.30
        }

    def compute_signals(self, inp: DecisionInput) -> RiskSignalBreakdown:
        """
        Compute normalized sub-scores and composite risk score for the input payload.
        """
        # 1. Intent Uncertainty Risk: 1.0 - confidence
        intent_uncertainty = max(0.0, min(1.0, 1.0 - inp.intent_confidence))

        # 2. Retrieval Quality Risk
        if not inp.retrieved_evidence or inp.top_retrieval_score <= 0.0:
            retrieval_quality_risk = 1.0
        else:
            # Score penalty if top retrieval similarity is below min threshold
            if inp.top_retrieval_score >= self.min_retrieval_score:
                retrieval_quality_risk = max(0.0, (1.0 - inp.top_retrieval_score) * 0.5)
            else:
                retrieval_quality_risk = min(1.0, 0.5 + (self.min_retrieval_score - inp.top_retrieval_score))

        # 3. Evidence Sufficiency Risk
        if (
            inp.grounding_status == "insufficient_evidence"
            or inp.draft_response == "INSUFFICIENT_EVIDENCE"
            or not inp.retrieved_evidence
        ):
            evidence_sufficiency_risk = 1.0
        elif len(inp.retrieved_evidence) == 1:
            evidence_sufficiency_risk = 0.3
        else:
            evidence_sufficiency_risk = 0.0

        # 4. Intent Category Risk
        if inp.predicted_intent in self.high_risk_intents:
            intent_risk = 0.60
        else:
            intent_risk = 0.10

        # 5. Composite Risk Score Calculation
        w_intent = self.weights.get("intent_uncertainty", 0.40)
        w_retrieval = self.weights.get("retrieval_quality", 0.30)
        w_category = self.weights.get("category_risk", 0.30)
        total_w = w_intent + w_retrieval + w_category

        composite_risk = (
            (w_intent * intent_uncertainty) +
            (w_retrieval * retrieval_quality_risk) +
            (w_category * intent_risk)
        ) / total_w if total_w > 0 else 0.5

        # If evidence sufficiency risk is 1.0, composite risk floor is set high
        if evidence_sufficiency_risk >= 1.0:
            composite_risk = max(composite_risk, 0.90)

        return RiskSignalBreakdown(
            intent_uncertainty_score=round(intent_uncertainty, 4),
            retrieval_quality_score=round(retrieval_quality_risk, 4),
            evidence_sufficiency_score=round(evidence_sufficiency_risk, 4),
            intent_risk_score=round(intent_risk, 4),
            composite_risk_score=round(min(1.0, composite_risk), 4)
        )
