"""
Unit tests for Phase 7 Decision and Escalation Layer.
"""
import pytest
from hiver_agent.generation.interface import EvidenceItem, GenerationResult
from hiver_agent.decision.interface import DecisionInput, DecisionResult, RiskSignalBreakdown
from hiver_agent.decision.risk_signals import RiskSignalCalculator
from hiver_agent.decision.policy import HardRulesPolicy, ConfidenceOnlyPolicy, RiskAwarePolicy
from hiver_agent.decision.decision_engine import RiskAwareDecisionEngine
from hiver_agent.config import DecisionConfig


def test_decision_input_and_result_schema():
    """Test DecisionInput and DecisionResult instantiation and serialization."""
    ev = EvidenceItem(
        case_id="C101",
        customer_problem="Late package",
        support_interaction="Please DM us",
        resolution="Please send us a DM with your order number.",
        intent="late_delivery_shipping_delay",
        score=0.88
    )
    gen_res = GenerationResult(
        case_id="CASE-1",
        customer_message="Where is my order?",
        predicted_intent="late_delivery_shipping_delay",
        intent_confidence=0.85,
        retrieval_mode="intent_aware",
        retrieved_case_ids=["C101"],
        retrieved_evidence=[ev],
        draft_response="Please send us a DM with your order number.",
        grounding_status="grounded",
        safety_check_passed=True,
        safety_violations=[]
    )

    inp = DecisionInput.from_generation_result(gen_res)
    assert inp.case_id == "CASE-1"
    assert inp.intent_confidence == 0.85
    assert inp.top_retrieval_score == 0.88
    assert len(inp.retrieved_evidence) == 1

    engine = RiskAwareDecisionEngine()
    dec = engine.decide(inp)
    assert dec.decision in {"AUTO_HANDLE", "ESCALATE_TO_HUMAN"}
    assert isinstance(dec.risk_score, float)
    assert len(dec.reason_code) > 0
    assert len(dec.reason) > 0
    assert dec.recommended_action in {"auto_send", "human_review"}


def test_hard_rules_escalation():
    """Test that hard safety/evidence rules force human escalation."""
    policy = HardRulesPolicy()

    # 1. Safety check failed
    inp_safety = DecisionInput(
        customer_message="Where is order",
        predicted_intent="late_delivery_shipping_delay",
        intent_confidence=0.90,
        retrieval_mode="intent_aware",
        draft_response="Your order #999 is refunded",
        grounding_status="grounded",
        safety_check_passed=False,
        safety_violations=["Fabricated order ID"]
    )
    res = policy.evaluate(inp_safety)
    assert res.decision == "ESCALATE_TO_HUMAN"
    assert res.reason_code == "SAFETY_CHECK_FAILED"
    assert "safety check failed" in res.reason.lower()

    # 2. Insufficient evidence
    inp_insuf = DecisionInput(
        customer_message="Random query",
        predicted_intent="general_inquiry_other",
        intent_confidence=0.50,
        retrieval_mode="semantic_fallback",
        draft_response="INSUFFICIENT_EVIDENCE",
        grounding_status="insufficient_evidence",
        safety_check_passed=True
    )
    res_insuf = policy.evaluate(inp_insuf)
    assert res_insuf.decision == "ESCALATE_TO_HUMAN"
    assert res_insuf.reason_code == "INSUFFICIENT_EVIDENCE"

    # 3. Empty response
    inp_empty = DecisionInput(
        customer_message="Random query",
        predicted_intent="general_inquiry_other",
        intent_confidence=0.80,
        retrieval_mode="intent_aware",
        retrieved_evidence=[EvidenceItem(case_id="C1", customer_problem="prob", support_interaction="sup", resolution="res", intent="int", score=0.8)],
        draft_response="",
        grounding_status="grounded",
        safety_check_passed=True
    )
    res_empty = policy.evaluate(inp_empty)
    assert res_empty.decision == "ESCALATE_TO_HUMAN"
    assert res_empty.reason_code == "INVALID_RESPONSE"


def test_safe_auto_handle_conditions():
    """Test that a high-confidence, well-grounded, safe query is auto-handled."""
    ev = EvidenceItem(
        case_id="C101",
        customer_problem="Late package",
        support_interaction="Please DM us",
        resolution="Please send us a DM with your order number.",
        intent="late_delivery_shipping_delay",
        score=0.92
    )
    inp = DecisionInput(
        case_id="CASE-SAFE",
        customer_message="My order is delayed",
        predicted_intent="late_delivery_shipping_delay",
        intent_confidence=0.90,
        retrieval_mode="intent_aware",
        retrieved_case_ids=["C101"],
        retrieved_evidence=[ev, ev],
        top_retrieval_score=0.92,
        avg_retrieval_score=0.92,
        draft_response="Please send us a DM with your order number so we can check.",
        grounding_status="grounded",
        safety_check_passed=True,
        safety_violations=[]
    )

    policy = RiskAwarePolicy(risk_threshold=0.35)
    res = policy.evaluate(inp)
    assert res.decision == "AUTO_HANDLE"
    assert res.reason_code == "SAFE_AUTO_HANDLE"
    assert res.recommended_action == "auto_send"
    assert res.risk_score <= 0.35


def test_confidence_only_baseline_policy():
    """Test Baseline 1 ConfidenceOnlyPolicy thresholding."""
    policy = ConfidenceOnlyPolicy(confidence_threshold=0.60)

    inp_low = DecisionInput(
        customer_message="query", predicted_intent="intent", intent_confidence=0.55,
        retrieval_mode="semantic_fallback", draft_response="resp", grounding_status="grounded"
    )
    res_low = policy.evaluate(inp_low)
    assert res_low.decision == "ESCALATE_TO_HUMAN"
    assert res_low.reason_code == "LOW_INTENT_CONFIDENCE"

    inp_high = DecisionInput(
        customer_message="query", predicted_intent="intent", intent_confidence=0.75,
        retrieval_mode="intent_aware", draft_response="resp", grounding_status="grounded"
    )
    res_high = policy.evaluate(inp_high)
    assert res_high.decision == "AUTO_HANDLE"
    assert res_high.reason_code == "HIGH_INTENT_CONFIDENCE"


def test_borderline_composite_risk_and_category_weights():
    """Test composite risk score calculation for borderline and high-risk intent cases."""
    ev = EvidenceItem(
        case_id="C1", customer_problem="prob", support_interaction="sup",
        resolution="res", intent="payment_billing_issue", score=0.65
    )
    # High-risk intent (payment_billing_issue) + mediocre retrieval score (0.65)
    inp_borderline = DecisionInput(
        case_id="CASE-BORDERLINE",
        customer_message="My credit card was charged twice",
        predicted_intent="payment_billing_issue",
        intent_confidence=0.62,
        retrieval_mode="intent_aware",
        retrieved_case_ids=["C1"],
        retrieved_evidence=[ev],
        top_retrieval_score=0.65,
        draft_response="Please DM us your billing email.",
        grounding_status="grounded",
        safety_check_passed=True
    )

    policy = RiskAwarePolicy(risk_threshold=0.35, high_risk_intents=["payment_billing_issue"])
    res = policy.evaluate(inp_borderline)

    # Should escalate due to composite risk score exceeding 0.35
    assert res.decision == "ESCALATE_TO_HUMAN"
    assert res.reason_code == "COMPOSITE_RISK_EXCEEDED"
    assert "payment_billing_issue" in res.reason


def test_determinism_and_explainability():
    """Verify that identical inputs produce identical, explainable DecisionResults."""
    ev = EvidenceItem(case_id="C1", customer_problem="p", support_interaction="s", resolution="r", intent="i", score=0.85)
    inp = DecisionInput(
        case_id="DET-1",
        customer_message="Order status please",
        predicted_intent="order_tracking_delivery_status",
        intent_confidence=0.82,
        retrieval_mode="intent_aware",
        retrieved_evidence=[ev],
        top_retrieval_score=0.85,
        draft_response="Please send us a DM.",
        grounding_status="grounded",
        safety_check_passed=True
    )

    engine = RiskAwareDecisionEngine()
    res1 = engine.decide(inp)
    res2 = engine.decide(inp)

    assert res1.decision == res2.decision
    assert res1.risk_score == res2.risk_score
    assert res1.reason_code == res2.reason_code
    assert res1.reason == res2.reason
    assert res1.signals == res2.signals
