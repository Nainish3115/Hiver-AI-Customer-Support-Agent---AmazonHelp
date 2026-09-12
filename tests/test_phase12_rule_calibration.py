"""
Unit tests for Day 3 Calibrated Rule Classifier and Ambiguity Resolver.
"""
import sys
import pytest
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from hiver_agent.classification.taxonomy import AMAZON_INTENTS
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.calibrated_rule_classifier import (
    CalibratedRuleIntentClassifier,
    RuleAmbiguityResolver,
    PATTERN_SPECIFICITY
)


def test_rule_specificity_scoring():
    clf = CalibratedRuleIntentClassifier(use_ambiguity_resolver=False)
    # Generic keyword match ("prime") should have lower score than specific phrase ("delivered but not")
    pred_generic = clf.predict("amazon prime")
    pred_specific = clf.predict("it says delivered but not received")
    assert pred_specific.confidence > pred_generic.confidence


def test_generic_keyword_behavior():
    clf = CalibratedRuleIntentClassifier(use_ambiguity_resolver=False)
    pred = clf.predict("my account")
    assert pred.intent in AMAZON_INTENTS
    assert pred.confidence <= 0.60


def test_specific_phrase_behavior():
    clf = CalibratedRuleIntentClassifier(use_ambiguity_resolver=False)
    pred = clf.predict("where is my order status")
    assert pred.intent == "order_tracking_delivery_status"
    assert pred.confidence >= 0.80


def test_conflicting_rule_detection():
    clf = CalibratedRuleIntentClassifier(use_ambiguity_resolver=False)
    # Query matching multiple competing intent rules (tracking vs refund)
    pred = clf.predict("where is my order status refund money back")
    meta = pred.metadata or {}
    assert meta.get("conflict_detected") is True


def test_confidence_reduction_under_conflict():
    clf_no_res = CalibratedRuleIntentClassifier(use_ambiguity_resolver=False)
    pred_single = clf_no_res.predict("where is my order status")
    pred_conflict = clf_no_res.predict("where is my order status refund money back")
    assert pred_conflict.confidence < pred_single.confidence


def test_delivery_tracking_vs_late_delivery():
    resolver = RuleAmbiguityResolver()
    # Tracking query without late keywords
    intent_tr, conf_tr = resolver.resolve_delivery_ambiguity("where is my package", "order_tracking_delivery_status", ["order_tracking_delivery_status"])
    assert intent_tr == "order_tracking_delivery_status"

    # Late delivery query with temporal keyword
    intent_late, conf_late = resolver.resolve_delivery_ambiguity("my package was supposed to arrive yesterday", "order_tracking_delivery_status", ["late_delivery_shipping_delay"])
    assert intent_late == "late_delivery_shipping_delay"


def test_complaint_vs_damaged_item():
    resolver = RuleAmbiguityResolver()
    # Damaged item query
    intent_dmg, _ = resolver.resolve_complaint_vs_damage("the package arrived broken and smashed", "customer_service_complaint")
    assert intent_dmg == "damaged_defective_item"

    # Terrible service complaint query
    intent_svc, _ = resolver.resolve_complaint_vs_damage("terrible service by supervisor", "customer_service_complaint")
    assert intent_svc == "customer_service_complaint"


def test_general_inquiry_protection():
    clf = CalibratedRuleIntentClassifier(use_ambiguity_resolver=True)
    # Ambiguous query with no matching keyword rules
    pred = clf.predict("hello can you help me today")
    assert pred.intent == "general_inquiry_other"
    assert pred.confidence <= 0.45


def test_valid_intent_output():
    clf = CalibratedRuleIntentClassifier(use_ambiguity_resolver=True)
    queries = ["where is my order", "I want a refund", "password reset account"]
    for q in queries:
        pred = clf.predict(q)
        assert pred.intent in AMAZON_INTENTS
        assert 0.0 <= pred.confidence <= 1.0


def test_deterministic_output():
    clf = CalibratedRuleIntentClassifier(use_ambiguity_resolver=True)
    pred1 = clf.predict("cancel my order please")
    pred2 = clf.predict("cancel my order please")
    assert pred1.intent == pred2.intent
    assert pred1.confidence == pred2.confidence


def test_no_changes_to_existing_rule_classifier():
    old_clf = RuleBasedIntentClassifier()
    pred = old_clf.predict("where is my order")
    assert pred.intent == "order_tracking_delivery_status"
    assert pred.confidence == 0.85


def test_confidence_metadata_structure():
    clf = CalibratedRuleIntentClassifier(use_ambiguity_resolver=True)
    pred = clf.predict("where is my order status")
    assert pred.metadata is not None
    assert "conflict_detected" in pred.metadata
    assert "matched_rules" in pred.metadata
    assert pred.probabilities is not None
