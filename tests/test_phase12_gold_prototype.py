"""
Unit tests for Day 4 Gold Prototype Classifier and Hybrid module.
"""
import sys
import pytest
import numpy as np
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from hiver_agent.classification.taxonomy import AMAZON_INTENTS
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.classification.calibrated_rule_classifier import CalibratedRuleIntentClassifier
from hiver_agent.classification.gold_prototype_classifier import (
    GoldPrototypeIntentClassifier,
    CalibratedRulePrototypeHybrid,
    normalize_vector
)
from scripts.evaluate_phase12_hierarchical_classifier import load_gold_records
from scripts.run_phase_5_6_experiments import split_gold_dataset_50_50


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")


def test_prototype_construction(embedder):
    proto_clf = GoldPrototypeIntentClassifier(embedder, variant_mode="mean")
    texts = ["where is my package", "I want a refund", "damaged product arrived"]
    labels = ["order_tracking_delivery_status", "refund_status_request", "damaged_defective_item"]
    proto_clf.fit(texts, labels)

    assert len(proto_clf.centroids) == 17
    assert "order_tracking_delivery_status" in proto_clf.centroids
    assert len(proto_clf.centroids["order_tracking_delivery_status"][0]) == 384


def test_prototype_normalization():
    vec = np.array([3.0, 4.0, 0.0])
    norm_v = normalize_vector(vec)
    assert abs(np.linalg.norm(norm_v) - 1.0) < 1e-5


def test_valid_17_intent_output(embedder):
    proto_clf = GoldPrototypeIntentClassifier(embedder, variant_mode="mean")
    proto_clf.fit(["package status"], ["order_tracking_delivery_status"])

    pred = proto_clf.predict("where is my order")
    assert pred.intent in AMAZON_INTENTS
    assert 0.0 <= pred.confidence <= 1.0


def test_deterministic_prediction(embedder):
    proto_clf = GoldPrototypeIntentClassifier(embedder, variant_mode="mean")
    proto_clf.fit(["track package", "refund money"], ["order_tracking_delivery_status", "refund_status_request"])

    pred1 = proto_clf.predict("where is package")
    pred2 = proto_clf.predict("where is package")
    assert pred1.intent == pred2.intent
    assert pred1.confidence == pred2.confidence


def test_empty_invalid_class_handling(embedder):
    proto_clf = GoldPrototypeIntentClassifier(embedder, variant_mode="mean")
    # Fit with empty inputs
    proto_clf.fit([], [])

    pred = proto_clf.predict("anything")
    assert pred.intent in AMAZON_INTENTS


def test_test_set_exclusion_verification(embedder):
    df_gold, gold_records = load_gold_records()
    gold_dev_recs, gold_test_recs = split_gold_dataset_50_50(gold_records, seed=42)

    dev_texts = [r["customer_text"] for r in gold_dev_recs]
    dev_labels = [r["intent"] for r in gold_dev_recs]
    test_texts = [r["customer_text"] for r in gold_test_recs]

    proto_clf = GoldPrototypeIntentClassifier(embedder, variant_mode="mean")
    proto_clf.fit(dev_texts, dev_labels)

    # Ensure model dev_support total equals dev set length
    assert sum(proto_clf.dev_support.values()) == len(dev_texts)

    pred = proto_clf.predict(test_texts[0])
    assert pred.intent in AMAZON_INTENTS


def test_confidence_margin_metadata(embedder):
    proto_clf = GoldPrototypeIntentClassifier(embedder, variant_mode="mean")
    proto_clf.fit(["order status", "cancel my order"], ["order_tracking_delivery_status", "order_cancellation_request"])

    pred = proto_clf.predict("status of order")
    meta = pred.metadata or {}
    assert "top_similarity" in meta
    assert "second_similarity" in meta
    assert "margin" in meta
    assert meta["margin"] >= 0.0


def test_hybrid_decision_logic(embedder):
    rule_clf = CalibratedRuleIntentClassifier(use_ambiguity_resolver=True)
    proto_clf = GoldPrototypeIntentClassifier(embedder, variant_mode="mean")
    proto_clf.fit(["order tracking", "damaged item"], ["order_tracking_delivery_status", "damaged_defective_item"])

    hybrid = CalibratedRulePrototypeHybrid(rule_clf, proto_clf)

    # 1. High-specificity rule query -> decision_source = high_specificity_rule
    pred_rule = hybrid.predict("it says delivered but not received")
    assert pred_rule.metadata["decision_source"] == "high_specificity_rule"

    # 2. Query with no keywords -> decision_source = prototype_semantic
    pred_proto = hybrid.predict("something else entirely")
    assert pred_proto.metadata["decision_source"] in {"prototype_semantic", "blended_rule_prototype"}
