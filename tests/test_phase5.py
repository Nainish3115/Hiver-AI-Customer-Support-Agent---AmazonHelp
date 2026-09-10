"""
Unit tests for Phase 5 AmazonHelp intent classification system.
"""
import pytest
import numpy as np
import pandas as pd

from hiver_agent.classification.taxonomy import AMAZON_INTENTS, SUPER_CATEGORIES
from hiver_agent.classification.interface import IntentPrediction, IntentClassifier
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.tfidf_classifier import TfidfIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier, softmax
from hiver_agent.classification.embedding_linear_classifier import EmbeddingLinearClassifier
from hiver_agent.classification.dataset import (
    normalize_text_for_grouping,
    create_near_duplicate_groups,
    split_grouped_dataset
)
from hiver_agent.classification.evaluation import (
    evaluate_classifier_predictions,
    extract_top_confusion_pairs,
    evaluate_confidence_abstention
)


class DummyEmbeddingProvider:
    """
    Fast mock embedding provider for unit testing without loading full transformer weights.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_query(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % 100000)
        vec = np.random.randn(self.dimension).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed_query(t) for t in texts], dtype=np.float32)


def test_taxonomy_integrity():
    assert len(AMAZON_INTENTS) == 17
    assert "order_tracking_delivery_status" in AMAZON_INTENTS
    assert "general_inquiry_other" in AMAZON_INTENTS

    # Check super categories cover all 17 intents exactly
    super_intents = []
    for cat, intents in SUPER_CATEGORIES.items():
        super_intents.extend(intents)

    assert set(super_intents) == set(AMAZON_INTENTS)
    assert len(super_intents) == 17


def test_text_normalization_and_grouping():
    t1 = "My order #123 is delayed! https://t.co/xyz @AmazonHelp"
    t2 = "My order #456 is delayed! https://t.co/abc @AmazonHelp"

    n1 = normalize_text_for_grouping(t1)
    n2 = normalize_text_for_grouping(t2)

    assert n1 == "my order is delayed"
    assert n2 == "my order is delayed"

    records = [
        {"case_id": "C1", "customer_text": t1, "intent": "late_delivery_shipping_delay"},
        {"case_id": "C2", "customer_text": t2, "intent": "late_delivery_shipping_delay"},
        {"case_id": "C3", "customer_text": "I need a refund for my item", "intent": "refund_status_request"},
    ]

    grouped, stats = create_near_duplicate_groups(records)
    assert stats["unique_groups"] == 2
    assert grouped[0]["group_id"] == grouped[1]["group_id"]
    assert grouped[0]["group_id"] != grouped[2]["group_id"]


def test_leakage_resistant_split():
    records = []
    intents = ["order_tracking_delivery_status", "refund_status_request", "prime_membership_subscription"]
    for i in range(100):
        records.append({
            "case_id": f"C-{i}",
            "customer_text": f"Customer problem description number {i // 3}", # duplicated text in groups of 3
            "intent": intents[i % len(intents)]
        })

    train, val, test, split_meta = split_grouped_dataset(records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(train) + len(val) + len(test) == len(records)

    train_groups = {r["group_id"] for r in train}
    val_groups = {r["group_id"] for r in val}
    test_groups = {r["group_id"] for r in test}

    # Verify ZERO leakage across splits
    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0


def test_rule_based_classifier():
    clf = RuleBasedIntentClassifier()
    pred = clf.predict("Where is my order? Track delivery status please")
    assert isinstance(pred, IntentPrediction)
    assert pred.intent == "order_tracking_delivery_status"
    assert pred.confidence >= 0.80


def test_tfidf_classifier():
    texts = [
        "Where is my package and delivery tracking",
        "Track my shipping delivery status",
        "I need a refund for my returned order",
        "Refund money back to my bank account",
        "Cancel my subscription and prime membership",
        "Cancel prime membership fee"
    ]
    labels = [
        "order_tracking_delivery_status",
        "order_tracking_delivery_status",
        "refund_status_request",
        "refund_status_request",
        "prime_membership_subscription",
        "prime_membership_subscription"
    ]

    clf = TfidfIntentClassifier(c_param=1.0)
    clf.fit(texts, labels)

    pred = clf.predict("Where is my package tracking?")
    assert isinstance(pred, IntentPrediction)
    assert pred.intent in AMAZON_INTENTS
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.probabilities) > 0


def test_embedding_prototype_classifier():
    embedder = DummyEmbeddingProvider()
    clf = EmbeddingPrototypeClassifier(embedder)

    texts = [
        "Where is my package tracking",
        "I want a refund for item",
        "Cancel prime membership"
    ]
    labels = [
        "order_tracking_delivery_status",
        "refund_status_request",
        "prime_membership_subscription"
    ]

    clf.fit(texts, labels)
    pred = clf.predict("Where is my package?")

    assert isinstance(pred, IntentPrediction)
    assert pred.intent in AMAZON_INTENTS
    assert 0.0 <= pred.confidence <= 1.0


def test_embedding_linear_classifier():
    embedder = DummyEmbeddingProvider()
    clf = EmbeddingLinearClassifier(embedder, c_param=1.0)

    texts = [
        "Where is my package tracking",
        "Track delivery status",
        "I want a refund for item",
        "Refund money back",
        "Cancel prime membership",
        "Prime subscription cancellation"
    ]
    labels = [
        "order_tracking_delivery_status",
        "order_tracking_delivery_status",
        "refund_status_request",
        "refund_status_request",
        "prime_membership_subscription",
        "prime_membership_subscription"
    ]

    clf.fit(texts, labels)
    pred = clf.predict("Track delivery status")

    assert isinstance(pred, IntentPrediction)
    assert pred.intent in AMAZON_INTENTS
    assert 0.0 <= pred.confidence <= 1.0


def test_evaluation_metrics_and_abstention():
    y_true = ["order_tracking_delivery_status", "refund_status_request", "prime_membership_subscription"]
    y_pred = ["order_tracking_delivery_status", "refund_status_request", "general_inquiry_other"]

    metrics, df_per_class = evaluate_classifier_predictions(y_true, y_pred, labels=AMAZON_INTENTS)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert round(metrics["accuracy"], 2) == 0.67

    preds = [
        IntentPrediction(text="t1", intent=p, confidence=0.85) for p in y_pred
    ]
    df_abst = evaluate_confidence_abstention(preds, y_true, thresholds=[0.5, 0.9])
    assert len(df_abst) == 2
    assert df_abst.iloc[0]["coverage"] == 1.0
    assert df_abst.iloc[1]["coverage"] == 0.0
