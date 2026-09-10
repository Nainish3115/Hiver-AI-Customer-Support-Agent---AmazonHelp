"""
Unit tests for Phase 5.6 Classifier Diagnostic and Improvement logic.
"""
import pytest
import numpy as np
from scripts.run_phase_5_6_experiments import (
    split_gold_dataset_50_50,
    compute_standard_error,
    TfidfSvcClassifier,
    HybridRuleEmbeddingClassifier
)
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier


class DummyEmbeddingProvider:
    """
    Fast mock embedding provider for unit testing.
    """
    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def embed_query(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % 100000)
        vec = np.random.randn(self.dimension).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed_query(t) for t in texts], dtype=np.float32)


def test_split_gold_dataset_50_50_determinism():
    """Verify 50/50 split produces exactly 50/50 split and is deterministic with seed=42."""
    intents = ["order_tracking_delivery_status", "refund_status_request", "prime_membership_subscription", "account_access_login"]
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    records = []
    for i in range(200):
        # Create distinct unique non-digit text queries
        txt = f"customer query {words[i % len(words)]} item unique_token_{i}"
        records.append({
            "case_id": f"case_{i}",
            "customer_text": txt,
            "final_human_intent": intents[i % len(intents)]
        })
    
    dev1, test1 = split_gold_dataset_50_50(records, seed=42)
    dev2, test2 = split_gold_dataset_50_50(records, seed=42)

    assert len(dev1) + len(test1) == 200
    assert len(dev1) == 100
    assert len(test1) == 100
    assert [r["case_id"] for r in dev1] == [r["case_id"] for r in dev2]
    assert [r["case_id"] for r in test1] == [r["case_id"] for r in test2]


def test_compute_standard_error():
    """Verify standard error calculation formula SE = sqrt(p * (1 - p) / N)."""
    se = compute_standard_error(0.50, 100)
    assert abs(se - 0.05) < 1e-4

    se_zero = compute_standard_error(0.50, 0)
    assert se_zero == 0.0


def test_tfidf_svc_classifier_fit_predict():
    """Test TF-IDF + Linear SVM classifier training and prediction."""
    texts = [
        "where is my package",
        "please cancel my order",
        "i was charged twice",
        "received wrong item in box"
    ]
    labels = [
        "order_tracking_delivery_status",
        "order_cancellation_request",
        "payment_billing_issue",
        "wrong_item_received"
    ]

    clf = TfidfSvcClassifier()
    clf.fit(texts, labels)

    pred = clf.predict("where is my package")
    assert pred.intent in labels
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.probabilities) == len(labels)


def test_hybrid_rule_embedding_classifier():
    """Test Hybrid Rule + Embedding Prototype classifier logic."""
    embedder = DummyEmbeddingProvider(dimension=64)
    rule_clf = RuleBasedIntentClassifier()
    proto_clf = EmbeddingPrototypeClassifier(embedder)

    texts = ["cancel my order", "random query about something"]
    labels = ["order_cancellation_request", "general_inquiry_other"]
    proto_clf.fit(texts, labels)

    hybrid = HybridRuleEmbeddingClassifier(rule_clf, proto_clf)

    # Keyword rule should fire on clear cancellation phrase
    pred_rule = hybrid.predict("cancel my order")
    assert pred_rule.intent == "order_cancellation_request"

    # Embedding prototype should fallback on ambiguous text
    pred_fallback = hybrid.predict("random query about something")
    assert pred_fallback.intent in labels
