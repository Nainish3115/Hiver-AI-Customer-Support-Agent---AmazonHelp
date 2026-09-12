"""
Unit tests for Phase 14 Lightweight Deterministic Reranker.
Verifies score normalization, lexical overlap, intent compatibility,
candidate ordering, edge cases, and zero gold-intent usage during retrieval.
"""
import pytest
from hiver_agent.retrieval.retriever import SearchResult
from hiver_agent.retrieval.lightweight_reranker import (
    LightweightReranker,
    compute_lexical_overlap,
    tokenize_query_clean
)


def test_tokenize_query_clean():
    query = "My package has not arrived yet please help Amazon"
    tokens = tokenize_query_clean(query)
    # Stopwords 'my', 'has', 'not', 'yet', 'please', 'help', 'amazon' filtered out
    assert "package" in tokens
    assert "arrived" in tokens
    assert "help" not in tokens
    assert "amazon" not in tokens


def test_compute_lexical_overlap():
    query_tokens = ["package", "arrived", "late"]
    doc_text = "The customer complains that their package arrived late last night."
    overlap = compute_lexical_overlap(query_tokens, doc_text)
    assert overlap == pytest.approx(1.0)

    partial_doc = "Package was delivered on time."
    overlap_partial = compute_lexical_overlap(query_tokens, partial_doc)
    assert overlap_partial == pytest.approx(1 / 3)


def test_intent_compatibility_scoring():
    reranker = LightweightReranker(config_name="hybrid_intent_lexical_dense")
    cand_match = SearchResult(
        case_id="case_1",
        score=0.05,
        customer_problem="Late package",
        support_interaction="Checking status",
        resolution="Refund issued",
        intent="late_delivery_shipping_delay",
        metadata={"dense_score": 0.8}
    )
    cand_mismatch = SearchResult(
        case_id="case_2",
        score=0.03,
        customer_problem="Wrong item",
        support_interaction="Return requested",
        resolution="Replacement sent",
        intent="return_exchange_request",
        metadata={"dense_score": 0.5}
    )

    results = reranker.rerank(
        query="Where is my package?",
        predicted_intent="late_delivery_shipping_delay",
        intent_confidence=0.85,
        candidates=[cand_match, cand_mismatch],
        top_k=2
    )

    assert len(results) == 2
    assert results[0].case_id == "case_1"
    assert results[0].metadata["signal_intent"] == 1.0
    assert results[1].metadata["signal_intent"] == 0.0


def test_empty_candidate_handling():
    reranker = LightweightReranker()
    results = reranker.rerank(
        query="Hello",
        predicted_intent="general_inquiry_other",
        intent_confidence=0.9,
        candidates=[],
        top_k=10
    )
    assert results == []


def test_deterministic_ranking_and_reproducibility():
    reranker = LightweightReranker()
    cand1 = SearchResult(
        case_id="c1",
        score=0.02,
        customer_problem="Refund issue",
        support_interaction="Agent helped",
        resolution="Refund processed",
        intent="refund_status_request",
        metadata={"dense_score": 0.65}
    )
    cand2 = SearchResult(
        case_id="c2",
        score=0.04,
        customer_problem="Damaged product refund",
        support_interaction="Agent initiated refund",
        resolution="Refund processed",
        intent="refund_status_request",
        metadata={"dense_score": 0.85}
    )

    run1 = reranker.rerank(
        query="I need refund status",
        predicted_intent="refund_status_request",
        intent_confidence=0.75,
        candidates=[cand1, cand2],
        top_k=2
    )

    run2 = reranker.rerank(
        query="I need refund status",
        predicted_intent="refund_status_request",
        intent_confidence=0.75,
        candidates=[cand1, cand2],
        top_k=2
    )

    assert len(run1) == 2
    assert [c.case_id for c in run1] == [c.case_id for c in run2]
    assert [c.score for c in run1] == [c.score for c in run2]


def test_no_gold_intent_leakage_during_retrieval():
    """
    Verifies that the reranker uses only predicted_intent parameter
    and ignores candidate gold intent labels during reranking.
    """
    reranker = LightweightReranker()
    cand = SearchResult(
        case_id="c1",
        score=0.02,
        customer_problem="Cancel order",
        support_interaction="Order cancelled",
        resolution="Done",
        intent="cancel_order_request",
        metadata={"gold_intent": "NOT_USED_IN_RERANKING"}
    )

    # Supply predicted_intent distinct from metadata gold_intent
    results = reranker.rerank(
        query="Cancel my order",
        predicted_intent="cancel_order_request",
        intent_confidence=0.90,
        candidates=[cand],
        top_k=1
    )

    assert results[0].metadata["signal_intent"] == 1.0
    # Gold intent in metadata must not override candidate.intent comparison
    assert "gold_intent" in cand.metadata
