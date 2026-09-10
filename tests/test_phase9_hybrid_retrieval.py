"""
Unit tests for Phase 9: Hybrid Retrieval Architecture (BM25 + Dense + RRF Candidate Fusion).
"""
import pytest
import numpy as np

from hiver_agent.retrieval.interfaces import EmbeddingProvider, VectorStore, Retriever
from hiver_agent.retrieval.retriever import SearchResult, FAISSRetriever, IntentAwareRetriever
from hiver_agent.retrieval.bm25_retriever import BM25Retriever, tokenize_text
from hiver_agent.retrieval.hybrid_retriever import HybridRetriever
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.generation.generator import TemplateResponseGenerator
from hiver_agent.pipeline import CustomerSupportPipeline


class MockDenseRetriever(Retriever):
    """
    Mock dense retriever returning controlled SearchResult candidates for fusion testing.
    """
    def __init__(self, results: list[SearchResult]):
        super().__init__(embedding_provider=None, vector_store=None) # type: ignore
        self.results = results

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self.results[:top_k]


def test_tokenize_text():
    tokens = tokenize_text("Where is my package? Order ID: 403-1234567!")
    assert tokens == ["where", "is", "my", "package", "order", "id", "403", "1234567"]


def test_bm25_retriever_basic():
    docs = [
        {"case_id": "CASE-1", "customer_problem": "Where is my refund for returned item?", "intent": "refund_status_request"},
        {"case_id": "CASE-2", "customer_problem": "My package shipping delay issue.", "intent": "late_delivery_shipping_delay"},
        {"case_id": "CASE-3", "customer_problem": "Cannot login to my prime account.", "intent": "account_access_login"},
    ]
    retriever = BM25Retriever(docs)

    results = retriever.retrieve("refund returned item", top_k=2)
    assert len(results) == 2
    assert results[0].case_id == "CASE-1"
    assert results[0].score > 0.0
    assert results[0].intent == "refund_status_request"


def test_bm25_retriever_empty_or_special_query():
    docs = [{"case_id": "CASE-1", "customer_problem": "Order tracking status"}]
    retriever = BM25Retriever(docs)

    assert retriever.retrieve("", top_k=5) == []
    assert retriever.retrieve("   !!!   ", top_k=5) == []


def test_hybrid_rrf_fusion_calculation():
    # Case 1: Present in Dense (rank 1) and BM25 (rank 2)
    # Case 2: Present in Dense (rank 2) and BM25 (rank 1)
    # Case 3: Present in BM25 only (rank 3)
    cand_dense_1 = SearchResult(case_id="C1", score=0.9, customer_problem="prob1", support_interaction="act1", resolution="res1", intent="late_delivery_shipping_delay")
    cand_dense_2 = SearchResult(case_id="C2", score=0.8, customer_problem="prob2", support_interaction="act2", resolution="res2", intent="refund_status_request")

    cand_bm25_1 = SearchResult(case_id="C2", score=5.0, customer_problem="prob2", support_interaction="act2", resolution="res2", intent="refund_status_request")
    cand_bm25_2 = SearchResult(case_id="C1", score=4.0, customer_problem="prob1", support_interaction="act1", resolution="res1", intent="late_delivery_shipping_delay")
    cand_bm25_3 = SearchResult(case_id="C3", score=3.0, customer_problem="prob3", support_interaction="act3", resolution="res3", intent="account_access_login")

    dense_retriever = MockDenseRetriever([cand_dense_1, cand_dense_2])
    bm25_retriever = MockDenseRetriever([cand_bm25_1, cand_bm25_2, cand_bm25_3])

    rrf_k = 60
    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=rrf_k,
        dense_top_k=10,
        bm25_top_k=10
    )

    fused = hybrid.retrieve("query text", top_k=10)
    assert len(fused) == 3

    # Expected scores:
    # C1: rank_dense=1, rank_bm25=2 -> 1/(61) + 1/(62) = 0.016393 + 0.016129 = 0.032522
    # C2: rank_dense=2, rank_bm25=1 -> 1/(62) + 1/(61) = 0.032522
    # C3: rank_bm25=3 -> 1/(63) = 0.015873
    expected_c1_score = round(1.0 / 61 + 1.0 / 62, 6)
    expected_c3_score = round(1.0 / 63, 6)

    case_scores = {r.case_id: r.score for r in fused}
    assert case_scores["C1"] == expected_c1_score
    assert case_scores["C2"] == expected_c1_score
    assert case_scores["C3"] == expected_c3_score


def test_intent_aware_retriever_wrapping_hybrid():
    cand1 = SearchResult(case_id="C1", score=0.03, customer_problem="p1", support_interaction="s1", resolution="r1", intent="refund_status_request")
    cand2 = SearchResult(case_id="C2", score=0.02, customer_problem="p2", support_interaction="s2", resolution="r2", intent="late_delivery_shipping_delay")

    hybrid_retriever = MockDenseRetriever([cand1, cand2])
    classifier = RuleBasedIntentClassifier()

    intent_aware = IntentAwareRetriever(
        embedding_provider=None, # type: ignore
        vector_store=None, # type: ignore
        classifier=classifier,
        confidence_threshold=0.60,
        intent_boost_factor=0.15,
        base_retriever=hybrid_retriever
    )

    # Rule classifier matches "refund" -> refund_status_request with confidence 0.95
    results = intent_aware.retrieve("I want a refund for my order", top_k=2)
    assert len(results) == 2
    assert results[0].case_id == "C1"
    assert results[0].score > 0.15


def test_pipeline_with_hybrid_retriever():
    docs = [
        {"case_id": "CASE-10", "customer_problem": "Where is my package? Delayed delivery", "intent": "late_delivery_shipping_delay"},
        {"case_id": "CASE-20", "customer_problem": "Need refund for damaged item", "intent": "refund_status_request"}
    ]
    bm25 = BM25Retriever(docs)
    dense = MockDenseRetriever([
        SearchResult(case_id="CASE-10", score=0.85, customer_problem="Where is my package? Delayed delivery", support_interaction="act", resolution="res", intent="late_delivery_shipping_delay")
    ])
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    clf = RuleBasedIntentClassifier()
    gen = TemplateResponseGenerator()
    pipeline = CustomerSupportPipeline(
        classifier=clf,
        base_retriever=hybrid,
        generator=gen
    )

    res = pipeline.process_end_to_end("Where is my package?", case_id="OTHER-99")
    assert res.decision in ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]
    assert len(res.retrieved_case_ids) > 0
