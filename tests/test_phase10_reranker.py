"""
Unit tests for Phase 10 Cross-Encoder Reranker and pipeline integration.
"""
import pytest
from hiver_agent.retrieval.reranker import SemanticReranker
from hiver_agent.retrieval.retriever import SearchResult, FAISSRetriever, IntentAwareRetriever
from hiver_agent.retrieval.bm25_retriever import BM25Retriever
from hiver_agent.retrieval.hybrid_retriever import HybridRetriever
from hiver_agent.pipeline import CustomerSupportPipeline
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.generation.generator import TemplateResponseGenerator


@pytest.fixture(scope="module")
def reranker():
    return SemanticReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_reranker_initialization(reranker):
    assert reranker is not None
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_reranker_empty_candidates(reranker):
    res = reranker.rerank("where is my order?", [], top_k=3)
    assert res == []


def test_reranker_candidate_ordering(reranker):
    cands = [
        SearchResult(
            case_id="RAG-1",
            score=0.5,
            customer_problem="I bought shoes",
            support_interaction="Shoes delivered",
            resolution="Enjoy shoes",
            intent="general_inquiry_other"
        ),
        SearchResult(
            case_id="RAG-2",
            score=0.4,
            customer_problem="My package has not been delivered yet and is delayed",
            support_interaction="Checking carrier status",
            resolution="Carrier delayed package. New delivery tomorrow.",
            intent="late_delivery_shipping_delay"
        )
    ]

    query = "my package is late where is it"
    reranked = reranker.rerank(query, cands, top_k=2)

    assert len(reranked) == 2
    # RAG-2 should rank higher than RAG-1 for a package delay query
    assert reranked[0].case_id == "RAG-2"
    assert "reranker_score" in reranked[0].metadata
    assert "original_score" in reranked[0].metadata


def test_reranker_top_k_truncation(reranker):
    cands = [
        SearchResult(
            case_id=f"RAG-{i}",
            score=0.1 * i,
            customer_problem=f"Problem {i}",
            support_interaction=f"Interaction {i}",
            resolution=f"Resolution {i}",
            intent="general_inquiry_other"
        )
        for i in range(5)
    ]
    reranked = reranker.rerank("test query", cands, top_k=2)
    assert len(reranked) == 2


def test_pipeline_without_reranker():
    class DummyClassifier:
        def predict(self, text: str):
            from hiver_agent.classification.interface import IntentPrediction
            return IntentPrediction(
                text=text,
                intent="order_tracking_delivery_status",
                confidence=0.90,
                probabilities={"order_tracking_delivery_status": 0.90}
            )

    class DummyRetriever:
        def retrieve(self, query: str, top_k: int = 5):
            return [
                SearchResult(
                    case_id="RAG-100",
                    score=0.8,
                    customer_problem="Order problem",
                    support_interaction="Help",
                    resolution="Resolved",
                    intent="order_tracking_delivery_status"
                )
            ]

    clf = DummyClassifier()
    ret = DummyRetriever()
    gen = TemplateResponseGenerator()

    pipeline = CustomerSupportPipeline(classifier=clf, base_retriever=ret, generator=gen)
    res = pipeline.process_message("track my order")

    assert res.retrieval_mode == "intent_aware"
    assert len(res.retrieved_evidence) > 0
    assert res.retrieved_evidence[0].case_id == "RAG-100"


def test_pipeline_with_reranker(reranker):
    class DummyRetriever:
        def retrieve(self, query: str, top_k: int = 5):
            return [
                SearchResult(
                    case_id="RAG-100",
                    score=0.5,
                    customer_problem="General question",
                    support_interaction="Hi",
                    resolution="Bye",
                    intent="general_inquiry_other"
                ),
                SearchResult(
                    case_id="RAG-200",
                    score=0.4,
                    customer_problem="Package missing and delayed",
                    support_interaction="Checking tracking",
                    resolution="Delivered tomorrow",
                    intent="late_delivery_shipping_delay"
                )
            ]

    clf = RuleBasedIntentClassifier()
    ret = DummyRetriever()
    gen = TemplateResponseGenerator()

    pipeline = CustomerSupportPipeline(
        classifier=clf,
        base_retriever=ret,
        generator=gen,
        reranker=reranker,
        reranker_candidate_k=10
    )
    res = pipeline.process_message("where is my delayed package")

    assert "+reranked" in res.retrieval_mode
    assert len(res.retrieved_evidence) > 0
    assert res.retrieved_evidence[0].case_id == "RAG-200"
