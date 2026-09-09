"""
Deterministic unit tests for Phase 4: Embeddings, FAISS vector store,
retrievers, classifiers, evaluation metrics, and edge cases.
"""
import pytest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd

from hiver_agent.retrieval.interfaces import EmbeddingProvider, VectorStore, Retriever
from hiver_agent.retrieval.faiss_store import FAISSVectorStore
from hiver_agent.retrieval.retriever import FAISSRetriever, SearchResult, IntentAwareRetriever
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier


class DummyEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 16):
        self._dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        # Create deterministic pseudo-embeddings based on text length
        res = []
        for t in texts:
            val = float(len(t) % 10 + 1)
            vec = np.ones(self._dim, dtype=np.float32) * val
            vec = vec / np.linalg.norm(vec)
            res.append(vec)
        return np.array(res, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])

    @property
    def dimension(self) -> int:
        return self._dim


def test_faiss_vector_store_add_and_search():
    embedder = DummyEmbeddingProvider(dim=8)
    store = FAISSVectorStore(dimension=8)

    docs = [
        {"case_id": "RAG-1", "intent": "order_tracking_delivery_status", "document_text": "INTENT: order_tracking\nCUSTOMER PROBLEM: Where is order?\nSUPPORT INTERACTION: Check tracking.\nRESOLUTION: Link sent."},
        {"case_id": "RAG-2", "intent": "refund_status_request", "document_text": "INTENT: refund_status\nCUSTOMER PROBLEM: Where is my refund?\nSUPPORT INTERACTION: Refund processed.\nRESOLUTION: Credited 5 days."}
    ]

    embs = embedder.embed_texts([d["document_text"] for d in docs])
    store.add_documents(docs, embs)

    assert store.count() == 2

    # Search
    q_emb = embedder.embed_query("Where is my order?")
    results = store.search(q_emb, top_k=2)

    assert len(results) == 2
    assert "_score" in results[0]
    assert results[0]["case_id"] in ["RAG-1", "RAG-2"]


def test_faiss_store_save_and_load(tmp_path):
    embedder = DummyEmbeddingProvider(dim=8)
    store = FAISSVectorStore(dimension=8)
    docs = [{"case_id": "RAG-100", "intent": "late_delivery_shipping_delay", "document_text": "Late delivery text"}]
    embs = embedder.embed_texts(["Late delivery text"])
    store.add_documents(docs, embs)

    save_dir = str(tmp_path / "faiss_test_index")
    store.save(save_dir)

    loaded_store = FAISSVectorStore.load(save_dir)
    assert loaded_store.count() == 1
    assert loaded_store.documents[0]["case_id"] == "RAG-100"


def test_faiss_retriever_structured_output():
    embedder = DummyEmbeddingProvider(dim=8)
    store = FAISSVectorStore(dimension=8)
    docs = [{
        "case_id": "RAG-500",
        "intent": "damaged_defective_item",
        "customer_problem": "Item is broken.",
        "support_interaction": "[cust]: Broken.\n[AmazonHelp]: Replacement sent.",
        "resolution": "Replacement sent.",
        "metadata": {"conversation_id": 500}
    }]
    embs = embedder.embed_texts(["Item is broken."])
    store.add_documents(docs, embs)

    retriever = FAISSRetriever(embedder, store)
    results = retriever.retrieve("Item is broken.", top_k=1)

    assert len(results) == 1
    res = results[0]
    assert isinstance(res, SearchResult)
    assert res.case_id == "RAG-500"
    assert res.intent == "damaged_defective_item"
    assert res.resolution == "Replacement sent."


def test_rule_based_classifier():
    clf = RuleBasedIntentClassifier()
    pred = clf.predict("Where is my tracking status for order?")
    assert pred.intent == "order_tracking_delivery_status"
    assert pred.confidence > 0.5


def test_embedding_prototype_classifier():
    embedder = DummyEmbeddingProvider(dim=8)
    clf = EmbeddingPrototypeClassifier(embedder)

    train_texts = [
        "Where is my order tracking?",
        "Where is my order package status?",
        "I need a refund for my item",
        "When will I get my refund money back"
    ]
    train_labels = [
        "order_tracking_delivery_status",
        "order_tracking_delivery_status",
        "refund_status_request",
        "refund_status_request"
    ]

    clf.fit(train_texts, train_labels)

    pred = clf.predict("Where is my package tracking?")
    assert pred.intent == "order_tracking_delivery_status"
    assert pred.confidence > 0.0


def test_retriever_edge_cases():
    embedder = DummyEmbeddingProvider(dim=8)
    store = FAISSVectorStore(dimension=8)
    retriever = FAISSRetriever(embedder, store)

    # Empty store retrieval
    res_empty = retriever.retrieve("hello", top_k=5)
    assert len(res_empty) == 0

    # Empty query string
    res_query_empty = retriever.retrieve("", top_k=5)
    assert isinstance(res_query_empty, list)
