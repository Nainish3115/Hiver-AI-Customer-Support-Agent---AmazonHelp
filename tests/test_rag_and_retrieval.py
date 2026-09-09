"""
Deterministic unit tests for Phase 3: RAG corpus building, document formatting,
taxonomy validation structures, and abstract retrieval interfaces.
"""
import pytest
import json
import numpy as np
import pandas as pd

from hiver_agent.data.golden_builder import build_taxonomy_validation_sample
from hiver_agent.data.rag_builder import (
    extract_case_resolution,
    format_rag_document_text,
    build_amazonhelp_rag_corpus
)
from hiver_agent.retrieval.interfaces import EmbeddingProvider, VectorStore, Retriever


# Mock implementations for testing retrieval interfaces without dependencies
class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 16):
        self._dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), self._dim), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return np.ones((1, self._dim), dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


class MockVectorStore(VectorStore):
    def __init__(self):
        self.docs = []
        self.vectors = None

    def add_documents(self, documents: list[dict[str, any]], embeddings: np.ndarray) -> None:
        self.docs.extend(documents)
        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict[str, any]]:
        return self.docs[:top_k]

    def count(self) -> int:
        return len(self.docs)


class MockRetriever(Retriever):
    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, any]]:
        q_emb = self.embedding_provider.embed_query(query)
        return self.vector_store.search(q_emb, top_k=top_k)


def test_rag_document_formatting():
    doc_text = format_rag_document_text(
        intent="order_tracking_delivery_status",
        customer_problem="Where is my order?",
        support_interaction="[cust]: Where is my order?\n[AmazonHelp]: Please check tracking link.",
        resolution="Please check tracking link."
    )
    assert "INTENT: order_tracking_delivery_status" in doc_text
    assert "CUSTOMER PROBLEM: Where is my order?" in doc_text
    assert "RESOLUTION: Please check tracking link." in doc_text


def test_extract_case_resolution():
    tweets = [
        {"author_id": "cust1", "inbound": True, "text": "My package hasn't arrived."},
        {"author_id": "AmazonHelp", "inbound": False, "text": "We have re-dispatched your item. Here is the new tracking link."}
    ]
    prob, interact, res, has_res = extract_case_resolution(tweets)
    assert prob == "My package hasn't arrived."
    assert "re-dispatched your item" in res
    assert has_res is True


def test_extract_case_resolution_dm_redirection():
    tweets = [
        {"author_id": "cust1", "inbound": True, "text": "Late delivery issue."},
        {"author_id": "AmazonHelp", "inbound": False, "text": "Please DM us to assist."}
    ]
    prob, interact, res, has_res = extract_case_resolution(tweets)
    assert has_res is False  # DM redirection without resolution text marked False


def test_mock_retrieval_interfaces():
    embedder = MockEmbeddingProvider(dim=8)
    store = MockVectorStore()
    retriever = MockRetriever(embedder, store)

    docs = [{"case_id": "1", "text": "tracking issue"}, {"case_id": "2", "text": "refund issue"}]
    embs = embedder.embed_texts(["tracking issue", "refund issue"])
    store.add_documents(docs, embs)

    assert store.count() == 2
    results = retriever.retrieve("where is my order?", top_k=1)
    assert len(results) == 1
    assert results[0]["case_id"] == "1"
