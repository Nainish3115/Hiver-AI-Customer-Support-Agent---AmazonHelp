"""
BM25 lexical retriever implementation using BM25Okapi algorithm.
Provides deterministic lexical search over historical resolution documents.
"""
import re
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from hiver_agent.retrieval.interfaces import Retriever
from hiver_agent.retrieval.retriever import SearchResult


def tokenize_text(text: str) -> List[str]:
    """
    Simple, deterministic word tokenizer for BM25.
    Converts text to lowercase and extracts alphanumeric tokens.
    """
    if not text:
        return []
    return re.findall(r'\w+', text.lower())


class BM25Retriever(Retriever):
    """
    Lexical BM25 retriever using BM25Okapi over historical case documents.
    """
    def __init__(self, documents: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        # We pass None for embedding_provider and vector_store to super() since BM25 is purely lexical
        super().__init__(embedding_provider=None, vector_store=None) # type: ignore
        self.documents = documents
        self.k1 = k1
        self.b = b

        # Extract and tokenize document text for corpus
        self.corpus_tokens: List[List[str]] = []
        for doc in documents:
            text = self._extract_doc_text(doc)
            self.corpus_tokens.append(tokenize_text(text))

        self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)

    def _extract_doc_text(self, doc: Dict[str, Any]) -> str:
        """
        Extract full indexable text from a document payload dict.
        """
        text = doc.get("document_text", "")
        if not text:
            prob = doc.get("customer_problem", "")
            interact = doc.get("support_interaction", "")
            res = doc.get("resolution", "")
            text = f"{prob} {interact} {res}".strip()
        return text

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve top_k historical cases matching query lexically using BM25.
        """
        if not self.documents or not query or not query.strip():
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []


        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-K indices sorted by score descending
        actual_k = min(top_k, len(self.documents))
        top_indices = np.argsort(scores)[::-1][:actual_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                # Still include candidate if we need top_k, but keep score
                pass

            doc = self.documents[idx]
            meta = doc.get("metadata", {})
            doc_text = doc.get("document_text", "")
            intent = doc.get("intent", meta.get("intent", "general_inquiry_other"))

            prob = doc.get("customer_problem", "")
            interact = doc.get("support_interaction", doc_text)
            res = doc.get("resolution", "")

            if not prob and "CUSTOMER PROBLEM: " in doc_text:
                prob = doc_text.split("CUSTOMER PROBLEM: ")[1].split("\nSUPPORT INTERACTION:")[0]
            if not res and "RESOLUTION: " in doc_text:
                res = doc_text.split("RESOLUTION: ")[-1]

            results.append(
                SearchResult(
                    case_id=str(doc.get("case_id", "RAG-UNKNOWN")),
                    score=score,
                    customer_problem=prob,
                    support_interaction=interact,
                    resolution=res,
                    intent=intent,
                    metadata=meta
                )
            )

        return results

    @classmethod
    def from_vector_store(cls, vector_store: Any, k1: float = 1.5, b: float = 0.75) -> "BM25Retriever":
        """
        Construct a BM25Retriever from a FAISSVectorStore instance's documents.
        """
        docs = getattr(vector_store, "documents", [])
        return cls(documents=docs, k1=k1, b=b)
