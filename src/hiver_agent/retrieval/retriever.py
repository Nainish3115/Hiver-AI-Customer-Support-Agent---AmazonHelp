"""
High-level retriever implementations (FAISSRetriever and IntentAwareRetriever).
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from hiver_agent.retrieval.interfaces import Retriever, EmbeddingProvider, VectorStore
from hiver_agent.classification.interface import IntentClassifier


class SearchResult(BaseModel):
    """
    Schema for structured retrieval search results.
    """
    case_id: str = Field(..., description="Unique historical case ID")
    score: float = Field(..., description="Similarity score (higher is more similar)")
    customer_problem: str = Field(..., description="Customer initial problem description")
    support_interaction: str = Field(..., description="Support interaction body")
    resolution: str = Field(..., description="Final support resolution response text")
    intent: str = Field(..., description="Historical case intent label")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class FAISSRetriever(Retriever):
    """
    Pure semantic vector retriever backed by FAISSVectorStore.
    """
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        q_emb = self.embedding_provider.embed_query(query)
        raw_results = self.vector_store.search(q_emb, top_k=top_k)

        search_results = []
        for r in raw_results:
            meta = r.get("metadata", {})
            doc_text = r.get("document_text", "")
            intent = r.get("intent", meta.get("intent", "general_inquiry_other"))

            # Parse problem / interaction / resolution from doc_text if fields missing
            prob = r.get("customer_problem", "")
            interact = r.get("support_interaction", doc_text)
            res = r.get("resolution", "")

            if not prob and "CUSTOMER PROBLEM: " in doc_text:
                prob = doc_text.split("CUSTOMER PROBLEM: ")[1].split("\nSUPPORT INTERACTION:")[0]
            if not res and "RESOLUTION: " in doc_text:
                res = doc_text.split("RESOLUTION: ")[-1]

            search_results.append(
                SearchResult(
                    case_id=r.get("case_id", "RAG-UNKNOWN"),
                    score=float(r.get("_score", 0.0)),
                    customer_problem=prob,
                    support_interaction=interact,
                    resolution=res,
                    intent=intent,
                    metadata=meta
                )
            )

        return search_results


class IntentAwareRetriever(Retriever):
    """
    Intent-aware retriever that predicts query intent and applies metadata filtering/boosting.
    """
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        classifier: IntentClassifier,
        confidence_threshold: float = 0.60,
        intent_boost_factor: float = 0.15,
        base_retriever: Optional[Retriever] = None
    ):
        super().__init__(embedding_provider, vector_store)
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self.intent_boost_factor = intent_boost_factor
        self.base_retriever = base_retriever or FAISSRetriever(embedding_provider, vector_store)


    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # 1. Predict query intent
        pred = self.classifier.predict(query)
        
        # Over-retrieve candidates for reranking
        candidate_k = max(top_k * 4, 20)
        candidates = self.base_retriever.retrieve(query, top_k=candidate_k)

        # 2. If confidence >= threshold, apply intent matching boost
        if pred.confidence >= self.confidence_threshold:
            target_intent = pred.intent
            for cand in candidates:
                if cand.intent == target_intent:
                    cand.score += self.intent_boost_factor

            # Re-sort by boosted score
            candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates[:top_k]
