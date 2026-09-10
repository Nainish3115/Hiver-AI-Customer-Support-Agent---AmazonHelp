"""
Semantic Cross-Encoder Reranker for fine-grained passage similarity re-ordering.
"""
from typing import List, Optional, Dict, Any
from sentence_transformers import CrossEncoder

from hiver_agent.retrieval.retriever import SearchResult


class SemanticReranker:
    """
    Reranks candidate search results using a joint query-document cross-encoder model.
    """
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 128,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self._model = CrossEncoder(model_name, max_length=max_length, device=device)

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
        batch_size: int = 128
    ) -> List[SearchResult]:
        if not candidates:
            return []

        # Construct (query, candidate_document) pairs
        pairs = []
        for cand in candidates:
            doc_text = f"PROBLEM: {cand.customer_problem}\nRESOLUTION: {cand.resolution}"
            pairs.append((query, doc_text))

        # Predict cross-encoder relevance scores
        scores = self._model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        if isinstance(scores, (float, int)):
            scores = [scores]

        # Attach reranker_score to metadata and build re-ordered candidates list
        reranked_candidates = []
        for cand, score in zip(candidates, scores):
            meta = dict(cand.metadata) if cand.metadata else {}
            meta["reranker_score"] = float(score)
            meta["original_score"] = float(cand.score)

            reranked_candidates.append(
                SearchResult(
                    case_id=cand.case_id,
                    score=float(score),
                    customer_problem=cand.customer_problem,
                    support_interaction=cand.support_interaction,
                    resolution=cand.resolution,
                    intent=cand.intent,
                    metadata=meta
                )
            )

        # Sort descending by cross-encoder score
        reranked_candidates.sort(key=lambda x: x.score, reverse=True)
        return reranked_candidates[:top_k]
