"""
Hybrid retriever performing Reciprocal Rank Fusion (RRF) between dense and lexical BM25 retrievers.
"""
from typing import List, Dict, Any, Optional
from hiver_agent.retrieval.interfaces import Retriever
from hiver_agent.retrieval.retriever import SearchResult


class HybridRetriever(Retriever):
    """
    Hybrid retriever that fuses candidates from a Dense Retriever and a BM25 Retriever
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_retriever: Retriever,
        rrf_k: int = 60,
        dense_top_k: int = 50,
        bm25_top_k: int = 50,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0
    ):
        # We store references to embedding provider and vector store if available
        emb_provider = getattr(dense_retriever, "embedding_provider", None)
        vec_store = getattr(dense_retriever, "vector_store", None)
        super().__init__(embedding_provider=emb_provider, vector_store=vec_store)

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve top_k historical cases by fusing Dense and BM25 candidates via RRF.
        """
        # 1. Fetch over-sampled candidate lists from both retrievers
        dense_results = self.dense_retriever.retrieve(query, top_k=max(self.dense_top_k, top_k))
        bm25_results = self.bm25_retriever.retrieve(query, top_k=max(self.bm25_top_k, top_k))

        # 2. Build rank maps (1-indexed) and object store by case_id
        dense_ranks: Dict[str, int] = {}
        candidate_store: Dict[str, SearchResult] = {}

        for rank, res in enumerate(dense_results, start=1):
            cid = str(res.case_id)
            dense_ranks[cid] = rank
            candidate_store[cid] = res

        bm25_ranks: Dict[str, int] = {}
        for rank, res in enumerate(bm25_results, start=1):
            cid = str(res.case_id)
            bm25_ranks[cid] = rank
            if cid not in candidate_store:
                candidate_store[cid] = res

        # 3. Compute RRF score for all unique candidate case IDs
        all_case_ids = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
        fused_candidates: List[SearchResult] = []

        for cid in all_case_ids:
            rrf_score = 0.0
            if cid in dense_ranks:
                rrf_score += self.dense_weight / (self.rrf_k + dense_ranks[cid])
            if cid in bm25_ranks:
                rrf_score += self.bm25_weight / (self.rrf_k + bm25_ranks[cid])

            base_res = candidate_store[cid]
            # Copy SearchResult with updated fused RRF score
            fused_res = SearchResult(
                case_id=base_res.case_id,
                score=round(rrf_score, 6),
                customer_problem=base_res.customer_problem,
                support_interaction=base_res.support_interaction,
                resolution=base_res.resolution,
                intent=base_res.intent,
                metadata=base_res.metadata
            )
            fused_candidates.append(fused_res)

        # 4. Sort candidates by RRF score descending
        fused_candidates.sort(key=lambda x: x.score, reverse=True)

        return fused_candidates[:top_k]
