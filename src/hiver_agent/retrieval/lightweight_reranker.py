"""
Lightweight Deterministic Reranker for AmazonHelp retrieval candidates.

Phase 14 Day 6 Implementation:
- Combines RRF rank, normalized dense similarity, lexical token overlap, and predicted intent compatibility.
- Zero neural network reranker dependency (100% CPU deterministic scoring).
"""
import re
from typing import List, Dict, Any, Optional
import numpy as np
from hiver_agent.retrieval.retriever import SearchResult

# Common English stopwords to ignore during lexical overlap calculation
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which",
    "this", "that", "these", "those", "then", "just", "so", "than", "such",
    "when", "who", "how", "where", "why", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "can", "could", "should",
    "would", "will", "my", "your", "his", "her", "its", "our", "their", "me", "him", "them",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up", "about", "into",
    "over", "after", "please", "help", "amazon", "support"
}


def tokenize_query_clean(text: str) -> List[str]:
    """
    Tokenize text into lowercase alphanumeric tokens excluding stopwords.
    """
    if not text:
        return []
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit()]


def compute_lexical_overlap(query_tokens: List[str], doc_text: str) -> float:
    """
    Compute normalized token overlap ratio between query tokens and document text.
    """
    if not query_tokens or not doc_text:
        return 0.0
    doc_tokens = set(re.findall(r'\w+', doc_text.lower()))
    if not doc_tokens:
        return 0.0

    matched = sum(1 for t in query_tokens if t in doc_tokens)
    return float(matched / len(query_tokens))


class LightweightReranker:
    """
    Lightweight Deterministic Scoring Layer.
    Fuses normalized RRF, Dense similarity, Lexical overlap, and Predicted Intent.
    """
    def __init__(
        self,
        config_name: str = "hybrid_intent_lexical_dense",
        weight_rrf: float = 0.35,
        weight_intent: float = 0.25,
        weight_dense: float = 0.20,
        weight_lexical: float = 0.20
    ):
        self.config_name = config_name
        self.weight_rrf = weight_rrf
        self.weight_intent = weight_intent
        self.weight_dense = weight_dense
        self.weight_lexical = weight_lexical

    def rerank(
        self,
        query: str,
        predicted_intent: str,
        intent_confidence: float,
        candidates: List[SearchResult],
        top_k: int = 10
    ) -> List[SearchResult]:
        if not candidates:
            return []

        query_tokens = tokenize_query_clean(query)

        # Extract raw score vectors for min-max normalization
        raw_rrf_scores = np.array([c.score for c in candidates])
        min_rrf, max_rrf = np.min(raw_rrf_scores), np.max(raw_rrf_scores)
        denom_rrf = (max_rrf - min_rrf) if (max_rrf - min_rrf) > 0 else 1.0
        norm_rrf = (raw_rrf_scores - min_rrf) / denom_rrf

        # Dense similarities from candidate metadata if present, else fallback to norm_rrf
        dense_sims = []
        for c in candidates:
            d_sim = c.metadata.get("dense_score", c.score)
            dense_sims.append(d_sim)
        raw_dense = np.array(dense_sims)
        min_d, max_d = np.min(raw_dense), np.max(raw_dense)
        denom_d = (max_d - min_d) if (max_d - min_d) > 0 else 1.0
        norm_dense = (raw_dense - min_d) / denom_d

        reranked_candidates = []
        for idx, cand in enumerate(candidates):
            # 1. Normalized RRF Score
            s_rrf = float(norm_rrf[idx])

            # 2. Normalized Dense Score
            s_dense = float(norm_dense[idx])

            # 3. Lexical Overlap
            doc_full = f"{cand.customer_problem} {cand.support_interaction} {cand.resolution}"
            s_lexical = compute_lexical_overlap(query_tokens, doc_full)

            # 4. Intent Compatibility
            s_intent = 1.0 if (intent_confidence >= 0.60 and cand.intent == predicted_intent) else 0.0

            # Composite Score Calculation according to weights
            composite_score = (
                (self.weight_rrf * s_rrf) +
                (self.weight_intent * s_intent) +
                (self.weight_dense * s_dense) +
                (self.weight_lexical * s_lexical)
            )

            # Copy candidate with updated lightweight score
            updated_meta = dict(cand.metadata)
            updated_meta.update({
                "lightweight_score": round(composite_score, 6),
                "signal_rrf_norm": round(s_rrf, 4),
                "signal_dense_norm": round(s_dense, 4),
                "signal_lexical": round(s_lexical, 4),
                "signal_intent": round(s_intent, 4)
            })

            res = SearchResult(
                case_id=cand.case_id,
                score=round(composite_score, 6),
                customer_problem=cand.customer_problem,
                support_interaction=cand.support_interaction,
                resolution=cand.resolution,
                intent=cand.intent,
                metadata=updated_meta
            )
            reranked_candidates.append(res)

        # Sort by composite score descending
        reranked_candidates.sort(key=lambda x: x.score, reverse=True)
        return reranked_candidates[:top_k]
