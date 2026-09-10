"""
Evidence selection and deduplication layer.
"""
from typing import List
from hiver_agent.generation.interface import EvidenceItem
from hiver_agent.classification.dataset import normalize_text_for_grouping


class EvidenceSelector:
    """
    Filters, deduplicates, and prioritizes retrieved historical evidence items.
    """
    def __init__(self, max_evidence: int = 3, confidence_threshold: float = 0.60):
        self.max_evidence = max_evidence
        self.confidence_threshold = confidence_threshold

    def select_evidence(
        self,
        evidence_items: List[EvidenceItem],
        predicted_intent: str,
        intent_confidence: float
    ) -> List[EvidenceItem]:
        """
        Deduplicate and select top-K relevant evidence items.
        """
        if not evidence_items:
            return []

        # 1. Deduplicate based on normalized resolution text
        unique_items: List[EvidenceItem] = []
        seen_resolutions = set()

        for item in evidence_items:
            res_text = item.resolution.strip()
            if not res_text or res_text.lower() in {"insufficient_evidence", "none", "n/a"}:
                continue

            norm_res = normalize_text_for_grouping(res_text)
            if not norm_res or norm_res in seen_resolutions:
                continue

            seen_resolutions.add(norm_res)
            unique_items.append(item)

        if not unique_items:
            return []

        # 2. Prioritize intent-matching evidence if confidence >= threshold
        use_intent_priority = (intent_confidence >= self.confidence_threshold)

        def score_key(item: EvidenceItem):
            intent_match = 1 if (use_intent_priority and item.intent == predicted_intent) else 0
            return (intent_match, item.score)

        # Sort descending by (intent_match, score)
        sorted_items = sorted(unique_items, key=score_key, reverse=True)

        return sorted_items[:self.max_evidence]
