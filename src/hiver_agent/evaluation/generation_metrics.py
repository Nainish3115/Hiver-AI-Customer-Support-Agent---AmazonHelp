"""
Automated evaluation metrics for Phase 6 response generation.
"""
from typing import List, Dict, Any
import numpy as np
from hiver_agent.generation.interface import GenerationResult


def evaluate_generation_results(results: List[GenerationResult]) -> Dict[str, float]:
    """
    Calculate automated generation and grounding quality metrics over a list of GenerationResults.
    """
    if not results:
        return {
            "total_cases": 0,
            "validity_rate": 0.0,
            "insufficient_evidence_rate": 0.0,
            "evidence_citation_rate": 0.0,
            "unsupported_claim_rate": 0.0,
            "avg_response_length": 0.0,
            "exact_copy_rate": 0.0,
            "intent_aware_mode_rate": 0.0
        }

    total = len(results)
    valid_count = 0
    insufficient_count = 0
    evidence_cited_count = 0
    unsupported_count = 0
    exact_copy_count = 0
    intent_aware_count = 0
    response_lengths = []

    for r in results:
        # Validity
        if r.grounding_status == "grounded" and r.safety_check_passed:
            valid_count += 1

        # Insufficient evidence
        if r.draft_response == "INSUFFICIENT_EVIDENCE" or r.grounding_status == "insufficient_evidence":
            insufficient_count += 1

        # Evidence citation
        if r.retrieved_evidence:
            evidence_cited_count += 1

        # Unsupported claims / safety failure
        if not r.safety_check_passed:
            unsupported_count += 1

        # Retrieval mode
        if r.retrieval_mode == "intent_aware":
            intent_aware_count += 1

        # Length and exact copy check
        if r.draft_response != "INSUFFICIENT_EVIDENCE":
            response_lengths.append(len(r.draft_response))
            if r.retrieved_evidence:
                top_res = r.retrieved_evidence[0].resolution.strip()
                if r.draft_response.strip() == top_res:
                    exact_copy_count += 1

    return {
        "total_cases": total,
        "validity_rate": round(valid_count / total, 4),
        "insufficient_evidence_rate": round(insufficient_count / total, 4),
        "evidence_citation_rate": round(evidence_cited_count / total, 4),
        "unsupported_claim_rate": round(unsupported_count / total, 4),
        "avg_response_length": round(float(np.mean(response_lengths)) if response_lengths else 0.0, 2),
        "exact_copy_rate": round(exact_copy_count / total, 4),
        "intent_aware_mode_rate": round(intent_aware_count / total, 4)
    }
