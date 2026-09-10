"""
Evidence representation and formatting for grounded response generation.
"""
from typing import List, Dict, Any
from hiver_agent.retrieval.retriever import SearchResult
from hiver_agent.generation.interface import EvidenceItem


class EvidenceFormatter:
    """
    Formats raw retrieval search results into clean, structured EvidenceItem objects
    and prompt-ready text blocks.
    """
    @staticmethod
    def from_search_result(result: SearchResult) -> EvidenceItem:
        """
        Convert a SearchResult object to a structured EvidenceItem.
        """
        # Ensure resolution text is non-empty
        res = result.resolution.strip()
        if not res and result.support_interaction.strip():
            res = result.support_interaction.strip()

        prob = result.customer_problem.strip()
        if not prob:
            prob = "Customer reported issue."

        return EvidenceItem(
            case_id=result.case_id,
            customer_problem=prob,
            support_interaction=result.support_interaction.strip(),
            resolution=res,
            intent=result.intent,
            score=float(result.score)
        )

    @staticmethod
    def format_search_results(results: List[SearchResult]) -> List[EvidenceItem]:
        """
        Convert a list of SearchResults into EvidenceItems.
        """
        return [EvidenceFormatter.from_search_result(r) for r in results]

    @staticmethod
    def format_evidence_prompt_context(evidence_items: List[EvidenceItem]) -> str:
        """
        Format selected evidence items into a clean, deterministic numbered context block for LLM prompts.
        """
        if not evidence_items:
            return "NO HISTORICAL EVIDENCE AVAILABLE."

        blocks = []
        for i, item in enumerate(evidence_items, 1):
            block = (
                f"--- EVIDENCE CASE #{i} (Case ID: {item.case_id}, Intent: {item.intent}) ---\n"
                f"Customer Query: {item.customer_problem}\n"
                f"Historical Resolution: {item.resolution}\n"
            )
            blocks.append(block)

        return "\n".join(blocks)
