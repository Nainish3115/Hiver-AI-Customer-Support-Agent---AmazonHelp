"""
Grounding safety validation layer.
"""
import re
from typing import List, Tuple, Set
from hiver_agent.generation.interface import EvidenceItem


FORBIDDEN_INTERNAL_TERMS = {
    "faiss", "vector", "embedding", "classifier", "retrieval", "confidence gate",
    "dataset", "jsonl", "parquet", "prompt", "llm", "grounding", "heuristic"
}


class GroundingSafetyChecker:
    """
    Validates drafted responses for grounding safety, hallucination signals,
    internal terminology leakage, unsupported numerical entities, and length bounds.
    """
    def __init__(self, max_length: int = 500):
        self.max_length = max_length

    def validate(
        self,
        draft_response: str,
        customer_message: str,
        evidence_items: List[EvidenceItem]
    ) -> Tuple[bool, List[str]]:
        """
        Validate draft response against customer message and retrieved evidence.
        Returns (passed, list_of_violations).
        """
        violations: List[str] = []

        resp = draft_response.strip()

        # 1. Empty or INSUFFICIENT_EVIDENCE
        if not resp or resp == "INSUFFICIENT_EVIDENCE":
            return True, []

        # 2. Check excessive length
        if len(resp) > self.max_length:
            violations.append(f"Response length ({len(resp)} chars) exceeds maximum allowed ({self.max_length} chars).")

        # 3. Check for internal technical terminology leakage
        resp_lower = resp.lower()
        for term in FORBIDDEN_INTERNAL_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", resp_lower):
                violations.append(f"Internal terminology leakage detected: '{term}'")

        # 4. Check for fabricated order numbers / tracking IDs (#\d+) not present in input or evidence
        evidence_text = customer_message + " " + " ".join(
            [e.customer_problem + " " + e.resolution for e in evidence_items]
        )

        resp_order_ids = set(re.findall(r"#\d{4,}", resp))
        evidence_order_ids = set(re.findall(r"#\d{4,}", evidence_text))

        unsupported_orders = resp_order_ids - evidence_order_ids
        if unsupported_orders:
            violations.append(f"Fabricated order ID(s) detected not present in evidence: {unsupported_orders}")

        # 5. Check for fabricated monetary amounts (\$\d+) not present in input or evidence
        resp_amounts = set(re.findall(r"\$\d+(?:\.\d+)?", resp))
        evidence_amounts = set(re.findall(r"\$\d+(?:\.\d+)?", evidence_text))

        unsupported_amounts = resp_amounts - evidence_amounts
        if unsupported_amounts:
            violations.append(f"Fabricated monetary amount(s) detected not present in evidence: {unsupported_amounts}")

        passed = len(violations) == 0
        return passed, violations
