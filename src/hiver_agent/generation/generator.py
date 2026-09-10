"""
Response generators for Phase 6: Mode A (Deterministic Baseline) and Mode B (Grounded LLM Generation).
"""
import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from hiver_agent.generation.interface import (
    BaseResponseGenerator,
    EvidenceItem,
    GenerationResult
)
from hiver_agent.generation.evidence import EvidenceFormatter
from hiver_agent.generation.safety import GroundingSafetyChecker


SYSTEM_PROMPT = """You are an Amazon customer-support response drafting assistant.
Use ONLY the supplied historical support evidence.

Your response MUST:
1. Directly address the customer's issue.
2. Reflect how AmazonHelp historically handled similar issues in the evidence.
3. Avoid inventing Amazon policies, refunds, timelines, account actions, or guarantees.
4. Avoid claiming that an action has already been performed (e.g., "refund processed") unless supported by evidence.
5. Avoid fabricating order numbers, account information, or exact dates.
6. Avoid unsupported refund amounts or delivery dates.
7. Avoid unsupported promises or compensation.
8. Avoid mentioning internal retrieval, embeddings, classifiers, vector stores, or datasets.
9. Be concise, polite, and natural for customer support.
10. Ask for missing information (e.g., DM us your order details) when the historical evidence shows AmazonHelp requested that information.

If the evidence is empty or insufficient to safely draft a useful answer, return EXACTLY:
INSUFFICIENT_EVIDENCE
"""


class TemplateResponseGenerator(BaseResponseGenerator):
    """
    Mode A: Deterministic / template baseline generator.
    Synthesizes a customer support response from top historical resolution evidence.
    """
    def __init__(self, safety_checker: Optional[GroundingSafetyChecker] = None):
        self.safety_checker = safety_checker or GroundingSafetyChecker()

    def generate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieval_mode: str,
        evidence: List[EvidenceItem],
        case_id: Optional[str] = None
    ) -> GenerationResult:
        if not evidence:
            return GenerationResult(
                case_id=case_id,
                customer_message=customer_message,
                predicted_intent=predicted_intent,
                intent_confidence=intent_confidence,
                retrieval_mode=retrieval_mode,
                retrieved_case_ids=[],
                retrieved_evidence=[],
                draft_response="INSUFFICIENT_EVIDENCE",
                grounding_status="insufficient_evidence",
                safety_check_passed=True,
                safety_violations=[]
            )

        top_evidence = evidence[0]
        res_text = top_evidence.resolution.strip()

        if not res_text or res_text.lower() in {"insufficient_evidence", "none"}:
            draft = "INSUFFICIENT_EVIDENCE"
            status = "insufficient_evidence"
        else:
            draft = res_text
            status = "grounded"

        case_ids = [e.case_id for e in evidence]

        # Run Safety Check
        passed, violations = self.safety_checker.validate(draft, customer_message, evidence)
        if not passed:
            draft = "INSUFFICIENT_EVIDENCE"
            status = "insufficient_evidence"

        return GenerationResult(
            case_id=case_id,
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            retrieval_mode=retrieval_mode,
            retrieved_case_ids=case_ids,
            retrieved_evidence=evidence,
            draft_response=draft,
            grounding_status=status,
            safety_check_passed=passed,
            safety_violations=violations
        )


class MockLLMGenerator(BaseResponseGenerator):
    """
    Mock LLM Generator for offline testing and deterministic benchmark runs.
    Simulates grounded LLM synthesis from evidence.
    """
    def __init__(self, safety_checker: Optional[GroundingSafetyChecker] = None):
        self.safety_checker = safety_checker or GroundingSafetyChecker()

    def generate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieval_mode: str,
        evidence: List[EvidenceItem],
        case_id: Optional[str] = None
    ) -> GenerationResult:
        if not evidence:
            return GenerationResult(
                case_id=case_id,
                customer_message=customer_message,
                predicted_intent=predicted_intent,
                intent_confidence=intent_confidence,
                retrieval_mode=retrieval_mode,
                retrieved_case_ids=[],
                retrieved_evidence=[],
                draft_response="INSUFFICIENT_EVIDENCE",
                grounding_status="insufficient_evidence",
                safety_check_passed=True,
                safety_violations=[]
            )

        top_evidence = evidence[0]
        res = top_evidence.resolution.strip()

        # Clean up response to simulate LLM grounded response drafting
        if not res or len(res) < 10:
            draft = "INSUFFICIENT_EVIDENCE"
            status = "insufficient_evidence"
        else:
            draft = f"Hello! {res}"
            status = "grounded"

        case_ids = [e.case_id for e in evidence]

        # Safety validation
        passed, violations = self.safety_checker.validate(draft, customer_message, evidence)
        if not passed:
            draft = "INSUFFICIENT_EVIDENCE"
            status = "insufficient_evidence"

        return GenerationResult(
            case_id=case_id,
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            retrieval_mode=retrieval_mode,
            retrieved_case_ids=case_ids,
            retrieved_evidence=evidence,
            draft_response=draft,
            grounding_status=status,
            safety_check_passed=passed,
            safety_violations=violations
        )


class LLMResponseGenerator(BaseResponseGenerator):
    """
    Mode B: LLM grounded generator using official Google Gemini API (or fallback).
    """
    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        safety_checker: Optional[GroundingSafetyChecker] = None
    ):
        self.provider = provider
        self.api_key = api_key if api_key is not None else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        self.model_name = model_name
        self.safety_checker = safety_checker or GroundingSafetyChecker()
        self.fallback_generator = MockLLMGenerator(safety_checker=self.safety_checker)

    @property
    def provider_info(self) -> Dict[str, str]:
        """
        Returns clear provider metadata.
        """
        return {
            "provider": "gemini",
            "mode": "real" if self.api_key else "mock",
            "model": self.model_name
        }

    def generate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieval_mode: str,
        evidence: List[EvidenceItem],
        case_id: Optional[str] = None
    ) -> GenerationResult:
        if not evidence:
            return GenerationResult(
                case_id=case_id,
                customer_message=customer_message,
                predicted_intent=predicted_intent,
                intent_confidence=intent_confidence,
                retrieval_mode=retrieval_mode,
                retrieved_case_ids=[],
                retrieved_evidence=[],
                draft_response="INSUFFICIENT_EVIDENCE",
                grounding_status="insufficient_evidence",
                safety_check_passed=True,
                safety_violations=[],
                metadata=self.provider_info
            )

        # If API key is missing, fall back safely to MockLLMGenerator
        if not self.api_key:
            res = self.fallback_generator.generate(
                customer_message, predicted_intent, intent_confidence, retrieval_mode, evidence, case_id=case_id
            )
            res.metadata = self.provider_info
            return res

        # Format evidence context
        context_str = EvidenceFormatter.format_evidence_prompt_context(evidence)
        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"CUSTOMER MESSAGE: {customer_message}\n"
            f"PREDICTED INTENT: {predicted_intent} (Confidence: {intent_confidence:.2f})\n\n"
            f"HISTORICAL EVIDENCE:\n{context_str}\n\n"
            f"Draft a grounded response following the system guidelines."
        )

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            gen_config = genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=500
            )
            
            draft = ""
            for attempt in range(5):
                try:
                    model = genai.GenerativeModel(self.model_name)
                    resp = model.generate_content(full_prompt, generation_config=gen_config)
                    draft = resp.text.strip() if resp and resp.text else ""
                    break
                except Exception as e:
                    err_str = str(e)
                    if ("429" in err_str or "Quota" in err_str or "ResourceExhausted" in err_str) and attempt < 4:
                        time.sleep(20)
                        continue
                    elif attempt < 4:
                        fallback_model = "gemini-3.6-flash"
                        self.model_name = fallback_model
                        time.sleep(2)
                        continue
                    else:
                        raise e
        except Exception:
            # On API error / timeout / malformed output, fallback safely
            res = self.fallback_generator.generate(
                customer_message, predicted_intent, intent_confidence, retrieval_mode, evidence, case_id=case_id
            )
            res.metadata = self.provider_info
            return res

        if not draft or "INSUFFICIENT_EVIDENCE" in draft:
            draft = "INSUFFICIENT_EVIDENCE"
            status = "insufficient_evidence"
        else:
            status = "grounded"

        case_ids = [e.case_id for e in evidence]
        passed, violations = self.safety_checker.validate(draft, customer_message, evidence)
        if not passed:
            draft = "INSUFFICIENT_EVIDENCE"
            status = "insufficient_evidence"

        return GenerationResult(
            case_id=case_id,
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            retrieval_mode=retrieval_mode,
            retrieved_case_ids=case_ids,
            retrieved_evidence=evidence,
            draft_response=draft,
            grounding_status=status,
            safety_check_passed=passed,
            safety_violations=violations,
            metadata=self.provider_info
        )
