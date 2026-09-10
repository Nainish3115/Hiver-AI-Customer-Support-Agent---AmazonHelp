"""
LLM-as-Judge evaluation layer for Phase 6 response generation.
"""
import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from hiver_agent.generation.interface import GenerationResult
from hiver_agent.generation.evidence import EvidenceFormatter


class LLMJudgeScore(BaseModel):
    """
    Structured evaluation schema for LLM-as-Judge.
    All numeric scores are on a 1-5 Likert scale.
    """
    relevance: int = Field(..., ge=1, le=5, description="Relevance to customer query")
    groundedness: int = Field(..., ge=1, le=5, description="Groundedness in retrieved evidence")
    helpfulness: int = Field(..., ge=1, le=5, description="Helpfulness of resolution")
    correctness: int = Field(..., ge=1, le=5, description="Factual correctness relative to evidence")
    hallucination_risk: int = Field(..., ge=1, le=5, description="1=High hallucination risk, 5=Zero hallucination risk")
    tone: int = Field(..., ge=1, le=5, description="Professionalism and tone")
    actionability: int = Field(..., ge=1, le=5, description="Clear next steps / actionability")
    overall: int = Field(..., ge=1, le=5, description="Overall response quality score")
    reason: str = Field(..., description="Concise justification of evaluation score")


class LLMAsJudgeEvaluator:
    """
    Evaluates generated customer support responses using Google Gemini as LLM-as-Judge.
    If no GEMINI_API_KEY is available, skips gracefully and reports skipped status.
    """
    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash"
    ):
        self.provider = provider
        self.api_key = api_key if api_key is not None else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        self.model_name = model_name

    @property
    def provider_info(self) -> Dict[str, str]:
        """
        Returns clear provider metadata without exposing the API key.
        """
        return {
            "provider": self.provider,
            "mode": "real" if self.api_key else "mock",
            "model": self.model_name
        }

    def evaluate_result(self, result: GenerationResult) -> Optional[LLMJudgeScore]:
        """
        Evaluate a single GenerationResult. Returns LLMJudgeScore or None if skipped.
        """
        if not self.api_key:
            return None

        evidence_str = EvidenceFormatter.format_evidence_prompt_context(result.retrieved_evidence)
        judge_prompt = f"""You are an expert AI Customer Support Evaluator.
Evaluate the following generated support response strictly against the supplied historical evidence and customer message.

CUSTOMER MESSAGE: {result.customer_message}
PREDICTED INTENT: {result.predicted_intent} (Confidence: {result.intent_confidence:.2f})

HISTORICAL EVIDENCE:
{evidence_str}

GENERATED RESPONSE TO EVALUATE:
{result.draft_response}

Score each dimension from 1 (poor/high risk) to 5 (excellent/zero risk).
Return ONLY a valid JSON object matching this exact schema:
{{
  "relevance": 1-5,
  "groundedness": 1-5,
  "helpfulness": 1-5,
  "correctness": 1-5,
  "hallucination_risk": 1-5,
  "tone": 1-5,
  "actionability": 1-5,
  "overall": 1-5,
  "reason": "short explanation"
}}
"""
        try:
            import re
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            gen_config = genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
            raw_txt = ""
            for attempt in range(5):
                try:
                    model = genai.GenerativeModel(self.model_name)
                    resp = model.generate_content(judge_prompt, generation_config=gen_config)
                    raw_txt = resp.text.strip() if resp and resp.text else ""
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
            
            # Clean markdown code blocks if present
            clean_txt = re.sub(r"^```(?:json)?\s*", "", raw_txt, flags=re.IGNORECASE)
            clean_txt = re.sub(r"\s*```$", "", clean_txt).strip()

            data = json.loads(clean_txt)
            return LLMJudgeScore(**data)
        except Exception:
            return None

    def evaluate_batch(self, results: List[GenerationResult]) -> Dict[str, Any]:
        """
        Evaluate a list of GenerationResults and compute summary averages.
        """
        if not self.api_key:
            return {
                "judge_status": "SKIPPED (No GEMINI_API_KEY set)",
                "evaluated_count": 0,
                "provider_info": self.provider_info,
                "avg_relevance": 0.0,
                "avg_groundedness": 0.0,
                "avg_helpfulness": 0.0,
                "avg_correctness": 0.0,
                "avg_hallucination_risk": 0.0,
                "avg_tone": 0.0,
                "avg_actionability": 0.0,
                "avg_overall": 0.0
            }

        scores: List[LLMJudgeScore] = []
        for r in results:
            s = self.evaluate_result(r)
            if s:
                scores.append(s)

        if not scores:
            return {
                "judge_status": "FAILED (API error or empty evaluations)",
                "evaluated_count": 0,
                "provider_info": self.provider_info,
                "avg_relevance": 0.0,
                "avg_groundedness": 0.0,
                "avg_helpfulness": 0.0,
                "avg_correctness": 0.0,
                "avg_hallucination_risk": 0.0,
                "avg_tone": 0.0,
                "avg_actionability": 0.0,
                "avg_overall": 0.0
            }

        return {
            "judge_status": "COMPLETED",
            "evaluated_count": len(scores),
            "provider_info": self.provider_info,
            "avg_relevance": round(sum(s.relevance for s in scores) / len(scores), 2),
            "avg_groundedness": round(sum(s.groundedness for s in scores) / len(scores), 2),
            "avg_helpfulness": round(sum(s.helpfulness for s in scores) / len(scores), 2),
            "avg_correctness": round(sum(s.correctness for s in scores) / len(scores), 2),
            "avg_hallucination_risk": round(sum(s.hallucination_risk for s in scores) / len(scores), 2),
            "avg_tone": round(sum(s.tone for s in scores) / len(scores), 2),
            "avg_actionability": round(sum(s.actionability for s in scores) / len(scores), 2),
            "avg_overall": round(sum(s.overall for s in scores) / len(scores), 2)
        }

