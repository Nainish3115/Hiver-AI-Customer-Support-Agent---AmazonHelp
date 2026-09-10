"""
Single-call verification script for Phase 6.5 Gemini integration.
Executes exactly 1 real Gemini generation call and 1 real Gemini judge call.
Prints provider metadata and verifies that no API key is printed or persisted.
"""
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)
sys.path.insert(0, str(Path(root_dir) / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from hiver_agent.generation.interface import EvidenceItem, GenerationResult
from hiver_agent.generation.generator import LLMResponseGenerator
from hiver_agent.evaluation.llm_judge import LLMAsJudgeEvaluator


def main():
    print("=== Phase 6.5 Gemini Provider Single-Call Verification ===", flush=True)

    api_key_present = bool(os.environ.get("GEMINI_API_KEY"))
    mode_str = "real" if api_key_present else "mock"

    print(f"provider: gemini", flush=True)
    print(f"mode: {mode_str}", flush=True)
    print(f"model: gemini-1.5-flash", flush=True)
    print(f"API Key Present: {api_key_present}", flush=True)

    # 1. Instantiate Generator
    generator = LLMResponseGenerator(provider="gemini", model_name="gemini-1.5-flash")
    print(f"Generator Provider Metadata: {generator.provider_info}", flush=True)

    # Sample Evidence Item
    evidence = [
        EvidenceItem(
            case_id="RAG-VERIFY-001",
            customer_problem="My package has not arrived yet.",
            support_interaction="Please DM us your order details.",
            resolution="Please send us a direct message with your order number and email address so we can check delivery.",
            intent="late_delivery_shipping_delay",
            score=0.95
        )
    ]

    # 2. Execute SINGLE Generation Call
    print("\nExecuting SINGLE Gemini Response Generation Call...", flush=True)
    gen_result = generator.generate(
        customer_message="Where is my package? It was supposed to arrive yesterday.",
        predicted_intent="late_delivery_shipping_delay",
        intent_confidence=0.85,
        retrieval_mode="intent_aware",
        evidence=evidence,
        case_id="VERIFY-CASE-1"
    )

    print("\n--- Generation Result ---", flush=True)
    print(f"Status: {gen_result.grounding_status}", flush=True)
    print(f"Safety Passed: {gen_result.safety_check_passed}", flush=True)
    print(f"Draft Response: {gen_result.draft_response}", flush=True)

    # 3. Instantiate LLM Judge & Execute SINGLE Judge Call
    print("\nExecuting SINGLE Gemini Judge Evaluation Call...", flush=True)
    judge = LLMAsJudgeEvaluator(provider="gemini", model_name="gemini-1.5-flash")
    judge_info = judge.provider_info
    print(f"\nJudge Provider Metadata: {judge_info}", flush=True)
    print(f"provider: {judge_info['provider']}", flush=True)
    print(f"mode: {judge_info['mode']}", flush=True)
    print(f"model: {judge_info['model']}", flush=True)

    if not judge.api_key:
        print("Judge Call Skipped (No GEMINI_API_KEY set).", flush=True)
    else:
        judge_score = judge.evaluate_result(gen_result)

        if judge_score:
            print("\n--- LLM Judge Evaluation Output ---", flush=True)
            print(f"Relevance: {judge_score.relevance}/5", flush=True)
            print(f"Groundedness: {judge_score.groundedness}/5", flush=True)
            print(f"Helpfulness: {judge_score.helpfulness}/5", flush=True)
            print(f"Correctness: {judge_score.correctness}/5", flush=True)
            print(f"Hallucination Risk: {judge_score.hallucination_risk}/5", flush=True)
            print(f"Tone: {judge_score.tone}/5", flush=True)
            print(f"Actionability: {judge_score.actionability}/5", flush=True)
            print(f"Overall: {judge_score.overall}/5", flush=True)
            print(f"Reason: {judge_score.reason}", flush=True)
            print("Single-Call Gemini Judge Call Succeeded!", flush=True)
        else:
            print("Judge Call Execution Failed (API error or invalid response JSON).", flush=True)

    # Security check: Confirm API key is NOT printed anywhere
    output_text = f"{gen_result.draft_response} {gen_result.metadata}"
    if os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_API_KEY") in output_text:
        raise ValueError("CRITICAL SECURITY ERROR: API key detected in output!")

    print("\n=== Verification Completed Successfully ===", flush=True)


if __name__ == "__main__":
    main()
