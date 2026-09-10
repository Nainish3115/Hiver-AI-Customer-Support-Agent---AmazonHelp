"""
Unit tests for Phase 6 Historically Grounded Response Generation.
"""
import pytest
import numpy as np
from hiver_agent.generation.interface import EvidenceItem, GenerationResult
from hiver_agent.generation.evidence import EvidenceFormatter
from hiver_agent.generation.evidence_selector import EvidenceSelector
from hiver_agent.generation.safety import GroundingSafetyChecker
from hiver_agent.generation.generator import TemplateResponseGenerator, MockLLMGenerator, LLMResponseGenerator
from hiver_agent.pipeline import CustomerSupportPipeline
from hiver_agent.evaluation.generation_metrics import evaluate_generation_results
from hiver_agent.evaluation.llm_judge import LLMAsJudgeEvaluator, LLMJudgeScore
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from hiver_agent.retrieval.retriever import SearchResult, FAISSRetriever
from hiver_agent.config import load_config


class DummyEmbeddingProvider:
    """
    Fast mock embedding provider for unit tests.
    """
    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def embed_query(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % 100000)
        vec = np.random.randn(self.dimension).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed_query(t) for t in texts], dtype=np.float32)


class DummyVectorStore:
    """
    Mock vector store for retrieval testing.
    """
    def __init__(self, docs: list[dict]):
        self.docs = docs

    def add_documents(self, documents, embeddings):
        self.docs.extend(documents)

    def search(self, query_embedding, top_k=5):
        results = []
        for i, d in enumerate(self.docs[:top_k]):
            item = dict(d)
            item["_score"] = 0.95 - (i * 0.05)
            results.append(item)
        return results

    def count(self):
        return len(self.docs)


def test_generation_result_schema():
    """Test GenerationResult and EvidenceItem schema instantiation."""
    ev = EvidenceItem(
        case_id="C-101",
        customer_problem="Where is my order?",
        support_interaction="Please DM your details.",
        resolution="Please send us a DM with your order number and email.",
        intent="order_tracking_delivery_status",
        score=0.92
    )
    res = GenerationResult(
        case_id="C-201",
        customer_message="My package is delayed",
        predicted_intent="late_delivery_shipping_delay",
        intent_confidence=0.85,
        retrieval_mode="intent_aware",
        retrieved_case_ids=["C-101"],
        retrieved_evidence=[ev],
        draft_response="Please DM us your order number.",
        grounding_status="grounded",
        safety_check_passed=True,
        safety_violations=[]
    )
    assert res.predicted_intent == "late_delivery_shipping_delay"
    assert res.grounding_status == "grounded"
    assert len(res.retrieved_evidence) == 1
    assert res.retrieved_evidence[0].case_id == "C-101"


def test_evidence_formatter_and_deduplication():
    """Test EvidenceFormatter and EvidenceSelector deduplication."""
    search_res1 = SearchResult(
        case_id="C1",
        score=0.90,
        customer_problem="Order delayed",
        support_interaction="Please DM us",
        resolution="Please send us a DM with order details.",
        intent="late_delivery_shipping_delay"
    )
    search_res2 = SearchResult(
        case_id="C2",
        score=0.85,
        customer_problem="Order delayed again",
        support_interaction="Please DM us",
        resolution="Please send us a DM with order details.",  # Duplicate resolution
        intent="late_delivery_shipping_delay"
    )
    search_res3 = SearchResult(
        case_id="C3",
        score=0.80,
        customer_problem="Refund request",
        support_interaction="Refund processed",
        resolution="We have issued a full refund to your original payment method.",
        intent="refund_status_request"
    )

    ev_items = EvidenceFormatter.format_search_results([search_res1, search_res2, search_res3])
    assert len(ev_items) == 3

    selector = EvidenceSelector(max_evidence=2, confidence_threshold=0.60)
    selected = selector.select_evidence(ev_items, predicted_intent="refund_status_request", intent_confidence=0.80)

    # Should deduplicate C2 because it has identical resolution to C1
    # Should prioritize C3 because it matches predicted_intent (refund_status_request)
    assert len(selected) == 2
    assert selected[0].case_id == "C3"


def test_grounding_safety_checker():
    """Test GroundingSafetyChecker detection of unsupported entities and internal terms."""
    checker = GroundingSafetyChecker(max_length=100)

    ev = EvidenceItem(
        case_id="C1",
        customer_problem="Late package",
        support_interaction="DM us",
        resolution="Please DM order details.",
        intent="late_delivery_shipping_delay",
        score=0.90
    )

    # 1. Valid safe response
    passed, v = checker.validate("Please send us a DM with order info.", "My order is late", [ev])
    assert passed
    assert len(v) == 0

    # 2. Terminology leakage (FAISS)
    passed, v = checker.validate("Our FAISS vector search found your issue.", "My order is late", [ev])
    assert not passed
    assert any("faiss" in err.lower() for err in v)

    # 3. Fabricated order ID
    passed, v = checker.validate("Your order #999999 has been refunded.", "My order is late", [ev])
    assert not passed
    assert any("order id" in err.lower() for err in v)

    # 4. Fabricated dollar amount
    passed, v = checker.validate("We issued $150.00 to your account.", "My order is late", [ev])
    assert not passed
    assert any("monetary amount" in err.lower() for err in v)


def test_pipeline_confidence_gating_and_fallback():
    """Test CustomerSupportPipeline confidence gating and semantic fallback."""
    embedder = DummyEmbeddingProvider()
    docs = [
        {
            "case_id": "C-01",
            "customer_text": "Where is my package?",
            "customer_problem": "Where is my package?",
            "support_interaction": "Please DM us.",
            "resolution": "Please send us a DM with your order number.",
            "intent": "order_tracking_delivery_status"
        },
        {
            "case_id": "C-02",
            "customer_text": "I want to cancel order",
            "customer_problem": "I want to cancel order",
            "support_interaction": "Cancellation processed",
            "resolution": "Your order cancellation request has been logged.",
            "intent": "order_cancellation_request"
        }
    ]
    store = DummyVectorStore(docs)
    base_retriever = FAISSRetriever(embedder, store)

    rule_clf = RuleBasedIntentClassifier()
    gen = TemplateResponseGenerator()

    pipeline = CustomerSupportPipeline(
        classifier=rule_clf,
        base_retriever=base_retriever,
        generator=gen,
        confidence_threshold=0.60
    )

    # High confidence query -> intent_aware
    res_high = pipeline.process_message("Where is my order? tracking status please")
    assert res_high.retrieval_mode == "intent_aware"
    assert res_high.intent_confidence >= 0.60

    # Low confidence query -> semantic_fallback
    res_low = pipeline.process_message("random general query without keywords")
    assert res_low.retrieval_mode == "semantic_fallback"
    assert res_low.intent_confidence < 0.60


def test_self_case_exclusion():
    """Verify that process_message excludes target self-case from evidence."""
    embedder = DummyEmbeddingProvider()
    docs = [
        {
            "case_id": "TARGET-123",
            "customer_text": "Need help with login password reset",
            "customer_problem": "Need help with login password reset",
            "support_interaction": "Reset link sent",
            "resolution": "We sent a password reset link to your registered email.",
            "intent": "account_access_login"
        },
        {
            "case_id": "OTHER-456",
            "customer_text": "Cannot sign into account",
            "customer_problem": "Cannot sign into account",
            "support_interaction": "DM us",
            "resolution": "Please DM us your account email address.",
            "intent": "account_access_login"
        }
    ]
    store = DummyVectorStore(docs)
    base_retriever = FAISSRetriever(embedder, store)
    rule_clf = RuleBasedIntentClassifier()
    gen = TemplateResponseGenerator()

    pipeline = CustomerSupportPipeline(
        classifier=rule_clf,
        base_retriever=base_retriever,
        generator=gen
    )

    res = pipeline.process_message(
        "Need help with login password reset",
        case_id="TARGET-123",
        exclude_case_id="TARGET-123"
    )

    # TARGET-123 must NOT be in retrieved case IDs
    assert "TARGET-123" not in res.retrieved_case_ids


def test_mock_llm_generator_and_insufficient_evidence():
    """Test MockLLMGenerator behavior with and without evidence."""
    gen = MockLLMGenerator()

    # Empty evidence -> INSUFFICIENT_EVIDENCE
    res_empty = gen.generate("Hello", "general_inquiry_other", 0.5, "semantic_fallback", [])
    assert res_empty.draft_response == "INSUFFICIENT_EVIDENCE"
    assert res_empty.grounding_status == "insufficient_evidence"

    # With evidence -> grounded response
    ev = EvidenceItem(
        case_id="C1",
        customer_problem="Late package",
        support_interaction="Please DM us",
        resolution="Please send us a DM with your order number.",
        intent="late_delivery_shipping_delay",
        score=0.90
    )
    res_valid = gen.generate("Where is package", "late_delivery_shipping_delay", 0.85, "intent_aware", [ev])
    assert res_valid.grounding_status == "grounded"
    assert "Please send us a DM" in res_valid.draft_response


def test_generation_config_loading():
    """Verify AppConfig parses generation settings properly."""
    cfg = load_config()
    assert hasattr(cfg, "generation")
    assert cfg.generation.confidence_threshold == 0.60
    assert cfg.generation.top_k_evidence == 3


def test_gemini_provider_metadata_and_mock_fallback():
    """Test LLMResponseGenerator provider metadata and missing key fallback to MockLLMGenerator."""
    gen = LLMResponseGenerator(provider="gemini", model_name="gemini-1.5-flash")
    info = gen.provider_info
    assert info["provider"] == "gemini"
    assert info["model"] == "gemini-1.5-flash"
    assert info["mode"] in {"real", "mock"}

    ev = EvidenceItem(
        case_id="C1",
        customer_problem="Late package",
        support_interaction="Please DM us",
        resolution="Please send us a DM with your order details.",
        intent="late_delivery_shipping_delay",
        score=0.90
    )

    res = gen.generate("Where is my package", "late_delivery_shipping_delay", 0.85, "intent_aware", [ev])
    assert res.metadata["provider"] == "gemini"
    assert res.grounding_status == "grounded"


def test_gemini_llm_judge_missing_key():
    """Test LLMAsJudgeEvaluator missing key behavior."""
    judge = LLMAsJudgeEvaluator(api_key="", model_name="gemini-1.5-flash")
    res = GenerationResult(
        customer_message="My order is delayed",
        predicted_intent="late_delivery_shipping_delay",
        intent_confidence=0.80,
        retrieval_mode="intent_aware",
        draft_response="Please send us a DM.",
        grounding_status="grounded"
    )
    score = judge.evaluate_result(res)
    assert score is None

    batch_m = judge.evaluate_batch([res])
    assert "SKIPPED" in batch_m["judge_status"]


def test_gemini_llm_judge_provider_metadata_and_with_key():
    """Test LLMAsJudgeEvaluator provider metadata with key present and absent."""
    # 1. With key
    judge_real = LLMAsJudgeEvaluator(provider="gemini", api_key="mock_key_123", model_name="gemini-1.5-flash")
    meta_real = judge_real.provider_info
    assert meta_real["provider"] == "gemini"
    assert meta_real["mode"] == "real"
    assert meta_real["model"] == "gemini-1.5-flash"
    assert "mock_key_123" not in str(meta_real)

    # 2. Missing key
    judge_mock = LLMAsJudgeEvaluator(provider="gemini", api_key="", model_name="gemini-1.5-flash")
    meta_mock = judge_mock.provider_info
    assert meta_mock["provider"] == "gemini"
    assert meta_mock["mode"] == "mock"
    assert meta_mock["model"] == "gemini-1.5-flash"


def test_gemini_llm_judge_successful_structured_response(monkeypatch):
    """Test LLMAsJudgeEvaluator returning valid structured evaluation JSON."""
    judge = LLMAsJudgeEvaluator(provider="gemini", api_key="mock_key_123", model_name="gemini-1.5-flash")
    res = GenerationResult(
        customer_message="My order is delayed",
        predicted_intent="late_delivery_shipping_delay",
        intent_confidence=0.80,
        retrieval_mode="intent_aware",
        draft_response="Please send us a DM with your order details.",
        grounding_status="grounded"
    )

    fake_json = """{
      "relevance": 5,
      "groundedness": 5,
      "helpfulness": 4,
      "correctness": 5,
      "hallucination_risk": 5,
      "tone": 5,
      "actionability": 4,
      "overall": 5,
      "reason": "The draft response is accurate, grounded, and polite."
    }"""

    class MockGenerativeModel:
        def __init__(self, model_name):
            pass
        def generate_content(self, prompt, generation_config=None):
            class MockResponse:
                text = fake_json
            return MockResponse()

    import google.generativeai as genai
    monkeypatch.setattr(genai, "GenerativeModel", MockGenerativeModel)
    monkeypatch.setattr(genai, "configure", lambda api_key: None)

    score = judge.evaluate_result(res)
    assert score is not None
    assert isinstance(score, LLMJudgeScore)
    assert score.relevance == 5
    assert score.groundedness == 5
    assert score.helpfulness == 4
    assert score.correctness == 5
    assert score.hallucination_risk == 5
    assert score.tone == 5
    assert score.actionability == 4
    assert score.overall == 5
    assert "accurate, grounded, and polite" in score.reason

    batch_res = judge.evaluate_batch([res])
    assert batch_res["judge_status"] == "COMPLETED"
    assert batch_res["evaluated_count"] == 1
    assert batch_res["avg_overall"] == 5.0


def test_gemini_llm_judge_malformed_json(monkeypatch):
    """Test LLMAsJudgeEvaluator handling malformed JSON response gracefully."""
    judge = LLMAsJudgeEvaluator(provider="gemini", api_key="mock_key_123", model_name="gemini-1.5-flash")
    res = GenerationResult(
        customer_message="My order is delayed",
        predicted_intent="late_delivery_shipping_delay",
        intent_confidence=0.80,
        retrieval_mode="intent_aware",
        draft_response="Please send us a DM.",
        grounding_status="grounded"
    )

    class MockGenerativeModel:
        def __init__(self, model_name):
            pass
        def generate_content(self, prompt, generation_config=None):
            class MockResponse:
                text = "INVALID_NON_JSON_RESPONSE"
            return MockResponse()

    import google.generativeai as genai
    monkeypatch.setattr(genai, "GenerativeModel", MockGenerativeModel)
    monkeypatch.setattr(genai, "configure", lambda api_key: None)

    score = judge.evaluate_result(res)
    assert score is None

    batch_res = judge.evaluate_batch([res])
    assert "FAILED" in batch_res["judge_status"]


def test_gemini_llm_judge_api_exception(monkeypatch):
    """Test LLMAsJudgeEvaluator handling Gemini API exceptions gracefully."""
    judge = LLMAsJudgeEvaluator(provider="gemini", api_key="mock_key_123", model_name="gemini-1.5-flash")
    res = GenerationResult(
        customer_message="My order is delayed",
        predicted_intent="late_delivery_shipping_delay",
        intent_confidence=0.80,
        retrieval_mode="intent_aware",
        draft_response="Please send us a DM.",
        grounding_status="grounded"
    )

    class MockGenerativeModel:
        def __init__(self, model_name):
            pass
        def generate_content(self, prompt, generation_config=None):
            raise RuntimeError("API Connection Error / Timeout")

    import google.generativeai as genai
    monkeypatch.setattr(genai, "GenerativeModel", MockGenerativeModel)
    monkeypatch.setattr(genai, "configure", lambda api_key: None)

    score = judge.evaluate_result(res)
    assert score is None


