"""
Unit tests for Phase 8 End-to-End Evaluation & System Integration.
"""
import sys
import pytest
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)
sys.path.insert(0, str(Path(root_dir) / "src"))

from hiver_agent.pipeline import CustomerSupportPipeline, EndToEndResult
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from scripts.run_phase_5_6_experiments import HybridRuleEmbeddingClassifier
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.retrieval.faiss_store import FAISSVectorStore
from hiver_agent.retrieval.retriever import FAISSRetriever
from hiver_agent.generation.generator import TemplateResponseGenerator
from hiver_agent.config import load_config
from scripts.evaluate_phase8_end_to_end import perform_leakage_and_security_audits, run_intent_benchmark, run_retrieval_benchmark


@pytest.fixture(scope="module")
def pipeline():
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2", batch_size=128)
    faiss_dir = Path("data/processed/faiss_index_15k")
    vector_store = FAISSVectorStore.load(str(faiss_dir))
    base_retriever = FAISSRetriever(embedder, vector_store)

    rule_clf = RuleBasedIntentClassifier()
    proto_clf = EmbeddingPrototypeClassifier(embedder)
    # Fit dummy on dev texts
    proto_clf.fit(["Where is my order?", "I want a refund"], ["order_tracking_delivery_status", "refund_status_request"])
    hybrid_clf = HybridRuleEmbeddingClassifier(rule_clf, proto_clf)

    generator = TemplateResponseGenerator()
    return CustomerSupportPipeline(
        classifier=hybrid_clf,
        base_retriever=base_retriever,
        generator=generator,
        confidence_threshold=0.60
    )


def test_end_to_end_result_contract(pipeline):
    """Test process_end_to_end API and EndToEndResult data contract."""
    res = pipeline.process_end_to_end("Where is my package? Order ID: 403-1234567-8901234", case_id="TEST-100")

    assert isinstance(res, EndToEndResult)
    assert res.case_id == "TEST-100"
    assert res.customer_message == "Where is my package? Order ID: 403-1234567-8901234"
    assert res.predicted_intent is not None
    assert 0.0 <= res.intent_confidence <= 1.0
    assert res.decision in {"AUTO_HANDLE", "ESCALATE_TO_HUMAN"}
    assert res.reason_code is not None
    assert res.decision_reason is not None
    assert isinstance(res.safety_check_passed, bool)


def test_self_retrieval_exclusion_leakage(pipeline):
    """Test that pipeline strictly excludes target case_id and duplicate text from retrieved evidence."""
    target_case_id = "RAG-AMAZON-101739"
    # Query with exact case_id and message
    res = pipeline.process_end_to_end("Love when I order something on @115821", case_id=target_case_id, exclude_case_id=target_case_id)

    # Confirm target_case_id is not in retrieved_case_ids
    assert target_case_id not in res.retrieved_case_ids, f"Leakage detected: {target_case_id} was retrieved."


def test_security_audit_no_committed_secrets():
    """Test that no raw Gemini API key secrets exist in source code."""
    repo_src = Path("src")
    secrets = []
    for fpath in repo_src.rglob("*.py"):
        text = fpath.read_text(encoding="utf-8")
        if "AIzaSy" in text:
            secrets.append(str(fpath))

    assert len(secrets) == 0, f"Security violation: Hardcoded API keys found in {secrets}"
