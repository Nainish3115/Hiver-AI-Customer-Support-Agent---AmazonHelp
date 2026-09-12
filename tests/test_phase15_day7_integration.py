"""
Unit tests for Day 7 Final Integration and Classifier Mode Switch.
"""
import sys
import pytest
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
sys.path.insert(0, str(Path(root_dir) / "src"))

from hiver_agent.pipeline import CustomerSupportPipeline, EndToEndResult
from hiver_agent.classification.gold_prototype_classifier import CalibratedRulePrototypeHybrid
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.retrieval.faiss_store import FAISSVectorStore
from hiver_agent.retrieval.retriever import FAISSRetriever
from hiver_agent.generation.generator import TemplateResponseGenerator


@pytest.fixture(scope="module")
def shared_components():
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2", batch_size=128)
    faiss_dir = Path("data/processed/faiss_index_15k")
    vector_store = FAISSVectorStore.load(str(faiss_dir))
    base_retriever = FAISSRetriever(embedder, vector_store)
    generator = TemplateResponseGenerator()
    return base_retriever, generator


def test_pipeline_default_calibrated_hybrid_integration(shared_components):
    """Test that default CustomerSupportPipeline constructs CalibratedRulePrototypeHybrid."""
    base_retriever, generator = shared_components
    pipeline = CustomerSupportPipeline(
        base_retriever=base_retriever,
        generator=generator,
        intent_classifier_mode="calibrated_prototype_hybrid"
    )

    assert isinstance(pipeline.classifier, CalibratedRulePrototypeHybrid)
    assert pipeline.intent_classifier_mode == "calibrated_prototype_hybrid"

    res = pipeline.process_end_to_end("Where is my package? Order # 403-1234567-8901234", case_id="DAY7-TEST-1")
    assert isinstance(res, EndToEndResult)
    assert res.case_id == "DAY7-TEST-1"
    assert res.predicted_intent is not None
    assert 0.0 <= res.intent_confidence <= 1.0
    assert res.decision in {"AUTO_HANDLE", "ESCALATE_TO_HUMAN"}


def test_pipeline_legacy_classifier_mode_switch(shared_components):
    """Test that setting intent_classifier_mode='legacy' constructs RuleBasedIntentClassifier."""
    base_retriever, generator = shared_components
    pipeline = CustomerSupportPipeline(
        base_retriever=base_retriever,
        generator=generator,
        intent_classifier_mode="legacy"
    )

    assert isinstance(pipeline.classifier, RuleBasedIntentClassifier)
    assert pipeline.intent_classifier_mode == "legacy"

    res = pipeline.process_end_to_end("I need a refund for my order", case_id="DAY7-TEST-2")
    assert isinstance(res, EndToEndResult)
    assert res.predicted_intent is not None
