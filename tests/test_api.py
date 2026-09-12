"""
Unit tests for FastAPI REST API endpoints (Phase P0).
"""
import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
sys.path.insert(0, str(Path(root_dir) / "src"))

from hiver_agent.api.app import app
from hiver_agent.pipeline import CustomerSupportPipeline
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.retrieval.faiss_store import FAISSVectorStore
from hiver_agent.retrieval.retriever import FAISSRetriever
from hiver_agent.generation.generator import TemplateResponseGenerator


@pytest.fixture(scope="module")
def api_client():
    """
    Module-scoped TestClient fixture with pre-initialized Mock/Template CustomerSupportPipeline.
    """
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2", batch_size=128)
    faiss_dir = Path("data/processed/faiss_index_15k")
    vector_store = FAISSVectorStore.load(str(faiss_dir))
    base_retriever = FAISSRetriever(embedder, vector_store)
    classifier = RuleBasedIntentClassifier()
    generator = TemplateResponseGenerator()

    pipeline = CustomerSupportPipeline(
        classifier=classifier,
        base_retriever=base_retriever,
        generator=generator,
        confidence_threshold=0.60
    )

    # Attach initialized pipeline to app.state
    app.state.pipeline = pipeline
    app.state.pipeline_ready = True

    with TestClient(app) as client:
        yield client


def test_health_endpoint(api_client):
    """Test GET /api/v1/health returns status ok and pipeline_ready True."""
    res = api_client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "hiver-ai-support-agent"
    assert data["pipeline_ready"] is True


def test_support_process_valid_query(api_client):
    """Test POST /api/v1/support/process with valid customer inquiry."""
    payload = {
        "customer_message": "Where is my package? Order ID: 403-1234567-8901234",
        "case_id": "API-TEST-100",
        "policy_type": "risk_aware"
    }
    res = api_client.post("/api/v1/support/process", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["case_id"] == "API-TEST-100"
    assert data["customer_message"] == "Where is my package? Order ID: 403-1234567-8901234"
    assert data["predicted_intent"] is not None
    assert 0.0 <= data["intent_confidence"] <= 1.0
    assert data["decision"] in {"AUTO_HANDLE", "ESCALATE_TO_HUMAN"}
    assert data["reason_code"] is not None
    assert data["decision_reason"] is not None
    assert data["draft_response"] is not None
    assert "latency_ms" in data
    assert data["latency_ms"] > 0.0


def test_support_process_empty_query_rejected(api_client):
    """Test POST /api/v1/support/process rejects empty and whitespace-only queries with 422."""
    # 1. Empty string
    res1 = api_client.post("/api/v1/support/process", json={"customer_message": ""})
    assert res1.status_code == 422

    # 2. Whitespace-only string
    res2 = api_client.post("/api/v1/support/process", json={"customer_message": "   \n\t  "})
    assert res2.status_code == 422


def test_support_process_missing_field_rejected(api_client):
    """Test POST /api/v1/support/process rejects request missing customer_message."""
    res = api_client.post("/api/v1/support/process", json={"case_id": "NO-MSG"})
    assert res.status_code == 422


def test_stats_endpoint(api_client):
    """Test GET /api/v1/stats exposes verified project metrics."""
    res = api_client.get("/api/v1/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "hiver-ai-support-agent"
    assert data["canonical_corpus_size"] == 15316
    assert "metrics" in data
    metrics = data["metrics"]
    assert metrics["intent_accuracy_untouched_test"]["value"] == 0.57
    assert metrics["escalation_false_auto_handle_rate"]["value"] == 0.06


def test_retrieval_search_post(api_client):
    """Test POST /api/v1/retrieval/search returns search results."""
    payload = {"query": "how do I track my package?", "top_k": 5}
    res = api_client.post("/api/v1/retrieval/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "how do I track my package?"
    assert len(data["results"]) == 5
    assert data["results"][0]["case_id"] is not None


def test_retrieval_search_get(api_client):
    """Test GET /api/v1/retrieval/search returns search results via query params."""
    res = api_client.get("/api/v1/retrieval/search?q=refund+status&top_k=3")
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "refund status"
    assert len(data["results"]) == 3


def test_cors_headers(api_client):
    """Test CORS headers for local development origin."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET"
    }
    res = api_client.options("/api/v1/health", headers=headers)
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"
