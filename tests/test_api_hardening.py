"""
P2 API Hardening Test Suite.
Tests Request ID propagation, readiness probes, input validation, structured error responses,
and singleton pipeline reuse.
"""
import pytest
from fastapi.testclient import TestClient
from hiver_agent.api.app import app
from hiver_agent.pipeline import CustomerSupportPipeline
from hiver_agent.generation.generator import LLMResponseGenerator, MockLLMGenerator, TemplateResponseGenerator, EvidenceItem


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_request_id_generated_when_missing(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert "X-Request-ID" in res.headers
    req_id = res.headers["X-Request-ID"]
    assert req_id.startswith("req-")
    data = res.json()
    assert data.get("request_id") == req_id


def test_incoming_request_id_preserved(client):
    custom_id = "req-custom-test-12345"
    res = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert res.status_code == 200
    assert res.headers.get("X-Request-ID") == custom_id
    data = res.json()
    assert data.get("request_id") == custom_id


def test_health_liveness_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "hiver-ai-support-agent"
    assert data["pipeline_ready"] is True


def test_ready_readiness_endpoint(client):
    res = client.get("/api/v1/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["pipeline_ready"] is True
    assert data["vector_store_documents"] >= 15000


def test_support_process_validation_empty_message(client):
    res = client.post("/api/v1/support/process", json={"customer_message": "   "})
    assert res.status_code == 422
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_REQUEST"


def test_support_process_validation_invalid_policy(client):
    res = client.post(
        "/api/v1/support/process",
        json={"customer_message": "Where is my refund?", "policy_type": "invalid_policy_name"}
    )
    assert res.status_code == 422
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_REQUEST"


def test_retrieval_validation_empty_query(client):
    res = client.post("/api/v1/retrieval/search", json={"query": "  ", "top_k": 10})
    assert res.status_code == 422
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_REQUEST"


def test_retrieval_validation_top_k_bounds(client):
    res_low = client.post("/api/v1/retrieval/search", json={"query": "my package", "top_k": 0})
    assert res_low.status_code == 422

    res_high = client.post("/api/v1/retrieval/search", json={"query": "my package", "top_k": 100})
    assert res_high.status_code == 422


def test_gemini_failure_fallback_generator():
    # Test LLMResponseGenerator fallback behavior when GEMINI_API_KEY is invalid/missing
    gen = LLMResponseGenerator(api_key="INVALID_TEST_KEY_123", model_name="invalid-model")
    evidence = [
        EvidenceItem(
            case_id="TEST-CASE-1",
            customer_problem="Item missing",
            support_interaction="Please contact support",
            resolution="Contact support via link",
            intent="late_delivery_shipping_delay",
            score=0.95
        )
    ]
    res = gen.generate(
        customer_message="Where is my package?",
        predicted_intent="late_delivery_shipping_delay",
        intent_confidence=0.90,
        retrieval_mode="intent_aware",
        evidence=evidence
    )
    assert res.draft_response is not None
    assert len(res.draft_response) > 0
    assert res.safety_check_passed is True


def test_singleton_pipeline_reuse(client):
    p1 = app.state.pipeline
    res1 = client.post("/api/v1/support/process", json={"customer_message": "Order tracking status update please"})
    assert res1.status_code == 200

    p2 = app.state.pipeline
    res2 = client.post("/api/v1/support/process", json={"customer_message": "How to get a refund?"})
    assert res2.status_code == 200

    # Ensure the exact same singleton pipeline instance was reused
    assert p1 is p2
