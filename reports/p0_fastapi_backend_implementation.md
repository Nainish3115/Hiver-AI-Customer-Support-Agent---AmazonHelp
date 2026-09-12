# P0 FastAPI Backend Implementation Report

**Project**: Hiver AI Customer Support Agent (AmazonHelp)  
**Phase**: P0 — FastAPI Backend for Hiver AI Support Agent  
**Date**: September 11, 2026  
**Status**: Implementation Complete & Regression Passed (130/130 Tests Passing)

---

## 1. Architecture

The FastAPI backend is built as a lightweight, non-invasive REST API layer around the validated `CustomerSupportPipeline`. It exposes the frozen ML, retrieval, generation, safety, and decision capabilities directly to web clients without duplicating ML logic or introducing microservice overhead.

```
Web Client (Browser / SPA)
       │
       ▼  (HTTP / JSON)
FastAPI Server (`src/hiver_agent/api/app.py`)
       │
       ├─► GET  /api/v1/health           (Readiness & health check)
       ├─► POST /api/v1/support/process   (End-to-end RAG support execution)
       ├─► GET  /api/v1/stats            (Authoritative benchmark metrics)
       ├─► POST /api/v1/retrieval/search  (RAG historical case corpus search)
       └─► GET  /api/v1/retrieval/search  (RAG query params search)
       │
       ▼
CustomerSupportPipeline (Singleton in app.state)
       │
       ├─► CalibratedRulePrototypeHybrid (Intent Classifier)
       ├─► Hybrid Dense (all-MiniLM-L6-v2) + BM25 (RRF k=60 + Intent Compatibility)
       ├─► Gemini 1.5 Flash (Response Generator)
       ├─► GroundingSafetyChecker (Entity & safety validator)
       └─► RiskAwareDecisionEngine (Policy threshold = 0.35)
```

---

## 2. API Endpoints

| Endpoint | Method | Description | Request Body / Query | Response Model |
|---|---|---|---|---|
| `/api/v1/health` | `GET` | Service & pipeline readiness status | None | `HealthResponse` |
| `/api/v1/support/process` | `POST` | Process inquiry end-to-end | `SupportProcessRequest` | `SupportProcessResponse` |
| `/api/v1/stats` | `GET` | Authoritative validated project metrics | None | `StatsResponse` |
| `/api/v1/retrieval/search` | `POST` | Search historical RAG case corpus | `RetrievalSearchRequest` | `RetrievalSearchResponse` |
| `/api/v1/retrieval/search` | `GET` | Search historical RAG case corpus | `q` (str), `top_k` (int) | `RetrievalSearchResponse` |
| `/docs` | `GET` | OpenAPI interactive Swagger UI | None | HTML |
| `/openapi.json` | `GET` | OpenAPI 3.0 specification | None | JSON |

---

## 3. Request Schema

### `SupportProcessRequest`
```json
{
  "customer_message": "Where is my package? Order ID: 403-1234567-8901234",
  "case_id": "API-TEST-100",
  "policy_type": "risk_aware"
}
```
- `customer_message`: (str, required, 1 to 2000 chars). Validated to reject empty or whitespace-only strings.
- `case_id`: (str, optional). Customer case reference ID.
- `policy_type`: (str, optional). Defaults to `"risk_aware"`.

---

## 4. Response Schema

### `SupportProcessResponse`
```json
{
  "case_id": "API-TEST-100",
  "customer_message": "Where is my package? Order ID: 403-1234567-8901234",
  "predicted_intent": "order_tracking_delivery_status",
  "intent_confidence": 0.85,
  "decision": "AUTO_HANDLE",
  "reason_code": "SAFE_AUTO_HANDLE",
  "decision_reason": "Risk score (0.1170) is below risk threshold (0.3500) and response passed safety checks.",
  "draft_response": "@customer I'm happy to help check the status of your order...",
  "risk_score": 0.117,
  "safety_check_passed": true,
  "grounding_status": "grounded",
  "selected_evidence_count": 3,
  "retrieved_case_ids": ["RAG-AMAZON-669198", "RAG-AMAZON-1719214"],
  "retrieved_scores": [0.8123, 0.7654],
  "signals": {
    "intent_uncertainty_risk": 0.15,
    "category_risk": 0.10,
    "retrieval_quality_risk": 0.12
  },
  "latency_ms": 1140.5
}
```

---

## 5. Pipeline Lifecycle

To avoid reloading models on every request, `CustomerSupportPipeline` is initialized **once** during server startup using FastAPI's `lifespan` context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded ONCE on startup
    pipeline = CustomerSupportPipeline(intent_classifier_mode="calibrated_prototype_hybrid")
    app.state.pipeline = pipeline
    app.state.pipeline_ready = True
    yield
```

Memory, embedding centroids, FAISS vector stores, and sparse indexes are held in application state for zero per-request model loading overhead.

---

## 6. Error Handling

Errors return clean, structured JSON payloads (`APIErrorDetail`):
- `422 Unprocessable Entity`: Request validation failure (e.g. empty or missing `customer_message`).
- `503 Service Unavailable`: Pipeline not initialized or unavailable.
- `500 Internal Server Error`: Generic unhandled server error. Stack traces and API keys are never exposed.

---

## 7. CORS

CORS is configured via FastAPI's `CORSMiddleware` supporting development origins:
- `http://localhost:3000`
- `http://localhost:5173`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`

Configurable via the `ALLOWED_ORIGINS` environment variable for production deployment.

---

## 8. Logging

Structured server logging (`logging.getLogger("hiver_agent.api")`) tracks startup duration, request processing, error events, and latency. Sensitive user credentials and API keys are strictly excluded from logs.

---

## 9. Performance

- **Cold Start Latency**: Startup initialization takes ~2.1s (one-time cost when launching server).
- **Per-Request Latency**: ~1.15s total processing time (dominated by Gemini 1.5 Flash API network call; intent, retrieval, and decision calculations take ~35ms).
- **Concurrency**: FastAPI async ASGI handling prevents event loop blocking.

---

## 10. Security

- **Secrets**: Zero API keys hardcoded. `GEMINI_API_KEY` read safely from environment variables.
- **Input Boundaries**: `customer_message` bounded to max 2000 chars; `top_k` bounded to max 50.
- **Data Protection**: Stack traces and raw internal file paths excluded from API responses.

---

## 11. Tests

8 dedicated API unit tests added in `tests/test_api.py`:
1. `test_health_endpoint`: Verified GET /api/v1/health returns 200 OK.
2. `test_support_process_valid_query`: Verified POST /api/v1/support/process end-to-end execution.
3. `test_support_process_empty_query_rejected`: Verified 422 error on empty/whitespace query.
4. `test_support_process_missing_field_rejected`: Verified 422 error on missing required fields.
5. `test_stats_endpoint`: Verified GET /api/v1/stats exposes verified metrics.
6. `test_retrieval_search_post`: Verified POST /api/v1/retrieval/search.
7. `test_retrieval_search_get`: Verified GET /api/v1/retrieval/search.
8. `test_cors_headers`: Verified CORS preflight headers.

**Test Suite Summary**:
- **Previous Total**: 122 tests passed
- **New Total**: 130 tests passed, 0 failed, 0 skipped (`pytest tests/ -q`)

---

## 12. Local Run Instructions

To start the FastAPI backend server locally:

```bash
# Using uvicorn CLI directly
uvicorn hiver_agent.api.app:app --host 0.0.0.0 --port 8000 --reload

# Or executing app.py directly
python -m hiver_agent.api.app
```

Swagger UI documentation will be available at `http://localhost:8000/docs`.

---

## 13. Known Limitations

1. **Gemini API Network Latency**: Gemini generation API adds ~1.0-1.2s per request.
2. **Synchronous Gemini Client**: Gemini API call is wrapped synchronously; under heavy multi-tenant concurrency, an async Gemini SDK client will further optimize throughput.

---

## 14. Files Created

- `src/hiver_agent/api/__init__.py`
- `src/hiver_agent/api/schemas.py`
- `src/hiver_agent/api/app.py`
- `tests/test_api.py`
- `reports/p0_fastapi_backend_implementation.md`

---

## 15. Files Modified

- `pyproject.toml` (Added `fastapi`, `uvicorn`, `httpx` dependencies)
- `requirements.txt` (Added `fastapi`, `uvicorn`, `httpx` dependencies)
