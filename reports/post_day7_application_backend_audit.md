# Post-Day 7 Application & Backend Audit

**Project**: Hiver AI Customer Support Agent (AmazonHelp)  
**Phase**: Audit Before Productization  
**Date**: September 11, 2026  
**Status**: Comprehensive Technical Audit (No Code/System Modifications Made)

---

## 1. Executive Summary

This technical audit assesses the Hiver AI Customer Support Agent repository following Day 7 final validation. With all core machine learning, retrieval, safety, generation, and escalation models validated and frozen (122 unit tests passing, 57.0% untouched test intent accuracy, 6.0% false auto-handle rate, 100% safety pass rate), the goal of this phase is to transition the project from a command-line ML research pipeline into a recruiter-ready, portfolio-grade AI support product.

**Key Audit Findings**:
- **Frontend Status**: No web frontend currently exists in the repository. The user interface must be created.
- **Backend/API Status**: No web API framework (e.g., FastAPI, Flask) is currently implemented. Pipeline calls are executed via direct Python script invocations.
- **Pipeline Readability**: `CustomerSupportPipeline` in `src/hiver_agent/pipeline.py` exposes a clean, unified `process_end_to_end()` API that returns a structured `EndToEndResult` Pydantic model suitable for direct REST serialization.
- **State & Performance**: Loading embedding models, FAISS vector indexes (15,316 documents), BM25 indexes, and prototypes requires ~2.1s on startup. Per-query processing takes ~1.1s (dominated by Gemini API network round-trip). Models must be initialized as singletons on server startup.
- **Security & Hygiene**: `.env` is ignored by git, zero hardcoded API keys exist in source code, and data/artifacts are correctly scoped.

---

## 2. Current Repository Architecture

The codebase is organized into a clean Python library structure under `src/hiver_agent/`:

```
d:\hiver-support-agent\
├── configs/
│   └── config.yaml                 # Pydantic-backed configuration schemas
├── data/
│   ├── golden/                     # 200 Human Gold & 50 Escalation Validation sets
│   ├── processed/faiss_index_15k/  # Indexed 15,316 AmazonHelp vector index
│   └── raw/                        # Raw Twitter dataset (git-ignored)
├── reports/                        # 50 markdown reports & tables/ CSV directory
├── scripts/                        # 36 evaluation, benchmark, and data scripts
├── src/hiver_agent/
│   ├── classification/             # Intent classifiers (CalibratedRulePrototypeHybrid, etc.)
│   ├── decision/                   # RiskAwareDecisionEngine & RiskAwarePolicy
│   ├── generation/                 # Gemini 1.5 Flash generator & GroundingSafetyChecker
│   ├── retrieval/                  # FAISSRetriever, BM25, RRF, IntentAwareRetriever
│   ├── config.py                   # AppConfig loader
│   └── pipeline.py                 # CustomerSupportPipeline & EndToEndResult contract
└── tests/                          # 122 pytest unit & integration tests
```

---

## 3. Existing Frontend Audit

- **Framework**: None present.
- **Routing / Components**: None.
- **Quality**: N/A (Non-existent).
- **Classification**:
  - **A. KEEP**: None.
  - **B. IMPROVE**: None.
  - **C. REBUILD / CREATE NEW**: Build a sleek, responsive, glassmorphic Web UI.
  - **D. REMOVE**: None.

---

## 4. Existing Backend / API Audit

- **Framework**: No Web API framework currently installed or configured.
- **Endpoints**: None exist.
- **Pipeline Lifecycle**: Currently, evaluation scripts instantiate `CustomerSupportPipeline` per execution. In a web server context, the pipeline must be instantiated **once** during server startup (`lifespan` handler) and attached to application state (`app.state.pipeline`).
- **Concurrency & Caching**: Model loading (`SentenceTransformer`, `FAISSVectorStore`, `rank_bm25`, prototype centroids) is thread-safe for reads and should remain cached in memory.
- **Error Handling & CORS**: Need to implement FastAPI exception handlers and CORS middleware for frontend communication.

---

## 5. CustomerSupportPipeline Interface

**Primary Python Entry Point**:
```python
pipeline.process_end_to_end(
    customer_message: str,
    case_id: Optional[str] = None,
    exclude_case_id: Optional[str] = None,
    policy_type: str = "risk_aware"
) -> EndToEndResult
```

**Returned `EndToEndResult` Structure**:
- `customer_message`: (str) Input query.
- `predicted_intent`: (str) Taxonomy intent.
- `intent_confidence`: (float) Intent prediction score (0.0 to 1.0).
- `retrieved_case_ids`: (List[str]) Retrieved AmazonHelp case IDs.
- `retrieved_scores`: (List[float]) Vector/BM25 similarity scores.
- `selected_evidence_count`: (int) Evidence items formatted into LLM context.
- `draft_response`: (str) Grounded Gemini 1.5 Flash output response.
- `grounding_status`: (str) `GROUNDED` vs `UNGROUNDED` flag.
- `safety_check_passed`: (bool) Entity/safety checker result.
- `safety_violations`: (List[str]) Safety violation descriptions if any.
- `risk_score`: (float) Composite risk score computed by `RiskAwarePolicy`.
- `decision`: (str) `AUTO_HANDLE` vs `ESCALATE_TO_HUMAN`.
- `reason_code`: (str) Trigger code (e.g. `SAFE_AUTO_HANDLE`, `COMPOSITE_RISK_EXCEEDED`).
- `decision_reason`: (str) Natural language decision rationale.

---

## 6. Frontend Data Requirements

Field exposure classification for the product interface:

| Field | Classification | UI Purpose / Render Details |
|---|---|---|
| `customer_message` | **REQUIRED** | User message bubble in Live Demo chat card. |
| `predicted_intent` | **REQUIRED** | Intent badge (e.g., `order_tracking_delivery_status`). |
| `intent_confidence` | **REQUIRED** | Confidence pill (e.g., 85.0%). |
| `decision` | **REQUIRED** | Decision banner (`AUTO_HANDLE` Green / `ESCALATE_TO_HUMAN` Amber). |
| `reason_code` | **REQUIRED** | Escalation trigger code badge. |
| `decision_reason` | **REQUIRED** | Detailed explanation of decision logic. |
| `draft_response` | **REQUIRED** | Support agent output response text. |
| `retrieved_case_ids` | **USEFUL** | Case ID reference chips. |
| `selected_evidence_count` | **USEFUL** | Count indicator for grounded evidence context. |
| `risk_score` | **USEFUL** | Risk level progress bar (relative to 0.35 threshold). |
| `safety_check_passed` | **USEFUL** | Safety check badge (PASS / FAIL). |
| `grounding_status` | **USEFUL** | Grounding verification pill. |
| `retrieval_scores` | **OPTIONAL** | Expandable technical details card. |
| `signals` | **OPTIONAL** | Component risk breakdown card. |
| `generation_metadata` | **OPTIONAL** | Model version & latency metadata. |
| `retrieval_mode` | **SHOULD NOT EXPOSE** | Internal routing detail. |

---

## 7. Planned Product Screens

1. **Landing / Home**:
   - *Purpose*: Overview of Hiver AI Support Agent, architecture highlights, and validated benchmark metrics.
   - *Backend Support*: Static metadata + `GET /api/v1/stats`. Complexity: **LOW**.
2. **Live AI Support Demo**:
   - *Purpose*: Interactive query sandbox with live RAG processing, risk gauge, evidence inspector, and decision banner.
   - *Backend Support*: `POST /api/v1/support/process`. Complexity: **MEDIUM**.
3. **How It Works / Architecture**:
   - *Purpose*: Visual interactive breakdown of the 7 pipeline stages.
   - *Backend Support*: Static interactive diagram. Complexity: **LOW**.
4. **Retrieval & Evidence Explorer**:
   - *Purpose*: Search & inspect 15,316 document RAG corpus, RRF scores, and intent compatibility.
   - *Backend Support*: `POST /api/v1/retrieval/search`. Complexity: **MEDIUM**.
5. **Evaluation & Benchmark Dashboard**:
   - *Purpose*: Interactive charts of intent accuracy, confusion matrix, false auto reduction, and recall curves.
   - *Backend Support*: `GET /api/v1/evaluations`. Complexity: **LOW**.
6. **Safety & Escalation Policy Inspector**:
   - *Purpose*: Risk engine playground showing composite risk calculation against 0.35 threshold.
   - *Backend Support*: `POST /api/v1/decision/evaluate`. Complexity: **LOW**.
7. **About & Challenge Technical Notes**:
   - *Purpose*: Candidate notes, trade-offs, and rejected experiment rationale.
   - *Backend Support*: Static content. Complexity: **LOW**.

---

## 8. Backend Architecture Recommendation

We recommend building a lightweight, high-performance REST API using **FastAPI**:

```
Web Browser (Frontend SPA / Vanilla JS / CSS Design System)
       │  (HTTP / REST API)
       ▼
FastAPI Application (src/hiver_agent/api/app.py)
       │
       ├─► GET  /api/v1/health           (Health check)
       ├─► GET  /api/v1/stats            (Verified benchmark metrics)
       ├─► POST /api/v1/support/process   (Calls CustomerSupportPipeline.process_end_to_end)
       ├─► POST /api/v1/retrieval/search  (Calls HybridRetriever search)
       └─► GET  /api/v1/evaluations       (Serves benchmark JSON files)
       │
       ▼
CustomerSupportPipeline (Singleton in app.state)
```

---

## 9. Dependency Audit

- **Core Runtime**: `pandas`, `numpy`, `scikit-learn`, `pydantic`, `pyyaml`, `sentence-transformers`, `faiss-cpu`, `rank-bm25`.
- **Gemini SDKs**: `google-genai` (v0.1.0) and `google-generativeai` (v0.3.0) are both present; `generator.py` uses `google.generativeai`. Both are preserved.
- **Productization Additions Needed**: `fastapi` and `uvicorn` (will be added to dependencies in the productization phase).

---

## 10. Security Audit

- **API Keys**: `GEMINI_API_KEY` stored securely in `.env` (git-ignored) or environment variables. Zero hardcoded credentials (**PASS**).
- **CORS**: Requires `CORSMiddleware` setup when FastAPI is created (**HIGH PRIORITY for productization**).
- **Input Sanitization**: Query strings passed to Gemini are guarded by system prompts; input length limiting will be added at API layer (**PASS**).

---

## 11. Performance Audit

- **Cold Start**: Initializing embedding provider + FAISS index (15,316 vectors) + BM25 takes ~2.1 seconds. Handled cleanly via FastAPI `lifespan` startup event.
- **Per-Request Latency**: Intent classification + retrieval + decision = ~35 ms; Gemini API generation = ~1.1s. Total per-query response time ~1.15s.

---

## 12. Test/Regression Status

- **Pytest Suite**: **122 passed, 0 failed, 0 skipped** (verified on `tests/`).
- **Regression Safety**: 100% test pass rate preserved.

---

## 13. Technical Debt

- **Critical**: No Web API or Frontend currently exists to showcase the working agent to recruiters.
- **Medium**: Embedding provider initialization output (TensorFlow/PyTorch warnings on load) should be silenced cleanly in API startup.
- **Low**: Legacy CLI evaluation scripts in `scripts/` duplicate some dataset loading helper logic.

---

## 14. What We Should KEEP

1. Core `CustomerSupportPipeline` logic in `src/hiver_agent/pipeline.py`.
2. Validated `CalibratedRulePrototypeHybrid` intent classifier (`intent_classifier_mode="calibrated_prototype_hybrid"`).
3. `Hybrid + Intent` retrieval pipeline (`faiss_index_15k` + `rank_bm25`).
4. `RiskAwarePolicy` escalation decision engine (threshold = 0.35).
5. All 122 existing unit and integration tests.

---

## 15. What We Should IMPROVE

1. Add a FastAPI web server layer (`src/hiver_agent/api/`) exposing structured REST endpoints.
2. Standardize JSON error responses and request validation models.
3. Centralize pipeline startup lifecycle management.

---

## 16. What We Should REBUILD

1. Create a modern, responsive Web Application frontend from scratch to replace raw CLI script outputs.

---

## 17. What We Should NOT Touch

1. Intent taxonomy definition (`AMAZON_INTENTS` 17 classes).
2. Human gold evaluation labels (`amazonhelp_gold_200_annotation.csv` & `amazonhelp_escalation_validation_50.csv`).
3. Model hyper-parameters, thresholds, or system prompts.
4. Frozen Day 7 metrics and evaluation protocols.

---

## 18. Recommended Productization Roadmap

### P0 — Required Before Frontend Demo
- Implement FastAPI REST server (`src/hiver_agent/api/app.py`) with singleton `CustomerSupportPipeline` startup.
- Create `/api/v1/support/process`, `/api/v1/stats`, and `/api/v1/health` endpoints.
- Build modern HTML5/CSS3/Vanilla JS Live Chat Demo interface.

### P1 — High-Value Product Improvements
- Build Retrieval & Evidence Explorer screen.
- Build Interactive Evaluation & Benchmark Dashboard screen.
- Build Safety & Escalation Policy Inspector screen.

### P2 — Polish & Advanced Features
- Add sample query scenario presets (e.g., "Track Order", "Damaged Item", "Billing Dispute").
- Add dark/light mode toggle, micro-animations, and latency timers.

### P3 — Optional Stretch Goals
- Containerize application via Docker / Docker Compose.
