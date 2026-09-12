# P3 Dockerization Report

## 1. Summary
The P3 Dockerization phase containerizes the complete Hiver AI Customer Support Agent application for one-command local recruiter deployment (`docker compose up --build`).

The build packages both the FastAPI backend and the React Vite frontend into isolated, deterministic containers while preserving the ML pipeline, vector indices, decision policies, and benchmark metrics without modification.

Key highlights:
- 144 Python unit & API hardening tests passing (`pytest tests/ -q`), 0 failing, 0 skipped.
- Frontend production build (`npm run build`) passing with 0 TypeScript errors.
- One-command docker compose orchestration for local recruiter execution.
- Runtime injection of `GEMINI_API_KEY` with zero secrets baked into image layers or frontend bundles.
- Explicit frontend rendering of LLM generation provider mode (Real Gemini 1.5 Flash vs Deterministic/Mock Fallback).

## 2. Architecture
```
Browser (http://localhost:3000)
    │
    ▼
Frontend Container (Nginx:alpine on port 3000)
    │
    ▼ REST API Requests (http://localhost:8000/api/v1)
Backend Container (FastAPI + Uvicorn on port 8000)
    │
    ├── Singleton CustomerSupportPipeline
    │     ├── CalibratedRulePrototypeHybrid Classifier
    │     ├── FAISS 15k Vector Store (IndexFlatIP)
    │     ├── BM25Okapi Lexical Reranker
    │     └── Evidence Selector
    │
    └── Response Generation Layer
          ├── Real Gemini 1.5 Flash (if GEMINI_API_KEY supplied)
          └── Deterministic / Mock Fallback Generator (if offline / missing key)
```

## 3. Docker Files
- `Dockerfile.backend`: `python:3.11-slim` base image, non-root working dir `/app`, installs requirements, copies minimum required runtime assets (`src/`, `scripts/`, `configs/`, `data/golden/`, `data/processed/faiss_index_15k/`), exposes port `8000`, runs Uvicorn.
- `frontend/Dockerfile`: Multi-stage build (`node:18-alpine` builder stage, `nginx:alpine` runner stage with custom `nginx.conf` listening on port `3000`).
- `docker-compose.yml`: Defines `backend` (port `8000:8000`, healthcheck on `/api/v1/ready`) and `frontend` (port `3000:3000`, depends on `backend` health).
- `.dockerignore`: Excludes `.git`, `.env`, `node_modules`, `__pycache__`, `.pytest_cache`, raw datasets (`data/raw/`), temporary files (`scratch/`), and logs while preserving `data/processed/faiss_index_15k/`.

## 4. Runtime Assets
- **FAISS Vector Store**: `data/processed/faiss_index_15k/` (contains `faiss_index.bin`, `document_map.json`, `index_meta.json`).
- **Classifier Artifacts**: `data/golden/amazonhelp_gold_200_annotation.csv` (used for centroid prototype fitting during startup).
- **Configuration**: `configs/pipeline_config.yaml` and default parameters.
- **Excluded Assets**: Unnecessary 500+ MB raw Twitter CSV files (`data/raw/`) and interim scratch files are excluded from container build context.

## 5. Secrets
- **GEMINI_API_KEY runtime injection**: Passed into container environment via `docker-compose.yml` (`GEMINI_API_KEY=${GEMINI_API_KEY:-}`).
- **Frontend exposure**: Zero API keys or backend secrets included in the compiled React SPA bundle.
- **Docker layer exposure**: No secrets copied or hardcoded in Dockerfiles.
- **Logs**: Operational logs log request IDs and metadata without printing raw secrets.

## 6. Health & Readiness
- `/api/v1/health`: Liveness probe verifying application process health.
- `/api/v1/ready`: Readiness probe verifying FAISS index loading (`15,316` documents) and classifier availability without making external LLM calls.
- **Compose health check**: Configured to poll `http://localhost:8000/api/v1/ready` every 10 seconds.

## 7. Real E2E Demo
- **Real Gemini**: When a valid `GEMINI_API_KEY` is provided, responses are generated using Gemini 1.5 Flash (`temperature=0.2, max_tokens=500`).
- **Intent**: Classified via `CalibratedRulePrototypeHybrid` with confidence score.
- **Retrieval**: Ranked evidence cases retrieved from FAISS + BM25 index.
- **Generation**: Grounded synthesis with evidence prompt constraints.
- **Grounding & Safety**: Validated by `GroundingSafetyChecker`.
- **Decision**: `RiskAwarePolicy` outputs `AUTO_HANDLE` or `ESCALATE_TO_HUMAN` based on composite risk score vs threshold `0.35`.
- **Request ID**: Generated or preserved `X-Request-ID` attached to all response payloads.

## 8. Gemini Failure Test
- **Missing/invalid key**: Automatically falls back to deterministic/mock response generator without process termination.
- **Fallback**: Returns structured evidence-grounded template text.
- **Provider visibility**: `generation_metadata` explicitly identifies whether generation mode is `real` or `mock/template`.
- **Crash**: Zero crashes or uncaught exceptions.
- **Secret leakage**: Secrets are concealed from logs and error responses.

## 9. Docker Build
- `Dockerfile.backend`: Builds clean Python 3.11 container.
- `frontend/Dockerfile`: Multi-stage build compiles React SPA with Vite and packages static assets into Nginx Alpine.
- `docker-compose.yml`: Syntactically valid Compose service definition.

## 10. Clean Machine Test
- **Python required outside Docker**: NO.
- **Node/npm required outside Docker**: NO.
- **Other dependencies**: Only Docker Desktop / Docker Engine and Git.

## 11. README
- Added concise `# Docker Quickstart` section with clone, `.env` setup, `docker compose up --build`, and service URL listings.

## 12. Security Audit
- `.env` ignored by git and docker context.
- No secrets baked into image layers.
- CORS restricted to allowed origins (`http://localhost:3000`, `http://127.0.0.1:3000`).
- Exception tracebacks hidden from API clients.

## 13. Regression
- **Python tests**: 144 passed, 0 failed, 0 skipped.
- **Frontend build**: Production build succeeds in 4.84s.
- **TypeScript**: 0 errors.
- **Compose configuration**: Valid.

## 14. ML/Evaluation Freeze Verification
"ML methodology unchanged."
"Evaluation methodology unchanged."
"Benchmark numbers unchanged."
"Gold data unchanged."

## 15. Remaining Limitations
- Requires local CPU resources to load FAISS index and MiniLM embedding model (~1.5 GB RAM).

## 16. Final Recommendation
The application is fully containerized, tested, and ready for **recruiter local demo** and final submission.
