# P1.5 API + Retrieval Correctness Audit

## 1. Executive Summary
This audit provides a comprehensive technical correctness verification of the REST API, retrieval pipeline implementation, frontend Retrieval Explorer UI, and evaluation metric presentation for the Hiver AI Customer Support Agent application.

The application backend, ML classifiers, vector indices, and frontend components were thoroughly inspected against authoritative project benchmarks, code implementations, and Pydantic schemas without making any structural code modifications.

Key findings:
- All 130 Python backend unit tests and Vite production builds (`npm run build`) pass cleanly.
- Real REST API endpoints (`/api/v1/support/process`, `/api/v1/retrieval/search`, `/api/v1/stats`, `/api/v1/health`) are fully integrated with zero fallback mock data in active UI routes.
- A critical score terminology mismatch was identified: the frontend modal labels returned scores as `"RRF Similarity Score"`, whereas the endpoint `/api/v1/retrieval/search` executes `base_retriever` (`FAISSRetriever`), returning FAISS Cosine Similarity scores.

## 2. Retrieval Score Semantics
1. **Dense Retrieval** (`FAISSRetriever` / `FAISSVectorStore`):
   - Uses `faiss.IndexFlatIP` over 384-dimensional L2-normalized embeddings (`all-MiniLM-L6-v2`).
   - Produces inner product values equivalent to **Cosine Similarity** ($\text{Range: } [-1.0, 1.0]$).
2. **BM25 Retrieval** (`BM25Retriever`):
   - Uses `rank_bm25.BM25Okapi` over tokenized document text.
   - Produces unnormalized BM25 scores based on term frequency and IDF ($\text{Range: } [0, +\infty)$).
3. **RRF Fusion** (`HybridRetriever`):
   - Calculates Reciprocal Rank Fusion: $RRF(d) = \sum_{m} \frac{w_m}{k + rank_m(d)}$ with $k=60$.
   - Max theoretical top-1 RRF score is $\frac{1}{61} + \frac{1}{61} \approx 0.032787$.
4. **Intent Compatibility Boost** (`IntentAwareRetriever`):
   - Adds $+0.15$ directly to `cand.score` if the candidate document intent matches predicted query intent with confidence $\ge 0.60$.
5. **API Score Returned**:
   - `/api/v1/retrieval/search` calls `app.state.pipeline.base_retriever.retrieve(query, top_k)`, which is initialized to `FAISSRetriever`.
   - The returned `score` is FAISS Cosine Similarity (e.g. `0.7523`).
6. **Frontend Score Displayed**:
   - `RetrievalExplorer.tsx` displays `Score: {res.score.toFixed(4)}` on cards and `"RRF Similarity Score: {res.score.toFixed(4)}"` in the detail inspector modal.

## 3. RRF Terminology Audit
- **Status**: **INCORRECT / TERMINOLOGY MISMATCH**
- **Finding**: The UI modal explicitly labels the score as `"RRF Similarity Score"`, while the backend endpoint `/api/v1/retrieval/search` returns pure FAISS Cosine Similarity. In addition, RRF scores are rank reciprocal values bounded by $\approx 0.0328$, whereas the UI displays values like `0.7523`.
- **Root Cause**: In `CustomerSupportPipeline.__init__`, `base_retriever` defaults to `FAISSRetriever`. In `app.py`, `execute_retrieval_search` calls `pipeline.base_retriever.retrieve(...)`.

## 4. Retrieval Document Terminology
- **Corpus Size**: 15,316 indexed historical support cases (`data/processed/faiss_index_15k`).
- **Raw Conversation Turns**: 81,186 turns in the raw Twitter AmazonHelp dataset.
- **Recommended User-Facing Term**: `"Historical Support Cases"` or `"RAG Resolution Cases"`.
- **Status**: **PASS**. Calling the 15,316 entries "historical support cases" accurately reflects resolved Twitter customer support threads containing customer problem, support interaction, and final resolution.

## 5. Retrieval Explorer Field Audit

| Frontend Label | Actual API Field | Actual Meaning | Audit Status | Notes / Discrepancy |
| :--- | :--- | :--- | :--- | :--- |
| **Rank** (`#{idx + 1}`) | Array index | 1-indexed rank order in top-k array | **PASS** | Accurate ordering |
| **Case ID** | `case_id` | Unique historical case identifier string | **PASS** | Matches API `case_id` |
| **Intent Taxonomy** | `intent` | Historical case intent label | **PASS** | Matches API `intent` |
| **Customer Problem** | `customer_problem` | Customer problem text | **PASS** | Matches API `customer_problem` |
| **Score** | `score` | FAISS IndexFlatIP Cosine Similarity | **WARNING** | Main card displays "Score" without specifying metric |
| **RRF Similarity Score** | `score` | FAISS IndexFlatIP Cosine Similarity | **INCORRECT** | Modal claims "RRF Similarity Score", but API returns FAISS Cosine Similarity |
| **Search Latency** | `latency_ms` | API execution time in ms | **PASS** | Real measured execution time |
| **Found X matching cases** | `total_retrieved` | Count of returned items | **WARNING** | Reflects top_k count returned, not total corpus matches |
| **Hybrid Retrieval (RRF k=60)** | Hardcoded UI banner | Architecture description | **WARNING** | Endpoint uses `base_retriever` (`FAISSRetriever`), not `HybridRetriever` |

## 6. Live Demo Field Audit

| Frontend Label | Actual Backend Field | Actual Meaning | Audit Status | Notes / Discrepancy |
| :--- | :--- | :--- | :--- | :--- |
| **Customer Message** | `customer_message` | Customer inquiry text | **PASS** | Accurate |
| **Predicted Intent** | `predicted_intent` | Predicted intent taxonomy label | **PASS** | Accurate |
| **Intent Confidence** | `intent_confidence` | Classifier confidence (0.0 to 1.0) | **PASS** | Calibrated hybrid confidence |
| **Latency** | `latency_ms` | Total API processing latency in ms | **PASS** | Real measured request time |
| **Retrieved Case IDs** | `retrieved_case_ids` | List of evidence case IDs | **PASS** | Accurate |
| **Retrieved Scores** | `retrieved_scores` | List of evidence scores | **WARNING** | UI formats with fallback `RRF`; actually boosted FAISS scores |
| **Grounding Status** | `grounding_status` | Status string ("GROUNDED", etc.) | **PASS** | Accurate |
| **Safety Checker** | `safety_check_passed` | Boolean from GroundingSafetyChecker | **WARNING** | UI shows PASS/FAIL; rule-based safety check, not absolute proof |
| **Escalation Decision** | `decision` | `AUTO_HANDLE` or `ESCALATE_TO_HUMAN` | **PASS** | Accurate |
| **Reason Code** | `reason_code` | Escalation rule code | **PASS** | Accurate |
| **Decision Reason** | `decision_reason` | Natural language explanation | **PASS** | Accurate |
| **Composite Risk Score**| `risk_score` | Risk score (0.0 to 1.0) | **PASS** | Accurately evaluated against 0.35 threshold |
| **Generated Response** | `draft_response` | Gemini 1.5 Flash text output | **PASS** | Real LLM generation |

## 7. Stats Endpoint Audit
- `GET /api/v1/stats` returns validated metrics matching all authoritative reports:
  - `intent_accuracy_untouched_test`: 0.5700 (Untouched Gold Test, N=100)
  - `intent_accuracy_full_gold_diagnostic`: 0.6900 (Full Gold Dataset, N=200)
  - `retrieval_recall_at_1`: 0.5250 (Human Gold RAG Benchmark, N=200)
  - `gemini_llm_judge_score`: 4.72 / 5.0 (N=50)
  - `escalation_false_auto_handle_rate`: 0.0600 (N=50)
- **Status**: **PASS**. All stat fields and notes are technically accurate.

## 8. Evaluation Dashboard Audit
- **57.0% Accuracy (N=100)**: Accurately labeled as Headline Generalization Benchmark with untouched test set isolation.
- **69.0% Accuracy (N=200)**: Accurately labeled as Full-Gold Diagnostic with clear disclaimer ("Full-gold diagnostic; not the headline generalization result.").
- **High-Confidence Errors**: Accurately shows 38 &rarr; 13 errors (65.8% reduction) with copy explaining it reflects fewer confidently incorrect predictions.
- **Phase 9 vs Day 6/7 Retrieval**: Phase 9 benchmark (R@1 53.5%) and Day 6/7 reproduction (R@1 52.5%) are kept distinct and clearly labeled.
- **Status**: **PASS**. Zero metric conflation or false generalization claims.

## 9. Rejected Experiment Audit
- Hierarchical Intent Classifier: 26.0% accuracy (REJECTED due to parent classifier collapse).
- MS-MARCO Cross-Encoder: 35.5% R@1 (REJECTED due to domain gap).
- Lightweight SentenceTransformer Reranker: 47.0% R@1 (REJECTED due to added latency).
- **Status**: **PASS**. All metrics and failure rationales match authoritative experimental logs.

## 10. Safety Claim Audit
- Gemini safety score 100% (N=50) and GroundingSafetyChecker status are accompanied by explicit disclaimers: `"Responses were evaluated using a real Gemini-based judge on a deterministic 50-case sample."`
- **Status**: **PASS**. UI does not claim "guaranteed hallucination-free production" or "100% absolute safety".

## 11. Escalation Claim Audit
- Exposes complete policy trade-off: False auto-handling decreased (12% &rarr; 6%), cost: False escalation increased (14% &rarr; 20%), Agreement: 74%, Auto Rate: 82% &rarr; 70%.
- **Status**: **PASS**. UI clearly communicates that the decision policy became more conservative rather than falsely claiming universal metric improvement.

## 12. API Error Contract
- FastAPI exception handlers return structured `APIErrorDetail(code, message, detail)`.
- HTTP 500 errors conceal internal stack traces from clients (`detail=None`), logging full tracebacks to server logs.
- **Status**: **PASS**. Secure, leak-free error handling.

## 13. CORS Audit
- Middleware reads `ALLOWED_ORIGINS` environment variable and defaults to `localhost:3000`, `localhost:5173`, `127.0.0.1:3000`, `127.0.0.1:5173`.
- **Status**: **PASS**. Standard secure development configuration.

## 14. Input Validation
- `SupportProcessRequest`: Pydantic validates `min_length=1`, `max_length=2000`, and rejects whitespace-only messages.
- `RetrievalSearchRequest`: Pydantic validates `query` non-empty (max 1000) and `top_k` between 1 and 50.
- **Status**: **PASS**. Strict schema-level input validation.

## 15. Performance Claims
- Startup pipeline loading: ~2.1s cold start (loading MiniLM model & FAISS index into memory).
- API request latency: Real-time execution measurements returned via `latency_ms` field.
- **Status**: **PASS**. Performance numbers reflect actual backend execution timings.

## 16. API Contract Consistency
- All TypeScript interface definitions in `frontend/src/types/api.ts` match FastAPI Pydantic schemas in `src/hiver_agent/api/schemas.py`.
- **Status**: **PASS**.

## 17. Mock Data Audit
- `LiveDemo.tsx` and `RetrievalExplorer.tsx` consume 100% real FastAPI backend responses. No fallback mock data exists in live UI flows.
- **Status**: **PASS**.

## 18. Regression Results
- **Python Backend Unit Tests**: `130 passed in 12.33s` (`pytest tests/ -q`).
- **Frontend Production Build**: `tsc -b && vite build` PASSED with 0 errors.

## 19. Required Corrections

### HIGH
1. **Retrieval Explorer Score Label Correction**: Update `RetrievalExplorer.tsx` detail modal label from `"RRF Similarity Score"` to `"FAISS Vector Similarity (Cosine Score)"` to match the score returned by `/api/v1/retrieval/search`.

### MEDIUM
2. **Retrieval Pipeline Wiring for RAG Search Endpoint**: Update `app.py` `execute_retrieval_search` to utilize `pipeline.intent_aware_retriever` (or `HybridRetriever`) if the endpoint is intended to expose the full hybrid RRF search engine rather than pure dense FAISS search.

### LOW
3. **Retrieval Explorer Total Results Count Copy**: Clarify copy from `"Found X matching historical cases"` to `"Displaying Top X retrieved cases"` to avoid implying $X$ is the total count of matches across the 15,316 document corpus.
