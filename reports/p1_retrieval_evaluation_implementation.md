# P1 Retrieval Explorer + Evaluation Dashboard

## 1. Retrieval Explorer
The Retrieval & Evidence Explorer capability is implemented as a dedicated route view (`/retrieval`) accessible from the top navigation bar. It enables technical recruiters and engineers to interactively query the historical support dataset (15,316 indexed cases) and inspect how evidence is selected for RAG response generation.

Features:
- Search interface with input box and Top-K filter dropdown (5, 10, 20, 50).
- 6 one-click preset queries covering high-volume support intent scenarios (Missing Package, Double Charge, Cancel Order, Damaged Item, Login Issue, Refund Inquiry).
- Clear visual explanation of the RRF hybrid retrieval architecture (Dense FAISS all-MiniLM-L6-v2, Sparse BM25 lexical match, Reciprocal Rank Fusion k=60, and Intent-Aware Compatibility ranking).

## 2. Retrieval API Integration
- Consumes the real backend endpoint `POST /api/v1/retrieval/search` via the typed API client `searchRetrievalCorpus()`.
- Passes `{ query: string, top_k: number }` and receives `{ query, total_retrieved, results, latency_ms }`.
- Zero duplication of retrieval algorithms in the frontend; all searches execute directly against the production FAISS + BM25 index.

## 3. Evidence Display
- Returned cases are displayed as ranked cards showing exact fields returned by the backend: Rank, `case_id`, `customer_problem`, `score` (RRF similarity score), and `intent` taxonomy label.
- Interactive modal inspection view ("Historical Evidence Inspector") allowing deep-dive inspection of customer problem text, taxonomy, similarity score, and prompt integration context without exposing raw file paths or internal credentials.
- Custom loading spinner state ("Searching 15,316 historical support cases...") and empty state handler ("No historical cases matched this query.").

## 4. Evaluation Dashboard
The Evaluation Dashboard is implemented as a dedicated route view (`/evaluation`) designed to demonstrate technical credibility beyond static UI demos. It consumes project metadata from `GET /api/v1/stats` while displaying pre-calculated, verified evaluation metrics across 7 comprehensive sections.

Sections:
1. **Hero**: "Evaluation, Not Just Demos" header highlighting human-labeled validation, controlled retrieval benchmarks, real Gemini evaluation, and risk-aware escalation testing.
2. **Intent Classification**: Headline generalization accuracy (57.0%, Macro F1 53.93%, N=100 untouched test) alongside Full Gold diagnostic accuracy (69.0%, Macro F1 0.6930, N=200). High-confidence error reduction (38 &rarr; 13 errors, 65.8% reduction).
3. **Retrieval Benchmark & Ablation**: Authoritative Phase 9 benchmark (R@1 53.5%, R@3 55.5%, R@5 58.0%, R@10 65.5%, MRR ~0.559) and Day 6/7 reproduction (R@1 52.5%, R@3 56.0%, R@5 61.0%, R@10 65.0%, MRR 0.5570). Visual ablation comparison bar chart showing Dense (28.0%), Intent-Dense (51.5%), BM25 (28.5%), Hybrid RRF (36.0%), and Hybrid+Intent (53.5%).
4. **Generation Quality**: Gemini 1.5 Flash evaluation on N=50 sample achieving 4.72 / 5.00 LLM Judge rating and 100% safety pass rate.
5. **Escalation Policy & Trade-off**: 74% human agreement, 6% false auto-handling (down from 12%), 20% false escalation (up from 14%), exposing the intentional conservative trade-off of RiskAwarePolicy (threshold 0.35).

## 5. Metrics and Methodology
- Strict adherence to official metric policies (no conflating diagnostic 69% with headline 57.0% generalization accuracy).
- 6-step visual evaluation methodology pipeline: Human Gold Labels &rarr; Dev/Test Isolation &rarr; Retrieval Benchmarks &rarr; Gemini LLM Judge &rarr; Escalation Policy Testing &rarr; Automated Regression Suite.

## 6. Failed Experiments
Transparently documents rejected technical approaches:
1. **Hierarchical Intent Classifier**: 26.0% accuracy (REJECTED due to class imbalance collapse).
2. **MS-MARCO Cross-Encoder**: 35.5% R@1 (REJECTED due to domain-specific e-commerce underperformance).
3. **SentenceTransformer Lightweight Reranker**: 47.0% R@1 (REJECTED due to added latency without beating RRF+Intent 53.5%).

## 7. Limitations
Clearly lists 5 operational boundaries:
1. Taxonomy ambiguity as a primary error source.
2. N=100 untouched test set sample variance.
3. 20% false escalation rate cost of conservative risk policy.
4. Proxy-based retrieval relevance judgments.
5. Cold-start embedding index loading latency.

## 8. Responsive Design
- Form layouts, card grids, modal dialogs, and comparison bar charts tested across 1440px (desktop), 1280px (laptop), 768px (tablet), and 390px (mobile).
- Mobile navigation drawer for seamless tab switching on small touch screens.
- Zero horizontal scroll or overflow.

## 9. Accessibility
- Semantic HTML markup (`header`, `main`, `section`, `form`, `h1`-`h3`).
- Keyboard navigable modal buttons (`X` close, backdrop click, `Enter` key triggers).
- WCAG compliant high contrast text over dark glassmorphism styling (`#0B0F19` background).

## 10. Testing
- Frontend TypeScript type check & production Vite build (`npm run build`) passed with 0 errors.
- Python backend test suite (`pytest tests/ -q`) passed with all 130 tests passing.

## 11. Build Validation
- TypeScript compilation (`tsc -b`): PASS (0 errors).
- Vite build (`vite build`): PASS (`dist/assets/index-D8xriyr7.js` 236.93 kB).
- Backend tests (`pytest`): PASS (130/130 tests passed).

## 12. Files Created
- `frontend/src/components/RetrievalExplorer.tsx`
- `frontend/src/components/EvaluationDashboard.tsx`
- `reports/p1_retrieval_evaluation_implementation.md`

## 13. Files Modified
- `frontend/src/components/Navbar.tsx`
- `frontend/src/pages/HomePage.tsx`

## 14. Known Limitations
- The retrieval search UI relies on FastAPI backend running on port 8000. When offline, an explicit error notice ("Retrieval service unavailable") is displayed gracefully.
