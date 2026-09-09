# Hiver AI Customer Support Agent — AmazonHelp

> Given historical customer-support conversations between customers and brands, this project builds an AI support-agent foundation for AmazonHelp that identifies customer intent, retrieves historically relevant support resolutions, and establishes a rigorous evaluation methodology for future grounded response generation and escalation.

---

## 1. Project Overview

This repository implements the engineering foundation for an automated customer support agent submitted for the Hiver SDE / AI Engineering Internship take-home assignment.

### Brand Selection: AmazonHelp
`AmazonHelp` was selected as the primary target domain because it represents the largest brand dataset in the source corpus (**373,464 tweets** across **82,676 reconstructed root conversation trees**). It provides realistic e-commerce support scenarios spanning order tracking, shipping delays, refunds, returns, Prime subscriptions, and account access.

### Role of Historical Support Conversations
Customer support logs contain pairs of customer problem descriptions and resolved agent actions. This historical conversation corpus forms the knowledge base for Retrieval-Augmented Generation (RAG). By indexing verified past resolutions, the support agent can retrieve relevant historical handling logic to ground automated response generation.

### Target Agent Architecture
The complete AI support agent is designed around three core operational capabilities:
1. **Intent Classification**: Categorizing incoming customer inquiries into fine-grained support intent taxonomy classes.
2. **Grounded Historical Retrieval**: Vector search and metadata reranking over historical resolution pairs to fetch candidate solutions.
3. **Automated Handling vs. Human Escalation**: Evaluating confidence, retrieval relevance, and resolution safety to determine whether to auto-resolve or route to human support agents.

> **Current Implementation Status**: Phases 1–4 are fully complete, establishing the data processing pipeline, empirical 17-intent taxonomy, canonical RAG corpus, FAISS vector search, evaluation leakage prevention, and reproducible 15K headline benchmark.  
> **Phase 5 — AmazonHelp intent classification — is the next planned phase and has NOT started.**

---

## 2. Current Project Status

| Phase | Scope | Status | Key Results / Deliverables |
| :--- | :--- | :---: | :--- |
| **Phase 1 — Dataset & Repository Audit** | Inspect raw Twitter & Banking77 data, setup architecture | ✅ Complete | Data schemas validated; audit report produced |
| **Phase 2 — Conversation Reconstruction** | Reconstruct root-to-leaf AmazonHelp thread trees | ✅ Complete | 82,676 conversation trees built (`amazonhelp_threads.parquet`) |
| **Phase 3 — Taxonomy & RAG Corpus** | Define intent taxonomy & historical RAG documents | ✅ Complete | 17 AmazonHelp intents; 81,186 canonical RAG documents |
| **Phase 4 — Retrieval & Evaluation** | Build FAISS vector index, reranker & leakage audit | ✅ Complete | FAISS IndexFlatIP; leakage-safe evaluation; 15K benchmark |
| **Phase 5 — Intent Classifier** | Train production AmazonHelp intent classifier | ⏳ Next | Planned next phase (Not started) |
| **Phase 6 — Response Generation** | Grounded response generation with RAG context | ⏳ Planned | Future phase (Not started) |
| **Phase 7 — Escalation & Integration** | Confidence estimation & auto-handle vs. escalation engine | ⏳ Planned | Future phase (Not started) |

---

## 3. Problem Definition

### Production Support Agent Architecture

```text
Customer Message
       │
       ▼
[ Intent Classification ]  ──► (Identify support category & initial confidence)
       │
       ▼
[ Historical-Case Retrieval ] ──► (Retrieve top-k relevant resolution pairs from FAISS)
       │
       ▼
[ Grounded Response Generation ] (Planned Phase 6 — Draft response grounded in retrieved resolution)
       │
       ▼
[ Safety & Escalation Engine ]  (Planned Phase 7 — Evaluate confidence & resolution risk)
       ├──► High Confidence ──► Auto-Handle Response
       └──► Low Confidence  ──► Escalate to Human Support Agent
```

### Implemented vs. Planned Scope

- **Implemented (Phases 1–4)**:
  - Multi-turn Twitter conversation tree reconstruction algorithm.
  - Empirical 17-intent taxonomy and validated gold evaluation dataset.
  - Canonical 81,186-document historical RAG resolution corpus.
  - SentenceTransformer (`all-MiniLM-L6-v2`) embedding pipeline + FAISS `IndexFlatIP` vector index.
  - Dynamic self-case ID leakage prevention in evaluation harness.
  - Stratified, deterministic 15,000-document headline benchmark (<15 min CPU build).
- **Planned (Phases 5–7)**:
  - Production AmazonHelp intent classifier (Phase 5).
  - Grounded response generation LLM pipeline (Phase 6).
  - Automated resolution vs. human escalation decision engine (Phase 7).

---

## 4. Dataset

### Primary Dataset: Customer Support on Twitter
- **Source**: Kaggle `thoughtvector/customer-support-on-twitter`
- **Volume**: 2,811,774 tweets (~2.8 million records), 516.51 MB raw CSV (`data/raw/customer_support_on_twitter.csv`).
- **Core Schema**: `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id`.
- **Git Storage Policy**: The 516 MB raw CSV is **not committed to Git** to ensure repository lightness and clean reproducibility. Local directory instructions are provided in setup.

### AmazonHelp Subset Statistics
- **Extracted Tweets**: 373,464 tweets associated with `AmazonHelp`.
- **Reconstructed Root Conversation Trees**: 82,676 complete multi-turn thread trees.
- **Canonical RAG Corpus**: 81,186 quality-filtered historical support documents.
- **Support Intent Categories**: 17 empirically validated intent classes.

---

## 5. AmazonHelp Intent Taxonomy

The 17-intent taxonomy was derived empirically from `AmazonHelp` support threads in Phase 3 and validated against a dedicated 800-example evaluation sample (`data/golden/taxonomy_validation_sample.parquet`):

1. `order_tracking_delivery_status` — Tracking package location and delivery progress.
2. `late_delivery_shipping_delay` — Packages arriving past estimated delivery date.
3. `missing_package_delivered_not_received` — Status marked delivered but package not received.
4. `refund_status_request` — Checking status or requesting money back for orders.
5. `return_exchange_inquiry` — Process, policies, and labels for returning items.
6. `order_cancellation_request` — Requests to cancel orders prior to or during dispatch.
7. `damaged_defective_item` — Physical damage or broken products received.
8. `wrong_item_received` — Incorrect item or missing components delivered.
9. `payment_billing_issue` — Double charges, payment failures, gift cards, invoices.
10. `account_access_login` — Password reset, locked accounts, OTP issues.
11. `prime_membership_subscription` — Prime auto-renewal, membership fees, benefits.
12. `digital_content_kindle_video` — E-books, Prime Video, digital downloads, device syncing.
13. `third_party_seller_issue` — Disputes with non-Amazon marketplace sellers.
14. `promotions_discounts_coupons` — Promotional codes, lighting deals, price matching.
15. `address_delivery_instructions` — Updating delivery address or special instructions.
16. `customer_service_complaint` — Complaints regarding agent behavior or service quality.
17. `general_inquiry_other` — General inquiries not covered by specific categories.

---

## 6. RAG Corpus

- **Canonical Corpus**: **81,186 documents** stored in `data/processed/amazonhelp_rag_documents.jsonl`.
- **Structure**: Each document pairs the initial customer problem statement (`customer_text`) with the resolved support interaction and final agent resolution (`resolution_text`).
- **Quality Filters**: Requires customer inbound text, agent response, minimum customer problem length $\ge 15$ characters, and conversation duration $\le 7$ days. Actionless, incomplete, or uninformative conversations (e.g. orphan outbound tweets) were strictly filtered out.
- **Data Provenance**: Built strictly from historical Twitter customer service conversations. No synthetic or fabricated resolution texts were created.

---

## 7. Retrieval System Architecture

### Pipeline Components
```text
Customer Query
      │
      ▼
[ SentenceTransformer Provider ] ──► (all-MiniLM-L6-v2, 384-dim, L2 Normalized)
      │
      ▼
[ FAISS Vector Store ] ──────────► (IndexFlatIP — Inner Product / Cosine Similarity)
      │
      ▼
[ Candidate Over-Retrieval ] ────► (Top-25 candidate historical documents)
      │
      ▼
[ Intent-Aware Reranker ] ────────► (Boost score +0.15 if candidate intent matches predicted query intent)
      │
      ▼
[ Leakage-Safe Evaluator ] ──────► (Filter out self-case_id dynamically during evaluation)
```

### Technical Configuration
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Dimension**: 384
- **Normalization**: L2 normalized vectors ($\|v\|_2 = 1.0$)
- **Max Sequence Length**: 128 tokens
- **Batch Size**: 128
- **PyTorch CPU Threads**: 4
- **Vector Store**: FAISS `IndexFlatIP`
- **Index Features**: Resumable chunk-checkpointing (5,000 docs/chunk), bounded processing (`--max-docs N`), deterministic ordering, embedding-to-document alignment verification.

---

## 8. Reproducible 15K Headline Benchmark

To meet the take-home assignment's **< 15-minute execution time requirement** for headline reproducibility on CPU environments, a 15,000-document reproducible headline corpus and index configuration was created:

- **Canonical Corpus**: 81,186 documents (`amazonhelp_rag_documents.jsonl`).
- **Headline Reproducible Corpus**: 15,000 documents (`amazonhelp_rag_documents_15k.jsonl`).
- **Sampling Methodology**:
  - Deterministic seed (`seed = 42`).
  - Stratified across all 17 intent categories (proportional allocation with a minimum floor of $N \ge 10$ to preserve rare intents like `wrong_item_received`).
  - 0 exact duplicate customer texts.
  - Deterministic sorting by `case_id`.
- **Measured Build Runtime**: **794.25 seconds (13.23 minutes)** on CPU (satisfying the <15-minute execution limit).
- *Note*: Full 81.1K corpus encoding requires ~64 minutes on CPU and is maintained as the canonical background corpus.

---

## 9. Retrieval Evaluation Results

Evaluated on **800 test queries** (`data/golden/taxonomy_validation_sample.parquet`) against the 15,000-document headline index with dynamic self-case ID filtering (`candidate.case_id != target.case_id`):

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Latency |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pure Semantic Retrieval (FAISS IndexFlatIP)** | 0.2750 | 0.4500 | 0.5200 | 0.6150 | 0.3905 | 60.61 ms |
| **Intent-Aware Retrieval (Metadata Reranked)** | **0.6300** | **0.6850** | **0.7050** | **0.7400** | **0.6669** | 5.03 ms |

> **Relevance Measure**: Intent-based proxy relevance (retrieved historical case matches query intent category with valid resolution action).  
> **Self-Case Leakage**: Exactly 0. Dynamic self-case ID filtering prevents query documents from retrieving themselves.

---

## 10. Retrieval Evaluation Caveats

> [!IMPORTANT]
> The following methodological caveats govern the interpretation of evaluation metrics:

1. **Subsample Performance Difference**: Semantic Recall@10 on the 15K headline corpus (61.50%) is lower than on the full 81.1K canonical corpus (86.50%) because reducing the candidate index by 81.5% removes historical resolution candidates. 15K metrics serve as a fast, reproducible headline benchmark and **do not represent full-corpus production capacity**.
2. **Intent-Based Proxy Relevance**: Relevance is evaluated using intent class matching and valid resolution presence, not manual pairwise human relevance judgments across all 81.1K documents.
3. **Predicted Query Intent Source**: Intent-aware retrieval uses heuristic predicted intent from `RuleBasedIntentClassifier` at query time. It is **not** oracle-assisted with gold labels.
4. **Latency Measurement Comparability**: Reported latencies (60.61 ms vs. 5.03 ms) measure total evaluation loop execution time per query. The initial pass (`semantic_retriever`) incurred PyTorch model cold-start and framework warm-up overhead. Isolated production latency was **not measured in this phase**.
5. **Near-Duplicate Overlap**: While exact duplicate texts were removed and self-case IDs dynamically excluded, near-duplicate customer tweets across threads remain a known limitation to be handled in future evaluation refinements.

---

## 11. Banking77 Auxiliary Benchmark

To validate generic intent classification methods on a standardized benchmark, an auxiliary experiment was conducted on Banking77:

- **Dataset**: `data/external/banking77/banking77_test.csv` (3,076 test examples across 77 intent classes).
- **Results**: **Accuracy = 85.01%** | **Macro F1 = 0.8485** | **Weighted F1 = 0.8486**.
- **Scope Note**: **Banking77 serves as auxiliary technical evidence only and is NOT used as AmazonHelp ground truth or performance.**

---

## 12. Engineering & Reproducibility

- **Environment**: Python $\ge 3.11$, PyTorch (CPU mode), `faiss-cpu==1.15.0`, `sentence-transformers==4.0.1`.
- **Determinism**: Fixed random seeds (`seed = 42`) across corpus sampling, stratified partitioning, and FAISS indexing.
- **Memory Safety**: Resumable 5,000-document chunk checkpointing ensures peak RSS memory remains under 400 MB.
- **Automated Tests**: `pytest tests/ -v` passes **23/23 unit tests**.

---

## 13. Repository Structure

```text
hiver-support-agent/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── configs/
├── data/
│   ├── external/             # Banking77 benchmark datasets
│   ├── golden/               # Taxonomy validation sample (Parquet & CSV)
│   ├── interim/              # Intermediary processing files
│   ├── processed/            # Reconstructed threads, RAG corpus, 15K JSONL & FAISS indices
│   ├── raw/                  # Protected raw datasets (customer_support_on_twitter.csv)
│   └── samples/              # Stratified review samples (Parquet & CSV)
├── reports/                  # Completion reports, closeout docs, taxonomy & audit reports
├── scripts/                  # CLI scripts for reconstruction, RAG build, indexing & eval
├── src/
│   └── hiver_agent/
│       ├── analysis/         # Intent discovery & pattern matching
│       ├── classification/   # Taxonomy, classifiers & evaluation harness
│       ├── config.py         # Global configuration settings
│       ├── data/             # Schemas, thread builder, filters & RAG corpus builder
│       ├── decision/         # Escalation decision engine interfaces (Phase 7)
│       ├── evaluation/       # Benchmark evaluation helpers
│       ├── generation/       # Grounded response generator interfaces (Phase 6)
│       └── retrieval/        # Embedding provider, FAISS vector store & retrievers
└── tests/                    # Pytest test suite (23 passing tests)
```

---

## 14. Reproduction Commands

### Environment Setup
```bash
# Clone repository and install editable package
git clone <repo_url>
cd hiver-support-agent
pip install -e .
```

### 1. Build Reconstructed AmazonHelp Threads & Canonical RAG Corpus (Phase 2 & 3)
```bash
python scripts/build_amazonhelp_threads.py
python scripts/build_rag_corpus.py
```

### 2. Generate Deterministic 15K Stratified Headline Corpus
```bash
python scripts/create_15k_retrieval_corpus.py
```
*Output*: `data/processed/amazonhelp_rag_documents_15k.jsonl`

### 3. Build Reproducible FAISS Retrieval Index (< 15 min execution)
```bash
python scripts/build_retrieval_index.py --input-jsonl data/processed/amazonhelp_rag_documents_15k.jsonl --out-dir data/processed/retrieval_index_15k --batch-size 128 --threads 4
```
*Output*: `data/processed/retrieval_index_15k/faiss_index.bin` (~23.0 MB)

### 4. Run Leakage-Safe Retrieval Evaluation
```bash
python scripts/evaluate_retrieval.py
```
*Output*: `reports/retrieval_evaluation.md`, `reports/retrieval_failure_analysis.md`

### 5. Run Full Automated Test Suite
```bash
pytest tests/ -v
```
*Output*: `23 passed in ~23s`

---

## 15. Evaluation Philosophy

This project adheres to rigorous AI engineering standards:
1. **Leakage Prevention First**: Strict dynamic self-case ID filtering guarantees no document self-retrieves during evaluation.
2. **Transparent Metric Labeling**: Clear distinction between proxy relevance metrics and human pairwise ground truth.
3. **Reproducibility Under Constraints**: Delivering an end-to-end reproducible headline benchmark that runs under strict 15-minute CPU constraints.
4. **Honest Technical Limitations**: Documenting framework warm-ups, subsample density effects, and candidate noise explicitly.

---

## 16. Current Limitations

- **Classifier Scope**: Intent classification in Phase 4 uses baseline rule-based and prototype embedding classifiers; production supervised classifier training is scheduled for Phase 5.
- **Relevance Annotations**: Evaluation relies on intent-based proxy matching rather than manual human pairwise relevance scores across all 81.1K documents.
- **Subsample Corpus**: The headline index uses 15,000 documents for fast reproducibility; full 81,186-document index requires ~64 minutes CPU encoding time.
- **Unimplemented Future Modules**: Grounded response generation (Phase 6), automated escalation decision engine (Phase 7), and Streamlit UI are not yet built.

---

## 17. Project Roadmap

- [x] **Phase 1**: Dataset & repository audit, schema validation, data quality report.
- [x] **Phase 2**: Multi-turn conversation reconstruction, AmazonHelp corpus creation.
- [x] **Phase 3**: 17-intent taxonomy formulation, validation golden set, 81.1K RAG corpus.
- [x] **Phase 4**: FAISS vector retrieval, intent-aware reranking, leakage audit, 15K headline benchmark.
- [ ] **Phase 5 (Next)**: Production AmazonHelp 17-intent classifier training & evaluation.
- [ ] **Phase 6**: Grounded response generation with RAG context & citation verification.
- [ ] **Phase 7**: Auto-handle vs. human escalation decision engine & safety gates.
- [ ] **Phase 8**: End-to-end agent evaluation, failure analysis, UI, and final submission report.

---

## 18. Hiver Assignment Requirements Mapping

| Hiver Take-Home Requirement | Implementation Status | Relevant Location / Artifact |
| :--- | :---: | :--- |
| **Multi-turn conversation handling** | ✅ Complete | `src/hiver_agent/data/thread_builder.py` |
| **Intent classification taxonomy** | ✅ Complete | `src/hiver_agent/classification/taxonomy.py` (17 intents) |
| **Historical resolution retrieval (RAG)** | ✅ Complete | `src/hiver_agent/retrieval/` & FAISS `IndexFlatIP` |
| **Leakage-safe evaluation harness** | ✅ Complete | `scripts/evaluate_retrieval.py` |
| **Reproducible benchmark (< 15 mins)** | ✅ Complete | `scripts/create_15k_retrieval_corpus.py` (13.23 mins) |
| **Production Intent Classifier** | ⏳ Next | Scheduled for Phase 5 |
| **Response Drafting (Grounded LLM)** | ⏳ Planned | Scheduled for Phase 6 |
| **Escalation Engine (Auto vs Human)** | ⏳ Planned | Scheduled for Phase 7 |
