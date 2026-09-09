# Phase 4 Closeout Report — Stratified 15K Reproducible Headline Corpus & Retrieval Index

> **Execution Timestamp**: 2026-09-09T21:34:00+05:30  
> **Phase**: Phase 4 Closeout Task  
> **Status**: **COMPLETE & VERIFIED (Ready for User Review)**  

---

## 1. Purpose

The objective of this Phase 4 Closeout task is to establish a deterministic, leakage-safe, stratified 15,000-document `AmazonHelp` retrieval corpus and FAISS vector index operating strictly under the assignment's **< 15-minute execution time constraint** (~13.23 minutes actual runtime). 

This reproducible headline pipeline ensures fast, scalable, deterministic verification of intent-aware support document retrieval without altering raw Twitter data or executing un-bounded full-corpus encoding jobs.

---

## 2. Canonical vs Headline Corpus

- **Canonical Source Corpus**: `data/processed/amazonhelp_rag_documents.jsonl` containing **81,186 documents**. This remains the official canonical production corpus.
- **Headline Reproducible Corpus**: `data/processed/amazonhelp_rag_documents_15k.jsonl` containing exactly **15,000 documents**. This serves as the reproducible headline evaluation corpus.

---

## 3. Sampling Methodology

The 15,000-document subset was generated using `scripts/create_15k_retrieval_corpus.py` following a deterministic, stratified sampling procedure:

1. **Exact Text Deduplication**: Analyzed all 81,186 documents and removed exact string duplicates across customer document texts.
2. **Intent Grouping**: Grouped documents into the 17 validated `AmazonHelp` intent taxonomy categories.
3. **Proportional Stratification**: Calculated sample target per intent proportionally based on its representation in the canonical corpus, enforcing a minimum representation floor ($N \ge 10$) to guarantee rare intent coverage (e.g., `wrong_item_received`).
4. **Fixed Random Seed**: Utilized `seed = 42` (with intent-salted sub-seeds) to sample documents deterministically from `case_id`-sorted intent candidate groups.
5. **Deterministic Output Ordering**: Sorted all 15,000 sampled documents strictly by `case_id` prior to JSONL serialization.
6. **Metadata Preservation**: Preserved all essential document fields (`case_id`, `tweet_id`, `created_at`, `intent`, `num_turns`, `customer_text`, `resolution_text`, `document_text`, and `metadata`) without altering raw document text.

---

## 4. Intent Distribution

The exact intent breakdown across the 15,000-document headline corpus is detailed below:

| Intent Category | Canonical Count (81.1K) | 15K Sampled Count | Proportion (%) |
| :--- | :---: | :---: | :---: |
| `general_inquiry_other` | 52,987 | 9,788 | 65.25% |
| `prime_membership_subscription` | 7,085 | 1,309 | 8.73% |
| `digital_content_kindle_video` | 2,698 | 498 | 3.32% |
| `account_access_login` | 2,514 | 464 | 3.09% |
| `late_delivery_shipping_delay` | 2,298 | 424 | 2.83% |
| `refund_status_request` | 2,258 | 417 | 2.78% |
| `payment_billing_issue` | 1,787 | 330 | 2.20% |
| `return_exchange_inquiry` | 1,555 | 287 | 1.91% |
| `address_delivery_instructions` | 1,495 | 276 | 1.84% |
| `order_tracking_delivery_status` | 1,437 | 265 | 1.77% |
| `order_cancellation_request` | 1,414 | 261 | 1.74% |
| `customer_service_complaint` | 1,111 | 205 | 1.37% |
| `damaged_defective_item` | 1,105 | 204 | 1.36% |
| `third_party_seller_issue` | 520 | 96 | 0.64% |
| `missing_package_delivered_not_received` | 504 | 93 | 0.62% |
| `promotions_discounts_coupons` | 271 | 50 | 0.33% |
| `wrong_item_received` | 147 | 33 | 0.22% |
| **Total** | **81,186** | **15,000** | **100.00%** |

---

## 5. Deduplication Methodology

- Exact customer problem text duplicates were scanned prior to sampling.
- No duplicate customer problem text exists within the 15,000-document headline corpus (0 exact duplicate documents).

---

## 6. Leakage Handling

- **Evaluation Exclusions**: To prevent evaluation contamination, `scripts/evaluate_retrieval.py` dynamically filters candidates by `case_id` (`candidate.case_id != target.case_id`) during query execution.
- **Corpus Inspection**: Evaluation queries are checked against candidate indices to ensure no query document self-retrieves.
- **Methodology Integrity**: Existing evaluation metrics are computed on strictly independent candidate sets without altering ground-truth evaluation logic.

---

## 7. Embedding Configuration

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Dimension**: `384`
- **Max Sequence Length**: `128` tokens
- **Batch Size**: `128`
- **PyTorch Threading**: `4` CPU threads
- **Normalization**: L2 normalized embeddings ($\|v\|_2 = 1.0$)

---

## 8. Actual Build Runtime

- **Total Execution Time**: **794.25 seconds** (**13.23 minutes**)
- **Execution Constraint**: < 15 minutes (900 seconds)
- **Status**: **PASS (Fulfills reproducibility constraint under 15 minutes)**

---

## 9. Index Configuration

- **FAISS Index Type**: `IndexFlatIP` (Inner Product on L2-normalized vectors = Cosine Similarity)
- **Index File Size**: `23,040,045 bytes` (~23.0 MB binary index file)
- **Index Directory**: `data/processed/retrieval_index_15k/`
- **Chunking / Checkpoints**: 3 chunks of 5,000 documents each (resumable checkpointing in `checkpoints/`)

---

## 10. Validation Results

1. **Document Count**: Exactly 15,000 documents.
2. **Intent Representation**: All 17 AmazonHelp intents represented.
3. **Exact Duplicates**: 0 exact duplicate documents.
4. **Determinism**: 100% deterministic dataset generation and FAISS ordering (`seed = 42`).
5. **Vector Alignment**: 15,000 embedding vectors aligned to 15,000 JSONL document map entries.
6. **Vector Dimension**: Exactly 384 dimensions.
7. **L2 Normalization**: Confirmed ($\|v\|_2 = 1.0$).
8. **FAISS Consistency**: Verified via `faiss_index.ntotal == 15000`.
9. **Unit Tests**: `pytest tests/ -v` passed cleanly (**23 passed in 22.87s**).
10. **Retrieval Benchmark (800 test queries)**:

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Semantic Retrieval (FAISS IndexFlatIP)** | **0.2750** | **0.4500** | **0.5200** | **0.6150** | **0.3905** | 60.6 ms |
| **Intent-Aware Retrieval (Metadata Boost)** | **0.6300** | **0.6850** | **0.7050** | **0.7400** | **0.6669** | 5.03 ms |

---

## 11. Retrieval Evaluation Caveats

> [!IMPORTANT]
> The following methodological clarifications govern the interpretation of retrieval evaluation results.

1. **Intent Source & Oracle Determination**:
   - **Intent Source Category**: **C (Heuristic / rule-based predicted intent)**.
   - **Mechanism**: The query intent passed to `IntentAwareRetriever` is dynamically generated at query time by `RuleBasedIntentClassifier.predict(query_text)`, which executes keyword regex pattern matching (`classify_single_text_intent`).
   - **Oracle Status**: The reported intent-aware retrieval result is **NOT** an oracle-assisted evaluation because the gold query intent label (`eval_intent`) is **NOT** used during retrieval or reranking. Ground-truth intent labels are used strictly within `compute_retrieval_metrics` at evaluation time to evaluate whether retrieved document metadata matches true intent for Recall@k and MRR calculations.

2. **Latency Measurement Methodology**:
   - **Scope**: Reported latencies (60.61 ms for Semantic, 5.03 ms for Intent-Aware) measure total execution wall-clock time divided by 800 test queries within the evaluation loop.
   - **Inclusions**: Includes query text extraction, Transformer embedding generation (`embedder.embed_query`), FAISS vector similarity search, keyword classifier inference (for Intent-Aware), candidate metadata score boosting (`+0.15`), candidate sorting, dynamic self-case ID filtering (`candidate.case_id != target.case_id`), and Python metric computation.
   - **Comparability Flag**: The reported latency numbers are **NOT directly comparable** due to PyTorch model cold-start and framework warm-up overhead on the initial evaluation pass (`semantic_retriever`), which artificially inflated semantic latency to 60.61 ms compared to the subsequent warm run (`intent_aware_retriever` at 5.03 ms).
   - **Production End-to-End Latency**: **Not measured in this phase.**

3. **15K Headline Corpus vs. 81.1K Canonical Corpus Performance**:
   - **Observed Recall Difference**: Canonical 81.1K corpus semantic Recall@10 = **86.50%**, whereas 15K headline corpus semantic Recall@10 = **61.50%**.
   - **Cause**: Subsampling the retrieval candidate pool from 81,186 documents to 15,000 documents (~18.5% sample) physically removes >81% of historical resolution candidates from the index. Consequently, fewer query-matching resolution documents exist in the 15K index.
   - **Interpretation Notice**: The 15K headline corpus is a lightweight, reproducible benchmark configuration designed to satisfy the assignment's <15-minute runtime constraint. 15K retrieval metrics **should NOT be interpreted as full-corpus production performance**.

---

## 12. Exact Reproduction Commands

```bash
# 1. Create the deterministic 15K stratified corpus
python scripts/create_15k_retrieval_corpus.py

# 2. Build the FAISS vector index (bounded CPU runtime ~13.23 minutes)
python scripts/build_retrieval_index.py --corpus data/processed/amazonhelp_rag_documents_15k.jsonl --output-dir data/processed/retrieval_index_15k --batch-size 128 --threads 4

# 3. Evaluate retrieval performance with self-case leakage protection
python scripts/evaluate_retrieval.py --index-dir data/processed/retrieval_index_15k

# 4. Run automated test suite
pytest tests/ -v
```

---

## 13. Limitations

- **Subsample Representation**: The 15,000-document corpus is a reproducible subsample designed for headline retrieval evaluation under strict time limits (<15 mins). It does **not** replace the full 81,186-document canonical production corpus.
- **Relevance Label Granularity**: Retrieval metrics use intent-aware proximate relevance matching; fine-grained human pairwise relevance annotations were not available across all candidate historical threads.
