# Phase 4 Mandatory Completion Report — Hiver Support Agent

> **Execution Timestamp**: 2026-09-09T21:59:00+05:30  
> **Phase Objective**: Build and evaluate AmazonHelp Intent Classification baselines, Historical-Support FAISS Vector Retrieval, create a deterministic 15,000-document reproducible headline retrieval corpus/index under <15-minute runtime constraint, and conduct a methodology audit of evaluation metrics.  
> **Status**: **PHASE 4 COMPLETE & METHODOLOGY AUDIT VERIFIED (Awaiting Phase 5 Approval)**  

---

## 1. Phase Overview

- **Phase Number**: Phase 4 (including Phase 4 Closeout & Methodology Audit)
- **Objective**: Establish the core intelligence foundation consisting of:
  1. Supervised & embedding prototype intent classification for `AmazonHelp`.
  2. Historical-support RAG document vector indexing using FAISS.
  3. Intent-aware vs pure semantic retrieval evaluation on both canonical (81.1K) and headline (15K) corpora.
  4. Auxiliary benchmark on Banking77 dataset.
  5. Deterministic, leakage-safe, stratified 15,000-document headline retrieval corpus/index satisfying the <15-minute runtime requirement.
  6. Comprehensive methodology audit of query intent sources, oracle-assistance, evaluation loop latencies, and corpus scale differences.
- **Date/Time of Execution**: 2026-09-09
- **Scope Boundaries**: Response generation, grounding verifiers, escalation engines, production classifiers, and Streamlit UI were explicitly **NOT** built in this phase.

---

## 2. Implementation Summary

- **Taxonomy & Schema**: Defined 17 `AmazonHelp` support intent categories in `src/hiver_agent/classification/taxonomy.py`.
- **Classification Baselines**: Implemented `RuleBasedIntentClassifier` (pattern matching) and `EmbeddingPrototypeClassifier` (SentenceTransformer nearest-centroid).
- **Retrieval Infrastructure**: Implemented `SentenceTransformerEmbeddingProvider`, `FAISSVectorStore` (`IndexFlatIP`), `FAISSRetriever`, and `IntentAwareRetriever`.
- **Resumable Index Builder**: Created chunk-checkpointed runner in `scripts/build_retrieval_index.py` supporting bounded execution (`--max-docs N`) and resumable 5,000-document checkpoints.
- **Stratified 15K Corpus Generator**: Created `scripts/create_15k_retrieval_corpus.py` to sample a deterministic, stratified, deduplicated 15,000-document headline corpus from the 81,186 canonical corpus using fixed seed `42`.
- **Diagnostic Benchmarking**: Created `scripts/benchmark_embedding.py` evaluating model loading, sequence length, and PyTorch CPU thread count.
- **Evaluation Harness**: Implemented `scripts/evaluate_retrieval.py` supporting dynamic self-case ID filtering and intent-aware candidate score boosting.

---

## 3. Data / Experiment Results

### Dataset Sizes
- **Canonical Source Corpus**: `data/processed/amazonhelp_rag_documents.jsonl` (81,186 documents).
- **Headline Reproducible Corpus**: `data/processed/amazonhelp_rag_documents_15k.jsonl` (15,000 documents).
- **Evaluation Dataset**: `data/golden/taxonomy_validation_sample.parquet` (800 test queries).

### Intent Classification Performance

| Model / Baseline | Dataset | Accuracy | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Rule-Based Classifier** | AmazonHelp Test (80 rows) | **0.6250** | **0.5841** | **0.6120** |
| **Embedding Prototype Classifier** | AmazonHelp Test (80 rows) | **0.7875** | **0.7632** | **0.7810** |
| **Banking77 Auxiliary Classifier** | Banking77 Test (3,076 rows) | **0.8501** | **0.8485** | **0.8486** |

### Vector Retrieval Performance (15K Headline Corpus)

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Semantic Retrieval (FAISS IndexFlatIP)** | **0.2750** | **0.4500** | **0.5200** | **0.6150** | **0.3905** | 60.6 ms |
| **Intent-Aware Retrieval (Metadata Reranked)** | **0.6300** | **0.6850** | **0.7050** | **0.7400** | **0.6669** | 5.03 ms |

- **Exact Pairwise Human Relevance**: Not measured in this phase.
- **Production End-to-End Latency**: Not measured in this phase.

---

## 4. Retrieval Evaluation Audit & Caveats

> [!IMPORTANT]
> The following section contains the findings of the Phase 4 Retrieval Evaluation Methodology Audit.

### Audit Objective
Audit the evaluation pipeline in `scripts/evaluate_retrieval.py` and `src/hiver_agent/retrieval/retriever.py` to identify the source of query intent, check for oracle-assistance, evaluate latency measurement methodology, and explain corpus scale performance differences between the 81.1K canonical corpus and the 15K headline corpus.

### Exact Code Paths Inspected
1. `scripts/evaluate_retrieval.py` (L87–90, L114–122): Evaluation execution loop and metric timing.
2. `src/hiver_agent/retrieval/retriever.py` (L63–100): `IntentAwareRetriever.retrieve()` implementation.
3. `src/hiver_agent/classification/rule_classifier.py` (L10–30): `RuleBasedIntentClassifier.predict()` implementation.

### Key Audit Findings

1. **Source of Query Intent**:
   - **Classification**: **Category C (Heuristic / rule-based predicted intent)**.
   - **Exact Code Flow**: `IntentAwareRetriever.retrieve()` calls `self.classifier.predict(query_text)`, where `self.classifier` is instantiated as `RuleBasedIntentClassifier()`. This classifier executes regex keyword matching (`classify_single_text_intent`).
   - **Oracle Status**: The reported intent-aware retrieval result is **NOT** an oracle-assisted evaluation because the gold query intent label (`eval_intent`) is **NOT** used during retrieval or reranking. Ground-truth intent labels are used strictly within `compute_retrieval_metrics` at evaluation time to evaluate whether retrieved document metadata matches true intent for Recall@k and MRR calculations.

2. **Retrieval Pipeline Architectures**:
   - **Pure Semantic Pipeline**:
     `query_text` $\rightarrow$ `embedder.embed_query()` $\rightarrow$ FAISS `IndexFlatIP` top-$k$ $\rightarrow$ dynamic self-case ID filter (`c.case_id != eval_cid`) $\rightarrow$ evaluation metric calculation.
   - **Intent-Aware Pipeline**:
     `query_text` $\rightarrow$ `RuleBasedIntentClassifier.predict()` (yields `pred.intent` and `pred.confidence`)  
     AND `query_text` $\rightarrow$ FAISS top-$25$ candidate over-retrieval  
     $\rightarrow$ Reranking (if `pred.confidence >= 0.60`, add `+0.15` boost to candidates where `cand.intent == pred.intent`)  
     $\rightarrow$ Sort candidates by boosted score $\rightarrow$ dynamic self-case ID filter $\rightarrow$ evaluation metric calculation.
   - **Entry Point of Intent**: Intent enters in the intent-aware pipeline during reranking via keyword classifier prediction. In the semantic pipeline, intent is ignored during retrieval and only checked at evaluation time to measure recall.

3. **Latency Measurement Methodology**:
   - **Scope**: Reported latencies (60.61 ms for Semantic, 5.03 ms for Intent-Aware) measure total wall-clock time per query inside `compute_retrieval_metrics`.
   - **Inclusions**: Includes query string extraction, embedding generation, FAISS search, classifier inference, metadata boosting, candidate sorting, dynamic self-case filtering, and metric calculation.
   - **Comparability Flag**: The reported latency numbers are **NOT directly comparable** due to PyTorch model cold-start and framework warm-up overhead on the initial evaluation pass (`semantic_retriever`), which artificially inflated semantic latency to 60.61 ms compared to the subsequent warm run (`intent_aware_retriever` at 5.03 ms).
   - **Production End-to-End Latency**: **Not measured in this phase.**

4. **15K Headline Corpus vs. 81.1K Canonical Corpus Comparison**:
   - **Metrics Comparison**: Canonical 81.1K corpus semantic Recall@10 = **86.50%**, whereas 15K headline corpus semantic Recall@10 = **61.50%**.
   - **Cause**: Subsampling the retrieval candidate pool from 81,186 documents to 15,000 documents (~18.5% sample) physically removes >81% of historical resolution candidates from the index. Consequently, fewer query-matching resolution documents exist in the 15K index.
   - **Interpretation Notice**: The 15K headline corpus is a lightweight, reproducible benchmark configuration designed to satisfy the assignment's <15-minute runtime constraint. 15K retrieval metrics **should NOT be interpreted as full-corpus production performance**.

---

## 5. Files Created & Modified

### Created
1. `src/hiver_agent/classification/taxonomy.py`
2. `src/hiver_agent/classification/interface.py`
3. `src/hiver_agent/classification/rule_classifier.py`
4. `src/hiver_agent/classification/embedding_classifier.py`
5. `src/hiver_agent/classification/evaluation.py`
6. `src/hiver_agent/retrieval/embeddings.py`
7. `src/hiver_agent/retrieval/faiss_store.py`
8. `src/hiver_agent/retrieval/retriever.py`
9. `scripts/create_15k_retrieval_corpus.py`
10. `scripts/train_intent_classifier.py`
11. `scripts/evaluate_intent_classifier.py`
12. `scripts/build_retrieval_index.py`
13. `scripts/benchmark_embedding.py`
14. `scripts/evaluate_retrieval.py`
15. `scripts/evaluate_banking77_baseline.py`
16. `tests/test_phase4.py`
17. `reports/phase_4_closeout.md`
18. `reports/phase_4_embedding_diagnostic.md`
19. `reports/phase_4_final_validation.md`
20. `reports/phase_4_completion.md`

### Modified
1. `src/hiver_agent/config.py`: Configuration paths for classification models, 15K corpus, and FAISS indices.
2. `pyproject.toml`: Dependencies for `faiss-cpu` and `sentence-transformers`.
3. `reports/phase_4_closeout.md`: Updated to include `### Retrieval Evaluation Caveats`.
4. `reports/phase_4_completion.md`: Updated to include methodology audit findings and caveats.

---

## 6. Dependencies

- `faiss-cpu==1.15.0`: CPU-optimized FAISS vector similarity search index library.
- `sentence-transformers==4.0.1`: PyTorch transformer embedding framework (`all-MiniLM-L6-v2`).

---

## 7. Exact Tests & Results

- **Command**: `pytest tests/ -v`
- **Result**: **23 passed in 23.32s**
- **Test Suite Verification**:
  - `tests/test_amazonhelp_threads.py`: 5 passed
  - `tests/test_config.py`: 2 passed
  - `tests/test_loader.py`: 3 passed
  - `tests/test_phase4.py`: 6 passed
  - `tests/test_rag_and_retrieval.py`: 4 passed
  - `tests/test_schema.py`: 3 passed

---

## 8. Leakage & Validation

- **Self-Case Filtering**: Verified dynamic filtering of self `case_id` (`candidate.case_id != target.case_id`) in `scripts/evaluate_retrieval.py`.
- **Exact Duplicate Checks**: Zero exact duplicate documents present in the 15K headline corpus.
- **Determinism**: 100% reproducible sampling using fixed random seed (`seed = 42`) and deterministic sorting.

---

## 9. Performance

- **Actual 15K Index Build Runtime**: **794.25 seconds** (**13.23 minutes**), strictly satisfying the <15-minute requirement.
- **Model Load Time**: ~1.32 seconds (`all-MiniLM-L6-v2`).
- **Embedding Throughput**: ~21.1 docs/sec on CPU (`threads=4`, `batch_size=128`).
- **Memory RSS**: ~250 MB base model, ~380 MB peak during FAISS indexing.

---

## 10. Problems & Risks

- **Full Corpus Encoding Overhead**: Embedding all 81,186 canonical documents on CPU requires ~64 minutes. Solved via the stratified 15,000-document headline corpus which embeds in ~13.23 minutes while maintaining complete intent stratification.
- **Latency Warm-up Artifact**: Latency loop evaluation in `scripts/evaluate_retrieval.py` is subject to PyTorch cold-start overhead on the first pass. Isolated production latency should be measured in Phase 5.

---

## 11. Exact Output Artifacts

- `data/processed/amazonhelp_rag_documents_15k.jsonl`: 15,000-document headline RAG corpus.
- `data/processed/retrieval_index_15k/faiss_index.bin`: 23,040,045 byte FAISS vector index.
- `data/processed/retrieval_index_15k/document_map.json`: 15,000 document metadata mapping.
- `data/processed/retrieval_index_15k/index_meta.json`: Index configuration file.
- `reports/phase_4_closeout.md`: Closeout report with audit caveats.
- `reports/phase_4_completion.md`: Mandatory completion report with audit findings.

---

## 12. Exact Reproduction Commands

```bash
# 1. Train and evaluate AmazonHelp intent classifier
python scripts/train_intent_classifier.py
python scripts/evaluate_intent_classifier.py

# 2. Generate deterministic 15K stratified headline corpus
python scripts/create_15k_retrieval_corpus.py

# 3. Build FAISS index for 15K headline corpus (< 15 mins CPU runtime)
python scripts/build_retrieval_index.py --corpus data/processed/amazonhelp_rag_documents_15k.jsonl --output-dir data/processed/retrieval_index_15k --batch-size 128 --threads 4

# 4. Run leakage-safe retrieval evaluation
python scripts/evaluate_retrieval.py --index-dir data/processed/retrieval_index_15k

# 5. Evaluate auxiliary Banking77 benchmark
python scripts/evaluate_banking77_baseline.py

# 6. Run automated test suite
pytest tests/ -v
```

---

## 13. Recommendation for Phase 5

- **Recommendation**: Phase 4 intelligence foundation, 15K headline corpus/index closeout, and retrieval evaluation methodology audit are complete, leakage-resistant, and fully verified. The repository is in a clean, reproducible state and ready for **Phase 5 (Response Generation & Escalation Engine)** approval.
- **Mandatory Stop Condition**: **STOPPING HERE**. Phase 5 has NOT been started. Awaiting explicit user approval before proceeding to Phase 5.
