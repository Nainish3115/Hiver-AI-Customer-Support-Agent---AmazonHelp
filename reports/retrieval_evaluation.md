# FAISS Vector Retrieval & Intent-Aware Retrieval Evaluation Report

> **Project Phase**: Phase 4 — Retrieval Pipeline Evaluation & Benchmark  
> **Index Location**: `data/processed/retrieval_index/` (15,000 indexed documents)  
> **Evaluation Dataset**: `data/golden/taxonomy_validation_sample.parquet` (200 test queries)  
> **Relevance Measure**: PROXIMATE RELEVANCE (Historical case matches query intent category with valid resolution action).  
> **Exact Ground-Truth Relevance**: Not measured because manual pair-wise relevance labels were unavailable for all 81.1K corpus documents.

---

## 1. Retrieval Performance Comparison

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Average Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Semantic Retrieval (FAISS IndexFlatIP)** | **0.275** | **0.45** | **0.52** | **0.615** | **0.3905** | **60.61 ms** |
| **Intent-Aware Retrieval (Boosted Metadata Reranking)** | **0.63** | **0.685** | **0.705** | **0.74** | **0.6669** | **5.03 ms** |

---

## 2. Data Leakage Prevention Verification

- **Leakage Prevention Action**: For every evaluation query, its corresponding self-case (`case_id`) was dynamically removed from retrieved candidate lists before computing recall and MRR metrics.
- **Result**: Zero self-retrieval contamination.
