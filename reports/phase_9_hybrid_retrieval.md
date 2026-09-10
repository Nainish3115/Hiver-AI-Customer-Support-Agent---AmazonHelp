# Phase 9: Hybrid Retrieval Improvement & Evaluation Audit Report

> **Project**: Hiver SDE Intern Take-Home — AI Customer Support Agent  
> **Target Brand**: AmazonHelp (Twitter Support Corpus)  
> **Status**: **COMPLETE & AUDITED (Phase 9.1 Evaluation Integrity Verified)**  

---

## 1. Executive Summary

Phase 9 implements a **Hybrid Retrieval Architecture** combining **Dense Semantic Retrieval** (`all-MiniLM-L6-v2` + FAISS), **Lexical Retrieval** (BM25Okapi), and **Reciprocal Rank Fusion (RRF)**, followed by intent-aware metadata reranking.

The evaluation benchmarks 5 retrieval configurations on the $N=200$ Human Gold Benchmark against the 15,316 document historical resolution RAG corpus.

---

## 2. Phase 4 vs Phase 9 Baseline Reconciliation

Phase 4 previously reported an Intent-Aware retrieval benchmark of R@1 = 63.0% and MRR = 0.6669, whereas Phase 9 reports R@1 = 51.5% and MRR = 0.5379. Below is the side-by-side methodology reconciliation:

| Dimension | Phase 4 Benchmark Protocol | Phase 9 Benchmark Protocol | Impact / Explanation |
| :--- | :--- | :--- | :--- |
| **Evaluation Dataset** | `taxonomy_validation_sample.parquet` ($N=200$) | `amazonhelp_gold_200_annotation.csv` ($N=200$) | Phase 4 used a heuristic sample; Phase 9 uses the human-gold annotated dataset. |
| **Ground-Truth Target Intent** | `proposed_intent` (Rule-generated pseudo-label) | `final_human_intent` (Human gold label) | Phase 4 evaluated against rule-generated pseudo-labels, artificially inflating intent match rates. |
| **Classifier in Intent-Aware** | `RuleBasedIntentClassifier` | `HybridRuleEmbeddingClassifier` | Phase 4 classifier matched the heuristic label generator; Phase 9 classifier evaluates against human gold (57.5% acc). |
| **Pure Dense R@1** | **27.5%** | **28.0%** | Dense semantic retrieval performance is highly consistent (~27.5–28.0%) across both datasets. |
| **Intent-Aware Dense R@1** | **63.0%** | **51.5%** | Phase 9 reflects realistic human intent classification error propagation under human gold labels. |
| **MRR** | **0.6669** | **0.5379** | Phase 9 MRR is evaluated strictly against human gold intent targets. |

---

## 3. Benchmark Comparison Table

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Mean Latency (ms) | Monotonicity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Baseline (FAISS)** | **28.0%** | **40.5%** | **49.0%** | **59.5%** | **0.3666** | 16.02 ms | Verified ($R@1 \le R@3 \le R@5 \le R@10$) |
| **Intent-Aware Dense** | **51.5%** | **55.5%** | **56.0%** | **60.0%** | **0.5379** | 2.49 ms | Verified ($R@1 \le R@3 \le R@5 \le R@10$) |
| **BM25 Lexical** | **28.5%** | **46.0%** | **55.5%** | **71.5%** | **0.4048** | 139.96 ms | Verified ($R@1 \le R@3 \le R@5 \le R@10$) |
| **Hybrid RRF (Dense + BM25)** | **36.0%** | **50.0%** | **58.5%** | **69.5%** | **0.4543** | 145.49 ms | Verified ($R@1 \le R@3 \le R@5 \le R@10$) |
| **Hybrid + Intent Rerank** | **53.5%** | **55.5%** | **58.0%** | **65.5%** | **0.5591** | 145.18 ms | Verified ($R@1 \le R@3 \le R@5 \le R@10$) |

*Note: Retrieval ground truth uses the intent-alignment retrieval proxy as established in Phase 4 & Phase 8.*

---

## 4. Candidate Overlap & Complementarity Analysis

To avoid conflating candidate set overlap with retrieval success, metrics are separated into document-level and case-level formulations:

### A. Document-Level Top-10 Overlap
- **Average Top-10 Document Overlap**: **0.73 / 10** documents shared between Dense and BM25 candidate lists per query.

### B. Case-Level Retrieval Success Attribution (at Top-10)
- **Dense ONLY**: **8.0%** (16/200 cases) — Dense retrieved a relevant candidate in Top-10 while BM25 failed.
- **BM25 ONLY**: **20.0%** (40/200 cases) — BM25 retrieved a relevant candidate in Top-10 while Dense failed.
- **BOTH**: **51.5%** (103/200 cases) — Both Dense and BM25 retrieved a relevant candidate in Top-10.
- **NEITHER**: **20.5%** (41/200 cases) — Neither retriever found a relevant candidate in Top-10.

---

## 5. Corpus & Benchmark Scope

- **Canonical AmazonHelp Corpus**: 81,186 valid multi-turn dialog documents reconstructed from Twitter support data.
- **Phase 9 Benchmark Corpus**: 15,316 filtered resolution documents containing complete customer problem descriptions and resolved agent actions.
- **Rationale for Benchmark Corpus**: Filtering eliminates unhelpful generic DM redirection tweets ("Please DM us your email"), providing high-quality grounded resolution precedents.
- **Active Production Pipeline Index**: Points to `data/processed/faiss_index_15k` (15,316 documents).

---

## 6. Latency Analysis

No production SLA was established for this assignment. Local CPU retrieval latencies measured per query:
- Pure Dense (FAISS): **16.02 ms**
- Intent-Aware Dense: **2.49 ms**
- BM25 Lexical: **139.96 ms**
- Hybrid RRF (Dense + BM25): **145.49 ms**
- Hybrid + Intent Rerank: **145.18 ms**

Hybrid retrieval increases measured local CPU retrieval latency by ~129.16 ms per query relative to pure dense search due to BM25 Python tokenization over the 15,316 document corpus.

---

## 7. System Status & Final Recommendation

- **Recommendation**: **ADOPT HYBRID + INTENT RERANK AS NEW DEFAULT RETRIEVAL MODE**.
- **Reasoning**: Hybrid retrieval combined with intent-aware reranking achieves higher overall recall (R@1 = 53.5%) and MRR (0.5591) compared to pure dense baseline (28.0% R@1, 0.3666 MRR) and Intent-Aware Dense baseline (51.5% R@1, 0.5379 MRR), demonstrating strong lexical-semantic complementarity.
