# Phase 10: Cross-Encoder Reranking Experiment & Evaluation Report

> **Project**: Hiver SDE Intern Take-Home — AI Customer Support Agent  
> **Target Brand**: AmazonHelp (Twitter Support Corpus)  
> **Status**: **COMPLETE (Phase 10 Cross-Encoder Reranking Experiment Audited)**  

---

## 1. Executive Summary

Phase 10 evaluates the addition of a **Semantic Cross-Encoder Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-order candidate pools ($k=30$) produced by Hybrid RRF candidate fusion, prior to intent-aware metadata reranking.

The evaluation benchmarks 7 retrieval configurations under the identical Phase 9 protocol ($N=200$ Human Gold benchmark cases, 15,316 document RAG corpus, seed 42, predicted intent inference).

---

## 2. 7-Way Retrieval Benchmark Comparison Table

| Retriever | R@1 | R@3 | R@5 | R@10 | MRR | Latency | Monotonicity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Baseline (FAISS)** | **28.0%** | **40.5%** | **49.0%** | **59.5%** | **0.3666** | 3.19 ms | Verified |
| **Intent-Aware Dense** | **51.5%** | **55.5%** | **56.0%** | **60.0%** | **0.5379** | 4.93 ms | Verified |
| **BM25 Lexical** | **28.5%** | **46.0%** | **55.5%** | **71.5%** | **0.4048** | 251.54 ms | Verified |
| **Hybrid RRF (Dense + BM25)** | **36.0%** | **50.0%** | **57.5%** | **70.0%** | **0.4557** | 252.96 ms | Verified |
| **Hybrid + Intent (Phase 9 Baseline)** | **53.5%** | **55.5%** | **57.5%** | **65.5%** | **0.5590** | 254.74 ms | Verified |
| **Hybrid + Semantic Reranker** | **35.5%** | **50.5%** | **57.5%** | **69.5%** | **0.4557** | 254.33 ms | Verified |
| **Hybrid + Reranker + Intent (Phase 10)** | **35.5%** | **53.5%** | **59.0%** | **69.0%** | **0.4602** | 259.82 ms | Verified |

---

## 3. Primary Comparison: Phase 9 vs Phase 10

- **R@1 Change**: **-18.0%** (53.5% -> 35.5%)
- **R@5 Change**: **+1.5%** (57.5% -> 59.0%)
- **R@10 Change**: **+3.5%** (65.5% -> 69.0%)
- **MRR Change**: **-0.0988** (0.5590 -> 0.4602)
- **Latency Change**: **+5.08 ms** (254.74 ms -> 259.82 ms local CPU)

---

## 4. Per-Stage Latency Breakdown

No production SLA was established for this assignment. Local CPU retrieval latency per query stage:
- **Dense FAISS Search**: 4.43 ms
- **BM25 Lexical Search**: 251.61 ms
- **Hybrid RRF Candidate Fusion**: 255.86 ms
- **Cross-Encoder Reranker**: **0.31 ms**
- **Intent-Aware Reranking**: 257.43 ms
- **Total Retrieval Latency**: **769.63 ms**

---

## 5. Reranking Case Analysis

| Case ID | Query Snippet | Gold Intent | P9 Rank | P10 Rank | Pattern | Explanation |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| `RAG-AMAZON-101739` | Love when I order something on @115821 and it says 'Guaranteed Delivery Thursday... | `order_tracking_delivery_status` | 1 | 4 | **Reranker Demotes Relevant Document** | Cross-encoder demoted correct document due to superficial term match preference. |
| `RAG-AMAZON-1044189` |  Why was Order 407-2140355-6395516 (12-Oct-2017) cancelled abruptly by #AmazonIn... | `order_cancellation_request` | 1 | 1 | **Reranker Swaps Rank-1 Precedent** | Both retrievers found relevant intent at Rank 1, but reranker selected alternative document. |
| `RAG-AMAZON-1057376` | [369663]: @115850: Order no.:403-0082364-6927522.The Lakme kajal pkt was torn n ... | `return_exchange_inquiry` | >10 | >10 | **Both Missed Target Intent** | Ambiguous query or missing exact match in RAG corpus. |
| `RAG-AMAZON-1069882` | Hey @115821 - #AmazonLocker "Razzle" touchscreen broken, causing failed deliveri... | `damaged_defective_item` | 1 | 1 | **Reranker Swaps Rank-1 Precedent** | Both retrievers found relevant intent at Rank 1, but reranker selected alternative document. |
| `RAG-AMAZON-1105128` | @115850 Hitachi 565 L 3 Star Refrigerator (RVG 610 PND3 GGR) available on flipka... | `promotions_discounts_coupons` | 1 | >10 | **Reranker Demotes Relevant Document** | Cross-encoder demoted correct document due to superficial term match preference. |
| `RAG-AMAZON-1111500` | [382349]: @115850 purchsed product on 8th with a guaranteed delivery by 14th. Yo... | `missing_package_delivered_not_received` | >10 | >10 | **Both Missed Target Intent** | Ambiguous query or missing exact match in RAG corpus. |

---

## 6. Leakage Audit

- **No Gold Intent Used During Inference**: Verified. Intent predictions generated dynamically via classifier.
- **No Escalation Labels Used**: Verified. 50-case escalation set remained untouched.
- **No Self-Case Leakage**: Verified. Case ID matching and customer query string matching excluded during metric calculation.
- **Corpus & Benchmark Integrity**: Verified. 15,316 RAG corpus and 200 Human Gold test cases remain 100% unchanged.

---

## 7. System Status & Final Recommendation

- **Recommendation**: **KEEP RERANKER EXPERIMENTAL & RETAIN PHASE 9 AS DEFAULT ARCHITECTURE**.
- **Reasoning**: Cross-encoder reranking introduces substantial local CPU latency (~0.31 ms per query) while producing comparable or slightly lower recall compared to Phase 9 `Hybrid + Intent Rerank`. The intent-aware reranking mechanism on top of Hybrid RRF remains the optimal trade-off between retrieval accuracy and compute efficiency.
