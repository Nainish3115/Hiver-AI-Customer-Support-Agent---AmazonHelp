# Phase 8: End-to-End Evaluation, Final Benchmark & System Report

> **Project**: Hiver SDE Intern Take-Home — AI Customer Support Agent  
> **Target Brand**: AmazonHelp (Twitter Support Corpus)  
> **Status**: **COMPLETE & ASSIGNMENT-READY (Evaluation-Ready Prototype)**  

---

## 1. Executive Summary

Phase 8 completes the end-to-end evaluation, benchmarking, and architectural packaging of the `AmazonHelp` customer support agent. The system integrates intent classification, vector retrieval, evidence selection, Gemini response generation, grounding safety checks, and risk-aware human escalation decisioning into a single unified pipeline.

---

## 2. Benchmark Summary

| Sub-System | Metric | Result | Benchmark Dataset / Setup |
| :--- | :--- | :--- | :--- |
| **Intent Classification** | Accuracy | **57.5%** | 200 Human Gold Benchmark Cases |
| | Macro F1 | **0.5841** | 200 Human Gold Benchmark Cases |
| | Weighted F1 | **0.5841** | 200 Human Gold Benchmark Cases |
| **Retrieval Engine** | Recall@1 | **49.0%** | 200 Gold Cases vs 15k RAG Corpus |
| | Recall@3 | **73.5%** | 200 Gold Cases vs 15k RAG Corpus |
| | Recall@5 | **81.5%** | 200 Gold Cases vs 15k RAG Corpus |
| | Recall@10 | **89.5%** | 200 Gold Cases vs 15k RAG Corpus |
| | MRR | **0.6384** | 200 Gold Cases vs 15k RAG Corpus |
| **Response Generation** | LLM Judge Overall | **4.72 / 5.0** | Gemini 1.5 Flash Live Run ($N=50$) |
| | Safety & Grounding Pass | **100.0%** | Gemini 1.5 Flash Live Run ($N=50$) |
| **Decision & Escalation** | Model–Human Decision Agreement | **74.0%** | 50 Human Escalation Validation Set |
| | Model Auto-Handle Rate | **82.0%** | 50 Human Escalation Validation Set |
| | False Auto-Handle Rate | **12.0%** (14.6% in Auto) | 50 Human Escalation Validation Set |
| | False Escalation Rate | **14.0%** (77.8% in Esc) | 50 Human Escalation Validation Set |

---

## 3. Latency Breakdown (Local CPU Pipeline)

- **Intent Classification**: 0.36 ms
- **Vector Retrieval**: 2.36 ms
- **Evidence Selection**: 0.25 ms
- **Response Generation (Local Template Baseline)**: 0.1 ms
- **Decision Engine**: 0.06 ms
- **Total Local Pipeline Latency**: **3.13 ms**
- *Note: Live Gemini 1.5 Flash API calls add ~1.85 seconds network overhead per generation call.*

---

## 4. Top 5 End-to-End Failure Modes

1. **Category Risk Over-Penalty (False Escalation - 5 cases)**: Conservative category risk penalties force escalation on routine queries in sensitive categories (`refund_status_request`, `account_access_login`, `payment_billing_issue`).
2. **Ambiguous Query Under-Escalation (False Auto-Handle - 3 cases)**: Broad inquiries classified as routine inquiries with high confidence despite requiring human review.
3. **Complex Dispute Bypassing Risk (False Auto-Handle - 2 cases)**: Complex promotional EMI disputes auto-handled due to high classifier confidence.
4. **Physical Delivery Anomaly Unchecked (False Auto-Handle - 1 case)**: Delivery anomaly auto-handled as standard delay.
5. **Weak Retrieval Score (False Escalation - 2 cases)**: Routine inquiries escalated due to top retrieval similarity score falling below 0.70.

---

## 5. System Status & Final Recommendation

The system is classified as an **Evaluation-Ready Prototype**. It is fully reproducible, deterministic, and modular.
