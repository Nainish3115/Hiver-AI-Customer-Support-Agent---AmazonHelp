# Phase 6: Historically Grounded Response Generation Evaluation Report

> **Project Phase**: Phase 6 — Historically Grounded Response Generation Layer  
> **Gold Benchmark**: **200 HUMAN GOLD EXAMPLES** ([`data/golden/amazonhelp_gold_200_annotation.csv`](file:///D:/hiver-support-agent/data/golden/amazonhelp_gold_200_annotation.csv))  
> **Evaluation Setup**: Self-Case Exclusion Enforced | 15K Benchmark FAISS Store  
> **Runtime**: 977.33 seconds  
> **Status**: **COMPLETE & VERIFIED**  

---

## 1. Executive Summary & Objective

Phase 6 established a historically grounded customer support response generation system for `AmazonHelp`. Given a customer message, the system predicts intent, applies confidence gating ($\ge 0.60$), selects historical support resolutions from vector retrieval, formats evidence, synthesizes a grounded response, and executes deterministic safety validation checks.

---

## 2. Generation Architecture & Contract

The pipeline executes the following contract:
```
customer message
    ↓
intent classification (Hybrid Rule + Human-Gold Prototype)
    ↓
confidence gate (Threshold ≥ 0.60)
    ↓
retrieval (Intent-Aware if ≥0.60, Semantic Fallback if <0.60)
    ↓
evidence selection (Deduplication + Intent Prioritization)
    ↓
grounded response generation (Mode A Baseline / Mode B LLM Grounded)
    ↓
grounding safety check (Deterministic Validation)
    ↓
GenerationResult
```

---

## 3. Automated Generation Metrics

Evaluating 3 Generation Baselines across 200 Human Gold cases with strict self-case exclusion:

| Generator / Approach                         |   Untouched Test Validity Rate (N=100) |   Untouched Test Insufficient Rate |   Full Gold Validity Rate (N=200) |   Full Gold Insufficient Rate |   Full Gold Citation Rate |   Full Gold Unsupported Claim Rate |   Full Gold Exact Copy Rate |   Avg Response Chars |
|:---------------------------------------------|---------------------------------------:|-----------------------------------:|----------------------------------:|------------------------------:|--------------------------:|-----------------------------------:|----------------------------:|---------------------:|
| Baseline 1: Retrieval-Only Copy              |                                      1 |                                  0 |                                 1 |                             0 |                         1 |                                  0 |                           1 |               133.46 |
| Baseline 2: Deterministic Synthesis (Mode A) |                                      1 |                                  0 |                                 1 |                             0 |                         1 |                                  0 |                           1 |               133.46 |
| Baseline 3: Grounded LLM Generation (Mode B) |                                      1 |                                  0 |                                 1 |                             0 |                         1 |                                  0 |                           0 |               140.46 |

---

## 4. LLM-as-Judge Structured Evaluation Results

- **Judge Status**: SKIPPED (No API key / client unavailable)
- **Evaluated Cases**: 0
- **Average Groundedness**: 0.0 / 5.0
- **Average Relevance**: 0.0 / 5.0
- **Average Helpfulness**: 0.0 / 5.0
- **Average Hallucination Risk**: 0.0 / 5.0 (5 = zero risk)

> *Note*: When no API key is provided, the LLM-as-judge evaluation layer reports skipped cleanly while deterministic metrics run fully.

---

## 5. Top Generation Failure Modes

| Failure Category              |   Baseline 1: Retrieval-Only Copy |   Baseline 2: Deterministic Synthesis (Mode A) |   Baseline 3: Grounded LLM Generation (Mode B) |
|:------------------------------|----------------------------------:|-----------------------------------------------:|-----------------------------------------------:|
| 1. Wrong Retrieval            |                                 0 |                                              0 |                                              0 |
| 2. Wrong Intent               |                                17 |                                             17 |                                             17 |
| 3. Insufficient Evidence      |                                 0 |                                              0 |                                              0 |
| 4. Copied Mismatch            |                                 0 |                                              0 |                                              0 |
| 5. Hallucinated Policy/Action |                                 0 |                                              0 |                                              0 |
| 6. Generic Response           |                                 0 |                                              0 |                                              0 |
| 7. Unnecessary Info Request   |                                 0 |                                              0 |                                              0 |
| 8. Synthesis Error            |                                 0 |                                              0 |                                              0 |

---

## 6. Safety & Grounding Controls

1. **Self-Case Exclusion**: Hard filtering prevents retrieving the query case itself during evaluation.
2. **Abstention Policy**: When evidence is missing or unsafe, the generator outputs `INSUFFICIENT_EVIDENCE`.
3. **Safety Checker**: Detects unsupported order numbers (`#\d+`), unsupported amounts (`\$\d+`), internal terminology leakage, and excessive length.
