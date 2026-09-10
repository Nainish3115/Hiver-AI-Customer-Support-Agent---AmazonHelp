# Phase 6: Human Generation Evaluation Annotation Guide

> **Dataset Path**: [`data/golden/amazonhelp_generation_human_eval_50.csv`](file:///D:/hiver-support-agent/data/golden/amazonhelp_generation_human_eval_50.csv)  
> **Total Cases**: **50 Stratified Cases**  
> **Status**: **READY FOR HUMAN ANNOTATION**  

---

## 1. Overview & Guidelines

This document provides standardized scoring rules for human annotators evaluating AI-generated customer support responses for `AmazonHelp`.

Annotators should review each row in `amazonhelp_generation_human_eval_50.csv` and populate the blank rating columns on a **1–5 scale**.

---

## 2. Evaluation Rating Dimensions (1–5 Scale)

| Rating Column | Description | 1 (Worst) | 3 (Acceptable) | 5 (Best) |
| :--- | :--- | :--- | :--- | :--- |
| `human_groundedness` | Response strictly supported by historical evidence without invented facts. | Fabricated policies, refunds, or dates. | Minor extrapolations, mostly faithful. | 100% grounded in retrieved evidence. |
| `human_helpfulness` | Direct usefulness to customer in resolving their problem. | Unhelpful or generic refusal. | Basic helpful guidance. | Direct, clear resolution or next steps. |
| `human_correctness` | Factual accuracy relative to customer query context. | Factually incorrect guidance. | Partially accurate. | Factually flawless relative to evidence. |
| `human_hallucination` | Hallucination risk assessment. | High hallucination (unsupported claims). | Moderate risk. | Zero hallucination (completely faithful). |

---

## 3. Special Rule: `INSUFFICIENT_EVIDENCE` Handling
- If `generated_response` is `"INSUFFICIENT_EVIDENCE"`, rate `human_groundedness` as **5** (safe abstention) if historical evidence was genuinely missing or ambiguous.
- Rate `human_helpfulness` as **2-3** depending on whether asking for info was appropriate.

---

## 4. Annotation Instructions
1. Open [`data/golden/amazonhelp_generation_human_eval_50.csv`](file:///D:/hiver-support-agent/data/golden/amazonhelp_generation_human_eval_50.csv).
2. For each row, inspect `customer_message`, `retrieved_case_ids`, and `generated_response`.
3. Assign integer scores (1 to 5) in `human_groundedness`, `human_helpfulness`, `human_correctness`, `human_hallucination`.
4. Add concise notes in `human_notes` for any flagged failures.
