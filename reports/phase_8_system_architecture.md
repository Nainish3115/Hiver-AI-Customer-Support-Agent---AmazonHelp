# Phase 8: System Architecture & Component Specifications

This document outlines the canonical end-to-end architecture of the `AmazonHelp` Customer Support AI Agent.

```
Customer Message (Twitter @AmazonHelp)
        ↓
1. Hybrid Intent Classifier (Rule-Based Regex + MiniLM Embedding Prototype)
        ↓
   Predicted Intent + Confidence Score
        ↓
2. Vector & Intent-Aware Retrieval (Sentence-Transformers MiniLM-L6-v2 + FAISS Index)
        ↓
   Top-K Historical Resolution Case Documents (Self-Case Excluded)
        ↓
3. Evidence Selector (Deduplication, Intent Prioritization, Formatting)
        ↓
   Selected Historical Resolution Evidence
        ↓
4. Grounded Response Generator (Gemini 1.5 Flash / Grounded Synthesis)
        ↓
   Draft Customer Support Response
        ↓
5. Grounding Safety Checker (Entity, Order ID, URL & Length Verification)
        ↓
   Safety & Grounding Validation Metadata
        ↓
6. Risk-Aware Decision Engine (Hard Safety Rules + Composite Risk Signals)
        ↓
Final EndToEndResult [AUTO_HANDLE | ESCALATE_TO_HUMAN + Reason Code & Risk Signals]
```

---

## Component Specifications

| Layer | Primary Technology | Input | Output | Main Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **1. Intent Classifier** | Hybrid Regex + Embedding Prototype | Customer Text | Intent + Confidence | Broad intents (`general_inquiry_other`) overlap. |
| **2. Retriever** | `all-MiniLM-L6-v2` + FAISS | Customer Text | Top-10 SearchResults | Relevance uses intent proxy ground truth. |
| **3. Evidence Selector** | Deterministic Match & Deduplication | SearchResults | Selected Evidence Items | Limited to max 3 evidence cases. |
| **4. Generator** | Google Gemini 1.5 Flash API | Message + Evidence | Draft Response | Network API dependency & latency (~1.8s). |
| **5. Safety Checker** | Regex Entity & Length Validator | Draft Response | Safety Passed (Bool) | Surface entity check; does not verify deep logic. |
| **6. Decision Engine** | Weighted Risk Signal Calculator | Generation Result | AUTO / ESCALATE | Composite risk threshold fixed at 0.35. |
