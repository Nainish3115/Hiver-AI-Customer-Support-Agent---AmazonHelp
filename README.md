# Hiver SDE Intern Take-Home — AI Customer Support Agent (AmazonHelp)

An end-to-end, RAG-powered Customer Support AI Agent built for `@AmazonHelp` Twitter customer support interactions.

---

## 1. Project Overview

This repository implements a complete, risk-aware AI customer support agent that classifies customer intent, retrieves grounded historical resolutions, synthesizes responses via Google Gemini 1.5 Flash, validates safety rules, and deterministically decides whether to **AUTO_HANDLE** or **ESCALATE_TO_HUMAN**.

---

## 2. System Architecture

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
4. Grounded Response Generator (Google Gemini 1.5 Flash API)
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

## 3. Dataset & 17-Intent Taxonomy

Reconstructed from 170,000+ tweets directed to `@AmazonHelp` into 15,316 clean dialog threads. Categorized using a domain-specific 17-intent taxonomy:

1. `order_tracking_delivery_status`
2. `late_delivery_shipping_delay`
3. `missing_package_delivered_not_received`
4. `damaged_defective_item`
5. `wrong_item_received`
6. `return_exchange_inquiry`
7. `refund_status_request`
8. `order_cancellation_request`
9. `payment_billing_issue`
10. `promotions_discounts_coupons`
11. `prime_membership_subscription`
12. `account_access_login`
13. `address_delivery_instructions`
14. `digital_content_kindle_video`
15. `third_party_seller_issue`
16. `customer_service_complaint`
17. `general_inquiry_other`

---

## 4. Benchmark Summary Table

| Sub-System | Metric | Value | Ground Truth / Setup |
| :--- | :--- | :--- | :--- |
| **Intent Classification** | Accuracy | **57.5%** | 200 Human Gold Intent Cases |
| | Macro F1 | **0.5841** | 200 Human Gold Intent Cases |
| | Weighted F1 | **0.5841** | 200 Human Gold Intent Cases |
| **Retrieval Engine** | Recall@3 | **73.5%** | 200 Gold Cases vs 15k FAISS Index |
| | MRR | **0.6384** | 200 Gold Cases vs 15k FAISS Index |
| **Response Generation** | LLM Judge Score | **4.72 / 5.0** | Gemini 1.5 Flash Live Run ($N=50$) |
| | Grounding & Safety Pass | **100.0%** | Gemini 1.5 Flash Live Run ($N=50$) |
| **Decision & Escalation** | Model–Human Decision Agreement | **74.0%** | 50 Human Escalation Validation Set |
| | Model Auto-Handle Rate | **82.0%** | 50 Human Escalation Validation Set |
| | False Auto-Handle Rate | **12.0%** (14.6% in Auto) | 50 Human Escalation Validation Set |
| | False Escalation Rate | **14.0%** (77.8% in Esc) | 50 Human Escalation Validation Set |

---

## 5. Quick Start & Installation

### Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
# OR using pyproject.toml
pip install -e .
```

### Environment Variables

Set your Google Gemini API Key:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

---

## 6. Reproducible Evaluation Commands

### Run Full Unit Test Suite

```bash
pytest tests/ -v
```

### Run End-to-End Evaluation Script

```bash
python scripts/evaluate_phase8_end_to_end.py
```

### Run Phase 7.1 Human Escalation Evaluation

```bash
python scripts/evaluate_phase7_1.py
```

---

## 7. Python API Quick Start

```python
from hiver_agent.config import load_config
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.retrieval.faiss_store import FAISSVectorStore
from hiver_agent.retrieval.retriever import FAISSRetriever
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from hiver_agent.generation.generator import TemplateResponseGenerator, LLMResponseGenerator
from hiver_agent.pipeline import CustomerSupportPipeline

# Load components
embedder = SentenceTransformerEmbeddingProvider()
vector_store = FAISSVectorStore.load("data/processed/faiss_index_15k")
base_retriever = FAISSRetriever(embedder, vector_store)
generator = LLMResponseGenerator(provider="gemini") # or TemplateResponseGenerator()

# Initialize Pipeline
pipeline = CustomerSupportPipeline(
    classifier=...,
    base_retriever=base_retriever,
    generator=generator
)

# Process message with decision
result = pipeline.process_end_to_end("My package has not arrived yet. Order ID: 403-1234567-8901234")
print(f"Decision: {result.decision}")
print(f"Reason: {result.decision_reason}")
print(f"Response: {result.draft_response}")
```

---

## 8. Key Reports & Documentation Artifacts

- **End-to-End Evaluation Report**: [`reports/phase_8_final_evaluation.md`](file:///D:/hiver-support-agent/reports/phase_8_final_evaluation.md)
- **6-Page Assignment Report**: [`reports/final_report.md`](file:///D:/hiver-support-agent/reports/final_report.md)
- **System Architecture**: [`reports/phase_8_system_architecture.md`](file:///D:/hiver-support-agent/reports/phase_8_system_architecture.md)
- **Engineering Decision Log**: [`reports/decision_log.md`](file:///D:/hiver-support-agent/reports/decision_log.md)
- **Human Escalation Guidelines**: [`reports/phase_7_1_escalation_annotation_guidelines.md`](file:///D:/hiver-support-agent/reports/phase_7_1_escalation_annotation_guidelines.md)
- **Case Traces Table**: [`reports/tables/phase_8_case_traces.csv`](file:///D:/hiver-support-agent/reports/tables/phase_8_case_traces.csv)
- **Failure Attribution Table**: [`reports/tables/phase_8_failure_attribution.csv`](file:///D:/hiver-support-agent/reports/tables/phase_8_failure_attribution.csv)

---

## 9. Top Known Limitations

1. **Validation Sample Size ($N=50$)**: Human escalation ground truth is evaluated on 50 validation cases.
2. **Single Annotator**: Single-annotator setup; inter-annotator agreement (Cohen's Kappa) was not measured.
3. **Evaluation-Ready Prototype**: Designed for evaluation and benchmarking; production deployment requires API gateway, rate-limiting, and auth.
