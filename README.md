# Hiver SDE Intern Take-Home — AI Customer Support Agent (AmazonHelp)

An end-to-end, RAG-powered Customer Support AI Agent built for `@AmazonHelp` Twitter customer support interactions.

> **Evaluation Disclaimer**: *This repository represents a technical prototype evaluated on controlled benchmark datasets under local CPU conditions. It demonstrates architectural feasibility, risk-aware escalation logic, and empirical evaluation, rather than production SLA or safety guarantees.*

---

## 1. Project Overview

Customer support teams face high query volumes, driving up response latency and agent burnout. Automated AI support agents can automate routine inquiries but risk generating ungrounded, incorrect, or hallucinated responses. 

This repository implements a complete, risk-aware AI Customer Support Agent for `@AmazonHelp`. The pipeline classifies customer intents across a 17-class e-commerce taxonomy, retrieves grounded historical resolution precedents, synthesizes responses via **Google Gemini 1.5 Flash**, validates safety rules, and deterministically decides whether to **AUTO_HANDLE** or **ESCALATE_TO_HUMAN**.

---

## 2. Problem Statement

Support automation presents a fundamental trade-off:
- **Automation Coverage**: Resolving high-frequency routine inquiries quickly without human agent intervention.
- **Customer Safety & Precision**: Preventing incorrect or ungrounded responses from reaching customers.

Existing automated agents often rely on static confidence thresholds or ungrounded LLM completions, leading to hallucinated order details or unsafe resolution actions. Our system addresses this challenge by combining dense-lexical hybrid retrieval, grounded LLM prompt constraints, post-generation entity validation, and a multi-signal risk decision engine.

---

## 3. System Architecture

```
Customer Message (Twitter @AmazonHelp)
        ↓
1. Intent Classifier (Linear SVM with TF-IDF n-grams + Regex Rules)
        ↓
   Predicted Intent + Confidence Score
        ↓
2. Hybrid Dense + Lexical Retrieval (Sentence-Transformers MiniLM-L6-v2 + BM25Okapi)
        ↓
   Reciprocal Rank Fusion (RRF k=60) & Intent-Aware Metadata Reranking
   *(Cross-encoder disabled by default: reranking.enabled=false)*
        ↓
   Top-K Historical Resolution Case Documents (Self-Case Excluded)
        ↓
3. Evidence Selector (Deduplication, Intent Prioritization, Formatting)
        ↓
   Selected Historical Resolution Evidence
        ↓
4. Grounded Response Generator (Google Gemini 1.5 Flash API via Native GenAI SDK)
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

*Note on Reranking*: A SentenceTransformers Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is fully implemented (`src/hiver_agent/retrieval/reranker.py`) but remains **disabled by default** (`reranking.enabled: false` in `configs/config.yaml`) because Phase 10 evaluation proved that pure cross-encoder passage scoring degraded Rank-1 precedent precision (53.5% → 35.5% R@1).

---

## 4. Dataset & Thread Reconstruction

- **Raw Source**: 2.8M customer support tweets from the *Customer Support on Twitter* dataset.
- **Brand Corpus**: 170,000+ tweets directed to `@AmazonHelp`.
- **Thread Reconstruction**: Built a graph-based dialog engine that reconstructs multi-turn customer–agent conversation trees (`min_depth=2`, `max_depth=10`).
- **Canonical Corpus**: **81,186** valid multi-turn dialog documents.
- **Benchmark Corpus**: **15,316** clean, grounded problem-resolution thread pairs used for retrieval indexing.

---

## 5. 17-Intent Domain Taxonomy

Formulated specifically for e-commerce customer support interactions:

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

## 6. Retrieval & Historical Grounding System

- **Dense Semantic Vector Search**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, batch size 128) indexed in Flat L2 FAISS (`data/processed/faiss_index_15k`).
- **Lexical Search**: `BM25Okapi` index over tokenized customer query texts.
- **Reciprocal Rank Fusion (RRF)**: Merges dense and lexical candidate ranks ($k=60$).
- **Intent-Aware Metadata Reranking**: When classifier confidence $\ge 0.30$, candidates matching predicted intent receive an intent boost multiplier ($1.5\times$).
- **Self-Case Exclusion**: Strictly excludes target case IDs and exact query strings to eliminate evaluation data leakage.

---

## 7. Response Generation Layer

- **LLM Provider**: **Google Gemini 1.5 Flash** (`gemini-1.5-flash`) via the native `google-genai` SDK.
- **Prompt Architecture**: Strict system prompts enforce grounding in retrieved evidence snippets, prohibiting assumptions about unprovided order numbers, accounts, or refund amounts.
- **Fallbacks**: Local template-based response generator fallback available when API key is unconfigured.

---

## 8. Grounding & Entity Safety Checker

Post-generation deterministic validation checks:
- **Entity Regex Match**: Order ID format (`\d{3}-\d{7}-\d{7}`), tracking numbers, account IDs.
- **URL Syntax Verification**: Validates HTTPS links.
- **Response Bounds**: Enforces maximum length boundaries ($\le 500$ characters).

---

## 9. Risk-Aware Escalation Decision Engine

Computes a composite risk score across upstream signals:
$$\text{Composite Risk} = 0.40 \cdot \text{Risk}_{\text{intent}} + 0.30 \cdot \text{Risk}_{\text{retrieval}} + 0.30 \cdot \text{Risk}_{\text{category}}$$

- **Hard Escalation Triggers**: `SAFETY_CHECK_FAILED`, `INSUFFICIENT_EVIDENCE`, `NO_RETRIEVABLE_EVIDENCE`, `INVALID_RESPONSE`.
- **Operating Point**: `risk_threshold = 0.35`.

---

## 10. Evaluation Methodology

Evaluated across three gold benchmark datasets:
1. **$N=200$ Human Gold Intent Benchmark**: Manually annotated AmazonHelp customer queries across 17 intent classes.
2. **$N=50$ Real Gemini Live Run**: Evaluated live response generation across 7 judge dimensions (Groundedness, Tone, Correctness, Relevance, Helpfulness, Hallucination Risk, Actionability).
3. **$N=50$ Human Escalation Validation Set**: Manually annotated escalation decisions for human–model decision agreement evaluation.

---

## 11. Master Authoritative Benchmark Table

| System Component | Evaluation Dataset | N | Corpus | Protocol | Metric | Result | Label Type | Status | Notes |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **Intent Classification (End-to-End)** | AmazonHelp Human Gold | 200 | N/A | End-to-end SVM + rule classifier inference | Accuracy / Macro F1 | **57.5% / 0.5841** | Human Gold | **Active Default** | Real-world noisy Twitter support threads |
| **Intent Classification (Gold Prototype)** | AmazonHelp Human Gold | 200 | N/A | Centroid prototype trained on 100 gold dev split | Accuracy / Macro F1 | **69.5% / 0.6889** | Human Gold | Diagnostic | Standalone centroid model on full gold set |
| **Intent Classification (Auxiliary)** | Banking77 Test Set | 3,076 | N/A | Clean banking benchmark classification | Accuracy / Macro F1 | **85.0% / 0.8485** | Human Gold | Diagnostic | Clean auxiliary benchmark baseline |
| **Dense Retrieval Baseline** | AmazonHelp Benchmark | 200 | 15,316 | FAISS vector search (`all-MiniLM-L6-v2`) | R@1 / R@10 / MRR | **28.0% / 59.5% / 0.3666** | Proxy (Intent) | Baseline | Stage 1 dense vector search |
| **Intent-Aware Dense Retrieval** | AmazonHelp Benchmark | 200 | 15,316 | FAISS vector search with predicted intent boost | R@1 / R@10 / MRR | **51.5% / 60.0% / 0.5379** | Proxy (Intent) | Baseline | Dense retrieval with intent category filtering |
| **BM25 Lexical Retrieval** | AmazonHelp Benchmark | 200 | 15,316 | Okapi BM25 index over query texts | R@1 / R@10 / MRR | **28.5% / 71.5% / 0.4048** | Proxy (Intent) | Baseline | Stage 1 lexical keyword search |
| **Hybrid RRF Retrieval** | AmazonHelp Benchmark | 200 | 15,316 | Dense + BM25 Reciprocal Rank Fusion ($k=60$) | R@1 / R@10 / MRR | **35.0% / 69.5% / 0.4508** | Proxy (Intent) | Baseline | Candidate fusion without intent boost |
| **Hybrid + Intent (Phase 9 Approved)** | AmazonHelp Benchmark | 200 | 15,316 | Hybrid RRF + predicted intent boost | R@1 / R@3 / R@5 / R@10 / MRR | **53.5% / 55.5% / 58.0% / 65.5% / 0.5590** | Proxy (Intent) | **Approved Default** | Phase 9 optimal retrieval architecture |
| **Hybrid + Reranker + Intent (Phase 10)** | AmazonHelp Benchmark | 200 | 15,316 | Hybrid RRF → Cross-Encoder → Intent Boost | R@1 / R@3 / R@5 / R@10 / MRR | **35.5% / 53.5% / 59.0% / 69.0% / 0.4602** | Proxy (Intent) | Experimental | Phase 10 cross-encoder (disabled by default) |
| **Grounded Response Generation** | AmazonHelp Live Run | 50 | 15,316 | Gemini 1.5 Flash API + 7-dim LLM Judge | Overall Judge / Safety Pass | **4.72 / 5.0 (100% Pass)** | Real API Eval | **Active Default** | Evaluated on 5-pt scale (Groundedness=4.96/5) |
| **Escalation Decision Engine** | Escalation Validation Set | 50 | N/A | Composite risk policy ($\text{threshold}=0.35$) | Agreement / False Auto / False Esc | **74.0% Agreement (12.0% FA / 14.0% FE)** | Human Gold | **Active Default** | 41/50 auto-handled (82%), 9/50 escalated (18%) |

---

## 12. Top Five System Failure Modes

Attributed across $N=50$ human escalation validation disagreement cases ([reports/tables/phase_8_failure_attribution.csv](file:///d:/hiver-support-agent/reports/tables/phase_8_failure_attribution.csv)):

1. **Category Risk Over-Penalty (False Escalation - 5 cases)**: Conservative category risk penalties force escalation on routine queries in sensitive categories (`refund_status_request`, `account_access_login`, `payment_billing_issue`).
2. **Ambiguous Query Under-Escalation (False Auto-Handle - 3 cases)**: Broad inquiries classified as routine inquiries with high confidence despite requiring human review.
3. **Complex Dispute Bypassing Risk (False Auto-Handle - 2 cases)**: Complex promotional EMI disputes auto-handled due to high classifier confidence.
4. **Physical Delivery Anomaly Unchecked (False Auto-Handle - 1 case)**: Delivery anomaly auto-handled as standard delay.
5. **Weak Retrieval Score Over-Escalation (False Escalation - 2 cases)**: Routine inquiries escalated due to top retrieval similarity score falling below 0.70.

---

## 13. What Is Misleading About My Headline Number?

To maintain complete evaluation integrity and transparency, we explicitly detail ten caveats regarding our benchmark numbers:

1. **Retrieval R@1 Uses Intent Matching Proxy Ground Truth**: Benchmark retrieval relevance uses intent category match as a diagnostic proxy for historical precedent relevance, rather than manual pairwise human judgments across all 15,316 documents.
2. **Benchmark Sample Size ($N=200$)**: The human gold intent benchmark consists of 200 manually annotated cases. While sufficient for statistical benchmarking, it represents a sample of e-commerce support queries.
3. **Gemini LLM Judge Sample Size ($N=50$)**: Live API generation and 7-dimension LLM judge scoring were conducted on $N=50$ cases due to API execution costs and quota constraints.
4. **Escalation Validation Dataset & Single Annotator ($N=50$)**: Escalation decision ground truth was annotated on $N=50$ cases by a single human annotator. Inter-annotator agreement (Cohen's Kappa) was not measured.
5. **Retrieval Corpus Scale**: The benchmark corpus consists of 15,316 clean multi-turn resolution threads. The raw Twitter dataset contains 81,186 valid dialog documents, which are preserved in the raw corpus.
6. **Near-Duplicate Leakage Audit**: Self-case matching (by case ID and exact customer query string) was strictly excluded during retrieval scoring. However, subtle semantic near-duplicates across customer tweets may provide minor retrieval lift.
7. **LLM-as-Judge vs. Human Expert Evaluation**: An overall LLM judge score of 4.72/5.0 confirms structural groundedness, tone, and absence of hallucinations, but cannot fully replace human domain expert review of actionability.
8. **Recall@10 vs. Rank-1 Evidence Selection Precision**: Achieving 65.5% Recall@10 (Phase 9) or 69.0% Recall@10 (Phase 10) does not guarantee that the single Rank-1 precedent selected for evidence prompting is the exact optimal resolution.
9. **Pipeline Error Propagation**: Upstream classification errors (57.5% accuracy) and retrieval misses propagate into downstream prompt generation and escalation risk calculations.
10. **Prototype Benchmark Evidence vs. Production SLA**: All metrics reflect controlled prototype evaluation under local CPU conditions and must not be interpreted as production SLA or safety guarantees.

---

## 14. Reproduction Instructions

### 1. Clone & Setup Repository

```bash
git clone https://github.com/your-username/hiver-ai-support-agent.git
cd hiver-ai-support-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment Variables

Copy template and set your API key:

```bash
cp .env.example .env
# Edit .env:
# GEMINI_API_KEY="your-gemini-api-key"
```

### 3. Run Automated Unit Test Suite

```bash
pytest tests/ -v
```
*(Expected output: 78 passed, 1 warning)*

### 4. Run End-to-End Benchmark Evaluation

```bash
python scripts/evaluate_phase8_end_to_end.py
```

### 5. Run Phase 9 Hybrid Retrieval Benchmark

```bash
python scripts/evaluate_phase9_hybrid_retrieval.py
```

---

## 15. System Configuration

Configuration is managed via `configs/config.yaml`:

```yaml
project:
  name: hiver-support-agent
  target_brand: AmazonHelp

generation:
  llm_provider: gemini
  model_name: gemini-1.5-flash
  temperature: 0.2

decision:
  risk_threshold: 0.35
  confidence_threshold: 0.60
  min_retrieval_score: 0.70

retrieval:
  mode: hybrid
  dense_top_k: 50
  bm25_top_k: 50
  rrf_k: 60
  intent_boost_factor: 0.15

reranking:
  enabled: false # Preserved Phase 9 default
  model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
  candidate_k: 30
```

---

## 16. Example Usage & Python CLI Demo

You can run an end-to-end single message inference directly via Python:

```python
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.retrieval.faiss_store import FAISSVectorStore
from hiver_agent.retrieval.retriever import FAISSRetriever
from hiver_agent.generation.generator import LLMResponseGenerator
from hiver_agent.pipeline import CustomerSupportPipeline

# 1. Load retriever index and LLM generator
embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
vector_store = FAISSVectorStore.load("data/processed/faiss_index_15k")
base_retriever = FAISSRetriever(embedder, vector_store)
generator = LLMResponseGenerator(provider="gemini", model_name="gemini-1.5-flash")

# 2. Instantiate pipeline
pipeline = CustomerSupportPipeline(
    base_retriever=base_retriever,
    generator=generator
)

# 3. Execute end-to-end processing
query = "Where is my package? It was supposed to be delivered yesterday. Order ID: 403-1234567-8901234"
result = pipeline.process_end_to_end(query)

print("--- End-to-End Processing Result ---")
print(f"Predicted Intent    : {result.predicted_intent} (Conf: {result.intent_confidence})")
print(f"Retrieval Mode      : {result.retrieval_mode}")
print(f"Top Retrieved Cases : {result.retrieved_case_ids[:3]}")
print(f"Draft Response      : {result.draft_response}")
print(f"Safety Check Passed : {result.safety_check_passed}")
print(f"Risk Score          : {result.risk_score}")
print(f"Decision            : {result.decision} ({result.reason_code})")
print(f"Decision Reason     : {result.decision_reason}")
```

---

## 17. Top Known Limitations

1. **Validation Sample Size ($N=50$)**: Human escalation ground truth is evaluated on 50 validation cases.
2. **Single Annotator Setup**: Inter-annotator agreement (Cohen's Kappa) was not measured due to single-annotator annotation.
3. **Retrieval Proxy Metric**: Retrieval evaluation uses intent alignment as a diagnostic proxy ground truth rather than full manual relevance judgments across 15,316 documents.
4. **Evaluation Prototype Context**: Built for technical take-home evaluation; production deployment requires API gateway, rate-limiting, and OAuth authentication layers.

---

## 18. One-Week Prioritized Roadmap (Next Steps)

- **Day 1 (Annotation & Agreement)**: Expand human escalation labels to 200 cases and collect multi-annotator labels to measure Cohen's Kappa.
- **Day 2 (Classifier Calibration)**: Fine-tune MiniLM embeddings on multi-class contrastive loss to improve separation between neighboring delivery intents.
- **Day 3 (Domain Reranking)**: Fine-tune the MS-MARCO cross-encoder on e-commerce support precedent pairs to eliminate intent-demotion artifacts.
- **Day 4 (Pairwise Relevance Labels)**: Annotate exact document relevance labels for top-10 candidates across 200 benchmark queries to replace intent proxy ground truth.
- **Day 5 (Cost-Sensitive Escalation)**: Implement dynamic risk thresholds sensitive to intent monetary risk (e.g. high-value refunds vs. routine tracking).
- **Day 6 (Observability & Regression Testing)**: Build real-time latency tracing, token usage logging, and automated CI regression testing for prompt updates.
- **Day 7 (Human-in-the-Loop Shadow Deployment)**: Deploy pipeline in shadow mode alongside human agents to collect live feedback and false auto-handle rates.
