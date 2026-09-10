# Hiver SDE Intern Take-Home — Final Evaluation Report: AmazonHelp AI Support Agent

> **Author**: AI Engineering Candidate  
> **Target Dataset**: `AmazonHelp` Twitter Support Corpus  
> **System Status**: **Evaluation-Ready Prototype**  
> **Repository URL**: Canonical Implementation Workspace  

---

## 1. Problem & Approach

Customer support operations face escalating query volumes, leading to high response latency and agent burnout. Automated support agents must balance two conflicting goals:
1. **Automation Coverage**: Resolving routine queries quickly without human agent intervention.
2. **Customer Safety & Precision**: Preventing incorrect, ungrounded, or hallucinatory responses from reaching customers.

We present an end-to-end, RAG-powered AI Customer Support Agent specifically built for `@AmazonHelp`. The pipeline integrates:
- **17-Intent Domain Taxonomy & Hybrid Classifier**: Categorizes customer inquiries using rule-based regex patterns and Linear SVM classifiers with TF-IDF n-grams.
- **Hybrid Vector & Lexical Retrieval**: Searches a 15,316-document RAG corpus of reconstructed AmazonHelp resolution threads using `all-MiniLM-L6-v2` dense vectors, BM25Okapi lexical index, Reciprocal Rank Fusion ($k=60$), and intent-aware metadata reranking.
- **Grounded LLM Response Generation**: Synthesizes customer responses using Google Gemini 1.5 Flash backed by historical evidence.
- **Grounding & Entity Safety Validation**: Validates entity formats, order IDs, URLs, and response lengths deterministically.
- **Risk-Aware Decision Engine**: Computes interpretable risk signals across intent uncertainty, retrieval quality, evidence sufficiency, intent category risk, and safety checks to decide whether to `AUTO_HANDLE` or `ESCALATE_TO_HUMAN`.

---

## 2. Dataset & Intent Taxonomy

- **Dataset**: Filtered and reconstructed from 2.8M tweets in the *Customer Support on Twitter* corpus.
- **Brand Corpus**: 170,000+ tweets directed to `@AmazonHelp` (81,186 valid multi-turn dialog documents).
- **Thread Reconstruction**: Built a graph-based thread assembly engine generating multi-turn dialog pairs (`min_depth=2`, `max_depth=10`), yielding 15,316 clean customer problem-resolution pairs.
- **17-Intent Taxonomy**: Formulated specifically for e-commerce support:
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

## 3. Retrieval + Historical Grounding

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, batch size 128).
- **Index**: Flat L2 FAISS Vector Index over 15,316 resolution documents, supplemented by Okapi BM25 lexical index.
- **Hybrid RRF + Intent-Aware Reranking**: Merges dense semantic ranks and BM25 lexical ranks via Reciprocal Rank Fusion ($k=60$). When classifier confidence $\ge 0.30$, candidates matching predicted intent receive an intent boost multiplier ($1.5\times$).
- **Approved Default Retrieval Performance** (Phase 9 Approved: $N=200$ Gold Cases vs 15k RAG Corpus):
  - **Recall@1**: **53.5%**
  - **Recall@3**: **55.5%**
  - **Recall@5**: **58.0%**
  - **Recall@10**: **65.5%**
  - **MRR**: **0.5590**
- **Experimental Reranker Note**: Phase 10 evaluated adding an MS-MARCO Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-order top 30 candidates before intent boosting. Reranking improved Recall@10 (65.5% → 69.0%) but severely degraded Recall@1 (53.5% → 35.5%) and MRR (0.5590 → 0.4602) due to demoting intent-aligned precedents. Consequently, Phase 9 remains the approved default system architecture (`reranking.enabled: false`).

---

## 4. Response Generation & Grounding Safety

- **LLM Provider**: Google Gemini 1.5 Flash API (`gemini-1.5-flash`).
- **Prompt Architecture**: System instructions enforce strict grounding in retrieved evidence, prohibiting assumptions about unprovided order numbers or account details.
- **Safety Checker**: Deterministic post-generation validation checks order ID regex, URL syntax, and response bounds.
- **Live Gemini Benchmark Results** ($N=50$ Live Calls):
  - **Response Validity Rate**: **100.0%**
  - **Grounding & Safety Check Pass Rate**: **100.0%**
  - **LLM-as-Judge Overall Score**: **4.72 / 5.0** (Relevance: 4.68, Helpfulness: 4.62, Correctness: 4.74, Groundedness: 4.96, Hallucination Risk: 4.92, Tone: 4.88, Actionability: 4.66)
  - **Average Generation Latency**: **1.85 seconds**

---

## 5. Auto-Handle vs Escalation Layer

The decision engine computes a composite risk score:
$$\text{Composite Risk} = 0.40 \cdot \text{Risk}_{\text{intent}} + 0.30 \cdot \text{Risk}_{\text{retrieval}} + 0.30 \cdot \text{Risk}_{\text{category}}$$

- **Hard Escalation Rules**: `SAFETY_CHECK_FAILED`, `INSUFFICIENT_EVIDENCE`, `NO_RETRIEVABLE_EVIDENCE`, `INVALID_RESPONSE`.
- **Operating Point**: `risk_threshold = 0.35`.

---

## 6. Authoritative Benchmark Evaluation Table

| Evaluation Benchmark | Metric | Value | Ground Truth / Setup | Label Type | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intent Classification (End-to-End)** | Accuracy / Macro F1 | **57.5% / 0.5841** | 200 Human Gold Benchmark Cases | Human Gold | Active Default |
| **Intent Classification (Gold Prototype)** | Accuracy / Macro F1 | **69.5% / 0.6889** | 200 Gold Cases (100 dev trained) | Human Gold | Diagnostic |
| **Intent Classification (Auxiliary)** | Accuracy / Macro F1 | **85.0% / 0.8485** | Banking77 Test Set (3,076 rows) | Human Gold | Diagnostic |
| **Dense Retrieval Baseline** | R@1 / R@10 / MRR | **28.0% / 59.5% / 0.3666** | 200 Gold Cases vs 15k RAG Corpus | Proxy (Intent) | Baseline |
| **Intent-Aware Dense Retrieval** | R@1 / R@10 / MRR | **51.5% / 60.0% / 0.5379** | 200 Gold Cases vs 15k RAG Corpus | Proxy (Intent) | Baseline |
| **BM25 Lexical Retrieval** | R@1 / R@10 / MRR | **28.5% / 71.5% / 0.4048** | 200 Gold Cases vs 15k RAG Corpus | Proxy (Intent) | Baseline |
| **Hybrid RRF Retrieval** | R@1 / R@10 / MRR | **35.0% / 69.5% / 0.4508** | 200 Gold Cases vs 15k RAG Corpus | Proxy (Intent) | Baseline |
| **Hybrid + Intent (Phase 9 Approved)** | R@1 / R@5 / MRR | **53.5% / 58.0% / 0.5590** | 200 Gold Cases vs 15k RAG Corpus | Proxy (Intent) | **Approved Default** |
| **Hybrid + Reranker + Intent (Phase 10)** | R@1 / R@5 / MRR | **35.5% / 59.0% / 0.4602** | 200 Gold Cases vs 15k RAG Corpus | Proxy (Intent) | Experimental |
| **Grounded Response Generation** | LLM Judge / Safety Pass | **4.72 / 5.0 (100% Pass)** | Gemini 1.5 Flash Live Run ($N=50$) | Real API Eval | Active Default |
| **Human Escalation Validation** | Agreement / False Auto | **74.0% Agreement (12.0% FA)** | 50 Human Escalation Validation Set | Human Gold | Active Default |

---

## 7. Failure Analysis

1. **Category Risk Over-Penalty (False Escalation)**: Conservative category risk penalties force escalation on routine queries in sensitive categories (`refund_status_request`, `account_access_login`).
2. **Ambiguous Query Under-Escalation (False Auto-Handle)**: Broad inquiries classified as routine inquiries with high confidence despite requiring human review.
3. **Complex Dispute Bypassing Risk (False Auto-Handle)**: Complex promotional EMI disputes auto-handled due to high classifier confidence.
4. **Physical Delivery Anomaly Unchecked (False Auto-Handle)**: Delivery anomaly auto-handled as standard delay.
5. **Weak Retrieval Score (False Escalation)**: Routine inquiries escalated due to top retrieval similarity score falling below 0.70.

---

## 8. Limitations

1. **Validation Sample Size**: Human escalation validation is evaluated on 50 manually annotated cases.
2. **Single Annotator**: Single-annotator setup; inter-annotator agreement (Cohen's Kappa) was not measured.
3. **Proxy Metric Distinction**: Retrieval relevance uses intent alignment as a proxy ground truth.

---

## 9. What Is Misleading About My Headline Number?

It is easy to misinterpret headline numbers in AI support agent evaluations. We explicitly clarify:

1. **82.0% Model Auto-Handle Rate $\neq$ 82.0% Safe Automation Rate**: On 50 human escalation validation cases, the model auto-handles 82.0% of queries with a **12.0% false auto-handle rate** (14.6% among auto-handled queries) and a **14.0% false escalation rate** (77.8% among escalated queries).
2. **LLM Judge Score (4.72/5.0) $\neq$ Production Readiness**: High LLM judge scores confirm that generated responses are coherent, polite, and grounded in retrieved evidence snippets, but do not guarantee that the historical snippet contains the exact policy resolution for a specific customer account.
3. **Intent Accuracy (57.5%) Reflects Real-World Twitter Noise**: Twitter support messages are brief, noisy, and unstructured. While 57.5% accuracy appears modest, it represents realistic multi-class performance on a 17-class taxonomy evaluated on human gold labels.
4. **Retrieval R@1 Uses Intent Alignment Proxy**: Benchmark retrieval metrics use intent category matching as a diagnostic proxy for historical precedent relevance, rather than manual pairwise human judgments across all 15,316 documents.
5. **Benchmark Sample Size ($N=200$)**: The human gold intent benchmark consists of 200 manually annotated cases. While sufficient for statistical benchmarking, it represents a sample of e-commerce support queries.
6. **Gemini LLM Judge Sample Size ($N=50$)**: Live API generation and 7-dimension LLM judge scoring were conducted on $N=50$ cases due to API execution costs and quota constraints.
7. **Escalation Validation Dataset & Single Annotator ($N=50$)**: Escalation decision ground truth was annotated on $N=50$ cases by a single human annotator. Inter-annotator agreement (Cohen's Kappa) was not measured.
8. **Retrieval Corpus Scale**: The benchmark corpus consists of 15,316 clean multi-turn resolution threads, whereas the canonical raw Twitter corpus contains 81,186 valid dialog documents.
9. **Near-Duplicate Leakage Audit**: Self-case matching (by case ID and exact customer query string) was strictly excluded during retrieval scoring. However, subtle semantic near-duplicates across customer tweets may provide minor retrieval lift.
10. **Pipeline Error Propagation**: Upstream classification errors (57.5% accuracy) and retrieval misses propagate into downstream prompt generation and escalation risk calculations. All metrics reflect controlled prototype evaluation under local CPU conditions and must not be interpreted as production SLA or safety guarantees.

---

## 10. One-Week Next Steps

- **Days 1–2 (Classifier Calibration)**: Fine-tune MiniLM embeddings on AmazonHelp multi-class contrastive loss to improve separation between neighboring delivery intents.
- **Days 3–4 (Dynamic Category Risk)**: Replace static category risk weights with dynamic query-complexity risk scoring to eliminate false escalations on routine refund queries.
- **Day 5 (Escalation Policy Calibration)**: Calibrate risk weights on a secondary 100-case development split.
- **Day 6 (Expanded Human Annotation)**: Collect multi-annotator labels for 200 cases to measure inter-annotator agreement (Cohen's Kappa).
- **Day 7 (Production Hardening & Monitoring)**: Implement API rate-limiting, token usage metering, and real-time latency tracing.
