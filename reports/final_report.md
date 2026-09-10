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
- **17-Intent Domain Taxonomy & Hybrid Classifier**: Categorizes customer inquiries using rule-based regex patterns and MiniLM embedding prototypes.
- **Intent-Aware Historical Retrieval**: Searches a 15,316-document RAG corpus of reconstructed AmazonHelp resolution threads using `all-MiniLM-L6-v2` and FAISS.
- **Grounded LLM Response Generation**: Synthesizes customer responses using Google Gemini 1.5 Flash backed by historical evidence.
- **Grounding & Entity Safety Validation**: Validates entity formats, order IDs, URLs, and response lengths deterministically.
- **Risk-Aware Decision Engine**: Computes interpretable risk signals across intent uncertainty, retrieval quality, evidence sufficiency, intent category risk, and safety checks to decide whether to `AUTO_HANDLE` or `ESCALATE_TO_HUMAN`.

---

## 2. Dataset & Intent Taxonomy

- **Dataset**: Filtered and reconstructed from 2.8M tweets in the *Customer Support on Twitter* corpus.
- **Brand Corpus**: 170,000+ tweets directed to `@AmazonHelp`.
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
- **Index**: Flat L2 FAISS Vector Index over 15,316 resolution documents.
- **Intent-Aware Reranking**: When intent classifier confidence $\ge 0.60$, search space is filtered by intent category before semantic ranking, with fallback to pure semantic search otherwise.
- **Retrieval Performance** ($N=200$ Gold Cases vs 15k RAG Corpus):
  - **Recall@1**: **49.0%**
  - **Recall@3**: **73.5%**
  - **Recall@5**: **81.5%**
  - **Recall@10**: **89.5%**
  - **MRR**: **0.6384**

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

## 6. Evaluation Summary

| Evaluation Benchmark | Metric | Value | Benchmark Description |
| :--- | :--- | :--- | :--- |
| **Intent Classification** | Accuracy | **57.5%** | 200 Human Gold Intent Cases |
| | Macro F1 | **0.5841** | 200 Human Gold Intent Cases |
| | Weighted F1 | **0.5841** | 200 Human Gold Intent Cases |
| **Retrieval Engine** | Recall@3 | **73.5%** | 200 Gold Cases vs 15k Index |
| | MRR | **0.6384** | 200 Gold Cases vs 15k Index |
| **Response Generation** | LLM Judge | **4.72 / 5.0** | Gemini 1.5 Flash Live Run ($N=50$) |
| | Grounding Pass | **100.0%** | Gemini 1.5 Flash Live Run ($N=50$) |
| **Human Escalation Set** | Model–Human Decision Agreement | **74.0%** | 50 Human Escalation Validation Set |
| | Model Auto-Handle Rate | **82.0%** | 50 Human Escalation Validation Set |
| | False Auto-Handle Rate | **12.0%** (14.6% in Auto) | 50 Human Escalation Validation Set |
| | False Escalation Rate | **14.0%** (77.8% in Esc) | 50 Human Escalation Validation Set |

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

1. **84.5% Model Routing Rate $\neq$ 84.5% Safe Automation Rate**: The 84.5% figure from the 200-case proxy run measures model decision outputs against diagnostic proxy rules, **NOT** independent human-verified safe automation. On human escalation validation ($N=50$), the model auto-handles 82.0% of queries with a **12.0% false auto-handle rate**.
2. **LLM Judge Score (4.67/5.0) $\neq$ Production Readiness**: High LLM judge scores confirm that generated responses are coherent and grounded in retrieved snippets, but do not guarantee that the underlying historical retrieval snippet is the correct policy solution for a customer's specific account.
3. **Intent Accuracy (57.5%) Reflects Noisy Real-World Twitter Data**: Twitter support messages are brief and noisy. While 57.5% accuracy appears modest, it represents realistic performance on a human-annotated 17-class taxonomy.

---

## 10. One-Week Next Steps

- **Days 1–2 (Classifier Calibration)**: Fine-tune MiniLM embeddings on AmazonHelp multi-class contrastive loss to improve separation between neighboring delivery intents.
- **Days 3–4 (Dynamic Category Risk)**: Replace static category risk weights with dynamic query-complexity risk scoring to eliminate false escalations on routine refund queries.
- **Day 5 (Escalation Policy Calibration)**: Calibrate risk weights on a secondary 100-case development split.
- **Day 6 (Expanded Human Annotation)**: Collect multi-annotator labels for 200 cases to measure inter-annotator agreement (Cohen's Kappa).
- **Day 7 (Production Hardening & Monitoring)**: Implement API rate-limiting, token usage metering, and real-time latency tracing.
