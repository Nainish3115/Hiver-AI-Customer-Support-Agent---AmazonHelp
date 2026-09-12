# Final Submission Evaluation Report

**Project**: Hiver AI Customer Support Agent  
**Brand**: AmazonHelp  
**Phase**: Day 7 — Final Integration, Regression, and Submission Validation  
**Date**: September 11, 2026  

---

## 1. Project Objective
The objective of this project is to build, evaluate, and package a production-grade, risk-aware customer support agent tailored for AmazonHelp customer inquiries. The agent classifies intent across a 17-class taxonomy, retrieves relevant historical cases via hybrid dense-sparse search, selects evidence, generates safe and grounded responses using Gemini 1.5 Flash, and enforces a risk-aware escalation policy to decide between automated response handling (`AUTO_HANDLE`) and human escalation (`ESCALATE_TO_HUMAN`).

---

## 2. Target Production Architecture

```
Customer Message
       │
       ▼
CalibratedRulePrototypeHybrid (Intent Classifier)
       │
       ▼
Hybrid Dense (all-MiniLM-L6-v2) + BM25 Retrieval
       │
       ▼
Reciprocal Rank Fusion (RRF k=60) + Intent-Aware Compatibility Ranking
       │
       ▼
Evidence Selection & Formatting
       │
       ▼
Gemini 1.5 Flash (Response Generator)
       │
       ▼
Grounding & Entity Safety Checker
       │
       ▼
RiskAwarePolicy (Threshold = 0.35)
       │
       ▼
Decision: [ AUTO-HANDLE  │  ESCALATE-TO-HUMAN ]
```

---

## 3. Active Production Components vs. Rejected Experiments

### Active Production Components
1. **Intent Classifier**: `CalibratedRulePrototypeHybrid` (Day 4 Candidate, integrated with `intent_classifier_mode` switch).
2. **Retrieval Engine**: `Hybrid + Intent` (Dense `all-MiniLM-L6-v2` + `rank_bm25` fused via RRF `k=60` with intent compatibility boost).
3. **Generator Engine**: `Gemini 1.5 Flash` with strict system prompt and temperature `0.2`.
4. **Safety Engine**: `GroundingSafetyChecker` verifying entity consistency, order ID preservation, and hallucination flags.
5. **Decision Engine**: `RiskAwareDecisionEngine` applying `RiskAwarePolicy` with risk threshold `0.35`.

### Rejected Experimental Components (Disabled in Production Path)
1. **Hierarchical Intent Classifier (Day 2)**:
   - *Result*: Accuracy collapsed to **26.00%** on untouched Gold Test (Macro F1 = 0.1391).
   - *Reason*: Stage-1 parent classifier collapsed 76% of predictions into Delivery/Order due to Dev set class imbalance, causing cascading Stage-2 errors.
2. **MS-MARCO Cross-Encoder Reranker (Phase 10)**:
   - *Result*: Top-1 Recall dropped to **35.5%** (vs 53.5% baseline).
   - *Reason*: Domain mismatch between general MS-MARCO passages and concise Twitter customer support dialogues.
3. **Lightweight SentenceTransformer Reranker (Day 6)**:
   - *Result*: Top-1 Recall degraded to **47.0%** (vs 52.5% reproduced baseline), MRR dropped from 0.5570 to 0.5398.
   - *Reason*: Direct dense dot-product rescoring distorted calibrated RRF rank scores.

---

## 4. End-to-End Intent Classification Evaluation

Evaluation on the authoritative **200 AmazonHelp Human-Gold Dataset** split into **100 Gold Dev** and **100 Untouched Gold Test** (deterministic seed 42):

| Evaluation Set | Classifier Candidate | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | High-Conf Errors (≥ 0.70) |
|---|---|---|---|---|---|---|---|
| **Untouched Gold Test (N=100)** | Baseline Hybrid (Day 0) | 56.00% | 0.5328 | 0.5850 | 0.5315 | 0.5595 | 38 |
| **Untouched Gold Test (N=100)** | **CalibratedRulePrototypeHybrid (Day 4)** | **57.00%** | **0.5478** | **0.5889** | **0.5393** | **0.5667** | **13** (65.8% reduction) |
| **Gold Dev Set (N=100)** | CalibratedRulePrototypeHybrid (Day 4) | 81.00% | 0.8163 | 0.8935 | 0.8255 | 0.8158 | 10 |
| **Full Gold Dataset (N=200)** | CalibratedRulePrototypeHybrid (Day 4) | 69.00% | 0.6846 | 0.7476 | 0.6930 | 0.6896 | 23 |

### Key Improvements
- **High-Confidence Errors**: Reduced from **38** to **13** (65.8% reduction), dramatically improving confidence calibration.
- **Specific Intent Recalls**:
  - `customer_service_complaint`: 54.5% → **63.6%**
  - `missing_package_delivered_not_received`: 37.5% → **50.0%**
  - `digital_content_kindle_prime_video`: 50.0% → **75.0%**
- **Inference Latency**: **7.36 ms/query** on CPU.

---

## 5. End-to-End Escalation Policy Evaluation

Evaluated against the **50-Case Human Escalation Validation Dataset** under the unchanged **RiskAwarePolicy (Risk Threshold = 0.35)**:

| Metric | Baseline Hybrid Classifier | Day 4 Integrated Candidate | Delta / Impact |
|---|---|---|---|
| **Human Label Agreement** | 74.0% (37/50) | **74.0% (37/50)** | Unchanged overall agreement |
| **False Auto-Handle (Overall)** | 12.0% (6/50) | **6.0% (3/50)** | **-50.0% relative reduction** |
| **False Auto-Handle (In Auto Group)** | 14.63% (6/41) | **8.57% (3/35)** | **-41.4% relative reduction** |
| **False Escalation (Overall)** | 14.0% (7/50) | **20.0% (10/50)** | +6.0% safety shift |
| **Auto-Handle Rate** | 82.0% (41/50) | **70.0% (35/50)** | More conservative gating |
| **Escalation Rate** | 18.0% (9/50) | **30.0% (15/50)** | Increased human oversight |

*Note*: The substitution of `CalibratedRulePrototypeHybrid` creates a safer, more conservative escalation posture: false auto-handle errors (the highest-risk failure mode) are halved from 12% to 6%, while overall human agreement remains stable at 74%.

---

## 6. Retrieval System Benchmark

Evaluated on **N=200 Human Gold Queries** against the **15,316 Canonical Document Index**:

| Retrieval Configuration | R@1 | R@3 | R@5 | R@10 | MRR | Status |
|---|---|---|---|---|---|---|
| **Dense Only (`all-MiniLM-L6-v2`)** | 28.0% | 40.5% | 49.0% | 59.5% | 0.3666 | Diagnostic |
| **BM25 Sparse Only** | 28.5% | 46.0% | 55.5% | 71.5% | 0.4012 | Diagnostic |
| **Phase 9 Benchmark: Hybrid + Intent** | 53.5% | 55.5% | 58.0% | 65.5% | 0.5590 | Baseline |
| **Day 6 Reproduced: Hybrid + Intent** | **52.5%** | **56.0%** | **61.0%** | **65.0%** | **0.5570** | **Production Active** |
| Config D Lightweight Reranker | 47.0% | 60.0% | 63.5% | 67.5% | 0.5398 | **REJECTED** |
| MS-MARCO Cross-Encoder | 35.5% | 48.0% | 52.0% | 61.0% | 0.4120 | **REJECTED** |

---

## 7. Response Generation & Safety

- **LLM Engine**: Gemini 1.5 Flash (`temperature = 0.2`, `max_tokens = 500`).
- **Phase 6.5 Benchmark Results (N=50)**:
  - **Overall LLM-as-Judge Quality**: **4.72 / 5.00**
  - **Safety Pass Rate**: **100.0%**
  - **Grounding Pass Rate**: **98.0%**
  - **Average Latency**: **1.14s** per response.

---

## 8. Metric Reconciliation Table

| Metric / Claim | Historical Metric | Authoritative Final Metric | Classification | Context & Explanation |
|---|---|---|---|---|
| Weak-Label Intent Classifier | 86.9% | N/A | **Historical / Non-comparable** | Early Phase 5 TF-IDF model trained on noisy weak labels. Not evaluated on human gold. |
| Standalone Prototype Claim | 69.5% | 69.00% | **Diagnostic** | Evaluated on full 200 Gold dataset without train/test split. |
| Gold Dev Prototype Fit | 81.5% | 81.00% | **Diagnostic (Dev)** | Training set fit on 100 Gold Dev examples. |
| Untouched Test Intent Accuracy | 56.00% | **57.00%** | **Authoritative Final** | Tested strictly on 100 untouched Gold Test examples (seed 42). |
| Hybrid Retrieval Top-1 Recall | 53.5% (P9) | **52.5% (Day 6/7)** | **Reproduced Active** | Minor 1% delta due to top_k candidate pool parameter settings; both protocols documented. |
| Gemini Model | Gemini Flash | **Gemini 1.5 Flash** | **Authoritative Active** | Canonical model designation used across production pipeline. |
| Gemini Judge Score | 4.72 / 5 | **4.72 / 5** | **Authoritative Final** | Established in Phase 6.5 benchmark evaluation. |
| Corpus Size | 81,186 | **15,316** | **Authoritative Active** | 81,186 raw Twitter turns filtered into 15,316 unique customer problem documents. |
| Test Suite Count | 78 tests | **122 tests** | **Authoritative Final** | Suite expanded with Phase 12-15 integration and regression unit tests. |

---

## 9. Security & Repository Hygiene Audit

- **Environment File Ignored**: `git check-ignore -v .env` returns `PASS`.
- **API Key Security**: Zero API keys tracked in repository source code or logs.
- **Example Environment Config**: `.env.example` contains clean placeholder (`GEMINI_API_KEY=`).
- **Raw/Large Files**: Raw Twitter CSVs and large vector indexes excluded from version control.
- **Test Suite Execution**: **122 passed, 0 failed, 0 skipped** (`pytest tests/ -q`).

---

## 10. Top 5 Final System Limitations

1. **Taxonomy Edge Ambiguity**: Overlap between `general_inquiry_other` and specific intents limits peak untouched-test intent accuracy to ~57%.
2. **Gold Test Sample Size**: Untouched test set is constrained to N=100 examples due to manual annotation costs.
3. **Over-Escalation on Low Confidence**: High-risk fallback policy causes a 20% false escalation rate on ambiguous queries.
4. **Retrieval Proxy Metric**: Retrieval benchmark evaluates against intent-aligned proxy cases rather than full manual relevance judgments for all 15k documents.
5. **Cold-Start Latency**: Initial load of SentenceTransformer embeddings adds ~2.1 seconds on first application invocation.

---

## 11. Final Recommendation

**Status**: **SUBMISSION READY**

The Hiver AI Customer Support Agent codebase is fully integrated, regression-tested, secure, and ready for final submission.
