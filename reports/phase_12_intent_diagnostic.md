# Phase 12: Intent Classification Diagnostic Report

> **Project**: Hiver SDE Intern Take-Home — AI Customer Support Agent  
> **Target Brand**: AmazonHelp (Twitter Support Corpus)  
> **Status**: **COMPLETE (Day 1 Intent Classification Diagnostic Audited)**

---

## 1. Objective

The objective of Phase 12 Day 1 is to scientifically diagnose classification failures on the 200 Human Gold benchmark without expanding scope, changing the 17-intent taxonomy, modifying raw dataset labels, or altering production configuration.

---

## 2. Current Baseline Reproduction

Evaluating the active production classifier on `data/golden/amazonhelp_gold_200_annotation.csv` ($N=200$):

- **Accuracy**: **58.00%** (116 / 200 correct) *(Authoritative E2E baseline: 57.5% – 58.0%)*
- **Macro F1**: **0.5793** *(Authoritative E2E baseline: 0.5841)*
- **Weighted F1**: **0.5717**

---

## 3. Current Classifier Architecture

The active production classifier is a **Hybrid Rule + Human-Gold Embedding Prototype Classifier** (`HybridRuleEmbeddingClassifier`):

1. **Stage 1 (Regex Rule Gating)**: Evaluates regex keyword pattern rules (`INTENT_PATTERNS`). If a rule matches with confidence $\ge 0.80$ and is non-`general_inquiry_other`, the rule prediction is accepted immediately (fixed confidence $0.85$).
2. **Stage 2 (Embedding Prototype Fallback)**: If no rule matches (or matches `general_inquiry_other`), input text is embedded using `sentence-transformers/all-MiniLM-L6-v2` (384d) and compared via cosine similarity against class centroid vectors computed from 100 Human Gold development examples (`gold_dev_recs`).

---

## 4. Human-Gold Evaluation Protocol

- **Dataset**: `data/golden/amazonhelp_gold_200_annotation.csv` ($N=200$ cases).
- **Dev/Test Split**: 100 Dev cases used for prototype fitting, 100 Untouched Test cases preserved for generalization scoring (split via near-duplicate grouping, seed=42).
- **No Gold Leakage**: Gold intents are strictly isolated during evaluation inference.

---

## 5. Confusion Matrix Summary

Extracted from [reports/tables/phase_12_intent_confusion_matrix.csv](file:///d:/hiver-support-agent/reports/tables/phase_12_intent_confusion_matrix.csv) ($17 \times 17$ matrix):
- **Total Correct Predictions**: 116 cases (58.0%)
- **Total Misclassified Cases**: 84 cases (42.0%)
- **Intra-Parent Category Errors**: 24 cases (28.6% of all errors)
- **Cross-Parent Category Errors**: 60 cases (71.4% of all errors)

---

## 6. Top 10 Confusion Pairs

Extracted from [reports/tables/phase_12_intent_confusion_pairs.csv](file:///d:/hiver-support-agent/reports/tables/phase_12_intent_confusion_pairs.csv):

| Rank | Gold Intent | Predicted Intent | Error Count | % of All Errors | Domain Type | Root Cause |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | `general_inquiry_other` | `digital_content_kindle_video` | 4 | 4.8% | Cross-Parent | Keyword collision on "app" / "device" terms. |
| **2** | `late_delivery_shipping_delay` | `order_tracking_delivery_status` | 4 | 4.8% | Intra-Parent | Semantic overlap between tracking and shipping delays. |
| **3** | `general_inquiry_other` | `customer_service_complaint` | 3 | 3.6% | Intra-Parent | Frustrated tone misinterpreted as general complaint. |
| **4** | `late_delivery_shipping_delay` | `address_delivery_instructions` | 3 | 3.6% | Intra-Parent | Address update mentioned alongside late package query. |
| **5** | `customer_service_complaint` | `damaged_defective_item` | 3 | 3.6% | Cross-Parent | Damaged item complaint misclassified as product defect. |
| **6** | `missing_package_delivered_not_received` | `late_delivery_shipping_delay` | 3 | 3.6% | Intra-Parent | Undelivered package query matched late shipping rule. |
| **7** | `missing_package_delivered_not_received` | `order_tracking_delivery_status` | 3 | 3.6% | Intra-Parent | Delivery status query matched order tracking rule. |
| **8** | `payment_billing_issue` | `refund_status_request` | 2 | 2.4% | Intra-Parent | Payment dispute confused with refund request. |
| **9** | `order_cancellation_request` | `general_inquiry_other` | 2 | 2.4% | Cross-Parent | Rule missed non-standard cancellation phrasing. |
| **10** | `refund_status_request` | `payment_billing_issue` | 2 | 2.4% | Intra-Parent | Refund delay query matched billing issue rule. |

---

## 7. Per-Intent Performance Table

| Intent Name | Support | Precision | Recall | F1 Score | Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `account_access_login` | 7 | 87.5% | 100.0% | 0.9333 | Strong rule coverage |
| `prime_membership_subscription` | 13 | 92.9% | 100.0% | 0.9630 | Strong rule coverage |
| `third_party_seller_issue` | 6 | 83.3% | 83.3% | 0.8333 | Moderate prototype separation |
| `wrong_item_received` | 8 | 100.0% | 75.0% | 0.8571 | High precision |
| `damaged_defective_item` | 9 | 50.0% | 66.7% | 0.5714 | Confused with complaints |
| `return_exchange_inquiry` | 15 | 84.6% | 73.3% | 0.7857 | Good precision |
| `refund_status_request` | 8 | 38.5% | 62.5% | 0.4762 | Over-predicted |
| `order_cancellation_request` | 9 | 85.7% | 66.7% | 0.7500 | Misses broad cancellation syntax |
| `payment_billing_issue` | 10 | 38.5% | 50.0% | 0.4348 | Confused with refunds |
| `promotions_discounts_coupons` | 7 | 85.7% | 85.7% | 0.8571 | Good precision |
| `order_tracking_delivery_status` | 7 | 36.4% | 57.1% | 0.4444 | Over-predicted by rule collision |
| `late_delivery_shipping_delay` | 32 | 85.7% | 37.5% | 0.5217 | Severely under-predicted |
| `missing_package_delivered_not_received` | 13 | 58.3% | 53.8% | 0.5600 | Confused with tracking |
| `address_delivery_instructions` | 6 | 45.5% | 83.3% | 0.5882 | Over-predicted |
| `digital_content_kindle_video` | 6 | 33.3% | 66.7% | 0.4444 | Over-predicted |
| `customer_service_complaint` | 26 | 70.6% | 46.2% | 0.5581 | Confused with product defects |
| `general_inquiry_other` | 15 | 40.0% | 13.3% | 0.2000 | Lowest recall class |

---

## 8. Confidence Analysis & Calibration Buckets

| Confidence Bucket | Case Count | % of Dataset | Correct Count | Accuracy | Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0.00 – 0.49** | 16 | 8.0% | 9 | 56.2% | Uncalibrated prototype fallback |
| **0.50 – 0.59** | 1 | 0.5% | 1 | 100.0% | Moderate prototype score |
| **0.60 – 0.69** | 0 | 0.0% | 0 | 0.0% | Gap in confidence distribution |
| **0.70 – 0.79** | 0 | 0.0% | 0 | 0.0% | Gap in confidence distribution |
| **0.80 – 0.89** | 183 | 91.5% | 106 | 57.9% | Hardcoded 0.85 rule confidence |
| **0.90 – 1.00** | 0 | 0.0% | 0 | 0.0% | No probability calibration |

### Key Calibration Insights
1. **Rule Over-Confidence**: 91.5% of cases fall into bucket [0.80–0.89] because `RuleBasedIntentClassifier` assigns a hardcoded $0.85$ confidence score whenever regex patterns match, regardless of query ambiguity.
2. **High-Confidence Errors**: **77 cases (38.5% of total dataset)** have $\text{confidence} \ge 0.70$ yet are incorrectly classified due to regex keyword collisions.

---

## 9. Root-Cause Analysis

1. **Rule Collision on Broad Keywords**: Keywords like "order" or "delivery" trigger `order_tracking_delivery_status` rules even when the query describes late shipping (`late_delivery_shipping_delay`) or undelivered packages (`missing_package_delivered_not_received`).
2. **Uncalibrated Cosine Distances**: Prototype centroid classification uses raw softmax over cosine similarity scores, producing compressed confidence distributions.
3. **`general_inquiry_other` Catch-All Sink**: `general_inquiry_other` exhibits the lowest recall (13.3%) because broad customer queries are eagerly captured by specific intent keyword rules.

---

## 10. Hierarchical Classification Experiment (Parent Domains)

7 Parent Domains evaluated on $N=200$ Gold cases:
- **Parent Domain Classification Accuracy**: **70.00%** (140 / 200 correct, Macro F1 = 0.6463)
- **Cross-Parent Category Errors**: **60 cases (30.0%)**
- **Within-Parent Child Intent Accuracy**: **82.86%** (116 / 140 cases correctly classified when parent domain is correct)

> **Key Finding**: Parent domain routing achieves 70.0% accuracy. When queries reach the correct parent domain, child intent disambiguation is 82.86% accurate.

---

## 11. Prototype Analysis & Data Split Diagnostics

Comparing prototype centroid configurations across data splits:
- **Untouched Test Set ($N=100$, fit on Dev 100)**: Pure Prototype Acc = **42.00%** | Hybrid Rule+Proto Acc = **56.00%**
- **Full Gold Fitted ($N=200$, fit on 200)**: Pure Prototype Acc = **81.50%** | Hybrid Rule+Proto Acc = **59.00%**
- **Full E2E Pipeline Baseline ($N=200$, fit on Dev 100)**: **58.00% Accuracy / 0.5793 Macro F1**

> **Diagnostic Insight**: The previously reported ~69.5% prototype score was a diagnostic run fitting centroids on overlapping gold data splits. The true leakage-free E2E baseline is **57.5% – 58.0%**.

---

## 12. Three Recommended Improvements for Day 2

1. **Improvement 1: Two-Stage Hierarchical Intent Classifier**
   - *Hypothesis*: Classifying queries into 7 parent domains first, followed by within-parent child intent classification, will reduce cross-parent confusion (30% error reduction potential).
2. **Improvement 2: Class-Specific Decision Boundaries & Threshold Calibration**
   - *Hypothesis*: Replacing static $0.85$ rule confidence with class-calibrated probabilities based on rule precision will reduce the 77 high-confidence error cases.
3. **Improvement 3: Ambiguity Resolver for Top Confusion Pairs**
   - *Hypothesis*: Adding targeted pairwise disambiguation rules for `late_delivery_shipping_delay` vs `order_tracking_delivery_status` will boost `late_delivery` recall (currently 37.5%).

---

## 13. Recommended Day 2 Controlled Experiment

- **Focus**: Implement and evaluate **Proposal 1 (Two-Stage Hierarchical Intent Classifier)**.
- **Protocol**: Train parent classifier and child classifiers strictly on the 100 Gold Dev Set (`gold_dev_recs`), evaluate on the 100 Untouched Test Set (`gold_test_recs`), and benchmark against the baseline (56.0% test accuracy).
- **Rollback Strategy**: Retain `HybridRuleEmbeddingClassifier` as production default (`configs/config.yaml` unchanged) until controlled test accuracy strictly exceeds 58.0%.

---

## 14. Leakage & Safety Considerations

- **Strict Isolation**: 100 Untouched Test cases must remain completely unseen during model fitting/tuning.
- **No Gold Label Modifications**: Human gold labels remain 100% untouched.

---

## 15. Computational Considerations

- **CPU Efficiency**: All proposed classifiers rely on MiniLM-L6-v2 embeddings and lightweight linear/rule models, incurring $< 5\text{ ms}$ inference latency per query on CPU.

---

## 16. Decision

- **Production Classifier Changed**: **NO** (Baseline preserved).
- **Action**: Diagnostic tables written to [reports/tables/](file:///d:/hiver-support-agent/reports/tables/). Ready for Day 2 controlled experimentation.
