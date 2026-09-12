# Phase 12 — Two-Stage Hierarchical Intent Classifier Experiment Report

## 1. Objective
The objective of Phase 12 is to design, implement, and rigorously evaluate a **Two-Stage Hierarchical Intent Classifier** (`HierarchicalIntentClassifier`) on the authoritative 200-case AmazonHelp human-gold dataset (`data/golden/amazonhelp_gold_200_annotation.csv`), benchmarked on 100 untouched Gold Test examples ($N=100$, seed 42). The primary goal is to evaluate whether partitioning the 17-intent taxonomy into 7 coarse parent domains reduces confusion and improves intent classification accuracy and macro F1 over the single-stage baseline.

## 2. Existing Baseline
The current authoritative production baseline is the **Hybrid Rule + Embedding Prototype Classifier** (`HybridRuleEmbeddingClassifier`). On the 100 untouched Gold Test examples (seed 42), the baseline performance is:

- **Accuracy**: 56.00%
- **Macro F1**: 0.5315
- **Weighted F1**: 0.5595

Day 1 diagnostics identified key failure modes in the baseline:
- `general_inquiry_other`: Recall = 12.5% (F1 = 0.1667)
- `late_delivery_shipping_delay`: Recall = 46.7% (F1 = 0.5833)
- `customer_service_complaint`: Recall = 54.5% (F1 = 0.6000)
- `payment_billing_issue`: Recall = 0.0% (F1 = 0.0000)
- `missing_package_delivered_not_received`: Recall = 37.5% (F1 = 0.4615)

## 3. Hierarchical Architecture
The `HierarchicalIntentClassifier` operates as a two-stage sequential system:

```
Customer Message (text)
       │
       ▼
Stage 1: Parent-Domain Classifier (TF-IDF + Logistic Regression, C=1.0)
       │
       ▼
Predicted Parent Domain (1 of 7 coarse domains)
       │
       ▼
Stage 2: Child-Intent Classifier (TF-IDF + Logistic Regression per parent)
       │
       ▼
Predicted Child Intent (1 of 17 authoritative intents)
```

Interface compatibility: Implements `IntentClassifier` interface, returning `IntentPrediction` containing predicted intent, combined confidence ($P(\text{parent}) \times P(\text{child}|\text{parent})$), and full 17-intent global normalized probabilities.

## 4. Parent Taxonomy
The 17 authoritative intents map into 7 parent domains:

| Parent Domain | Child Intents | Dev Count ($N=100$) |
|---|---|---|
| **Delivery / Order** | `order_tracking_delivery_status`, `late_delivery_shipping_delay`, `missing_package_delivered_not_received`, `order_cancellation_request`, `address_delivery_instructions` | 61 |
| **Returns / Product** | `return_exchange_inquiry`, `damaged_defective_item`, `wrong_item_received` | 13 |
| **Financial** | `refund_status_request`, `payment_billing_issue`, `promotions_discounts_coupons` | 10 |
| **Account / Membership** | `account_access_login`, `prime_membership_subscription` | 5 |
| **Content** | `digital_content_kindle_video` | 2 |
| **Seller** | `third_party_seller_issue` | 3 |
| **General / Complaint** | `customer_service_complaint`, `general_inquiry_other` | 6 |

## 5. Training Protocol
- **Training Set**: Strictly limited to 100 Gold Dev examples (`gold_dev_recs`).
- **Feature Extraction**: Word n-grams (1, 2) + Character WB n-grams (3, 5) TF-IDF vectors (max 3,000 features each).
- **Stage 1**: Single LogisticRegression classifier trained to classify texts into 7 parent domains.
- **Stage 2**: Separate LogisticRegression classifiers trained per parent domain over children of that domain using the subset of Dev examples matching that parent domain.

## 6. Leakage Controls
- **Zero Test Set Contamination**: Gold Test set (100 cases) was used strictly for evaluation; zero test examples were used during feature extraction, TF-IDF fitting, or model fitting.
- **No External/Synthetic Data**: No Banking77 labels, weak labels, synthetic examples, Gemini outputs, or retrieval scores were used in training.
- **Deterministic**: Random seed fixed to 42.

## 7. Parent Classifier Results (Stage 1)
Evaluated on 100 untouched Gold Test examples:

- **Parent Domain Accuracy**: 49.00% (49/100 correct parent domain calls)
- **Parent Domain Macro F1**: 0.2248
- **Parent Domain Weighted F1**: 0.4438

*Root Cause*: Severe data sparsity in the 100 Gold Dev dataset for non-delivery domains (`Account_Membership`: 5 dev cases, `Content`: 2 dev cases, `Seller`: 3 dev cases, `General_Complaint`: 6 dev cases). As a result, the Stage 1 model heavily overpredicted `Delivery_Order` (61/100 predictions).

## 8. Child Classifier Results (Stage 2 Conditional)
Conditional evaluation on test examples where Stage 1 correctly predicted the true parent domain ($N=49$):

- **Conditional Child Accuracy**: **53.06%** (26/49 correct child intent predictions given correct parent domain)

When Stage 1 routed correctly to `Delivery_Order` or `Returns_Product`, Stage 2 succeeded in resolving the specific child intent 53.06% of the time.

## 9. End-to-End Results
Full two-stage pipeline on 100 untouched Gold Test cases:

- **Accuracy**: **26.00%** (26/100)
- **Macro F1**: **0.1391**
- **Weighted F1**: **0.1693**

## 10. Per-Intent Results

| Intent | Test Support | Baseline Rec. | Baseline F1 | Hier. Rec. | Hier. F1 | Recall Delta |
|---|---|---|---|---|---|---|
| `order_tracking_delivery_status` | 3 | 33.3% | 0.2500 | 0.0% | 0.0000 | -33.3% |
| `late_delivery_shipping_delay` | 15 | 46.7% | 0.5833 | 100.0% | 0.3297 | +53.3% |
| `missing_package_delivered_not_received` | 8 | 37.5% | 0.4615 | 0.0% | 0.0000 | -37.5% |
| `refund_status_request` | 3 | 66.7% | 0.3636 | 66.7% | 0.8000 | 0.0% |
| `return_exchange_inquiry` | 6 | 83.3% | 0.7143 | 0.0% | 0.0000 | -83.3% |
| `order_cancellation_request` | 4 | 100.0% | 0.7273 | 0.0% | 0.0000 | -100.0% |
| `damaged_defective_item` | 6 | 66.7% | 0.7273 | 0.0% | 0.0000 | -66.7% |
| `wrong_item_received` | 6 | 66.7% | 0.8000 | 66.7% | 0.8000 | 0.0% |
| `payment_billing_issue` | 3 | 0.0% | 0.0000 | 0.0% | 0.0000 | 0.0% |
| `account_access_login` | 4 | 100.0% | 0.8000 | 0.0% | 0.0000 | -100.0% |
| `prime_membership_subscription` | 5 | 60.0% | 0.6000 | 0.0% | 0.0000 | -60.0% |
| `digital_content_kindle_video` | 4 | 50.0% | 0.5000 | 0.0% | 0.0000 | -50.0% |
| `third_party_seller_issue` | 3 | 100.0% | 0.7500 | 0.0% | 0.0000 | -100.0% |
| `promotions_discounts_coupons` | 9 | 66.7% | 0.7059 | 0.0% | 0.0000 | -66.7% |
| `address_delivery_instructions` | 2 | 50.0% | 0.2857 | 0.0% | 0.0000 | -50.0% |
| `customer_service_complaint` | 11 | 54.5% | 0.6000 | 45.5% | 0.4348 | -9.1% |
| `general_inquiry_other` | 8 | 12.5% | 0.1667 | 0.0% | 0.0000 | -12.5% |

## 11. Confusion Matrix
The Stage 1 parent domain misclassifications propagate error directly to Stage 2. Top confusion patterns:
1. All 5 non-delivery delivery intents (`missing_package`, `order_tracking`, `order_cancellation`, etc.) collapsed into `late_delivery_shipping_delay` (76 total predictions for `late_delivery_shipping_delay`).
2. `general_inquiry_other` (8 test cases) $\rightarrow$ 5 routed to `late_delivery_shipping_delay`, 3 to `customer_service_complaint`.
3. `return_exchange_inquiry` (6 test cases) $\rightarrow$ 5 routed to `late_delivery_shipping_delay`, 1 to `damaged_defective_item`.
4. `account_access_login` (4 test cases) $\rightarrow$ 4 routed to `late_delivery_shipping_delay`.
5. `prime_membership_subscription` (5 test cases) $\rightarrow$ 5 routed to `late_delivery_shipping_delay`.

## 12. Known Failure-Mode Comparison

| Intent | Gold Test Support | Baseline Recall | Hierarchical Recall | Status |
|---|---|---|---|---|
| `general_inquiry_other` | 8 | 12.5% | **0.0%** | Worsened |
| `late_delivery_shipping_delay` | 15 | 46.7% | **100.0%** | Overfitted/Collapses all delivery queries |
| `customer_service_complaint` | 11 | 54.5% | **45.5%** | Worsened |
| `payment_billing_issue` | 3 | 0.0% | **0.0%** | Unchanged |
| `missing_package_delivered_not_received` | 8 | 37.5% | **0.0%** | Worsened |

Inspect specific pairs:
- `late_delivery_shipping_delay` vs `order_tracking_delivery_status`: Hierarchical collapsed 100% of order tracking and missing package cases into late delivery.
- `customer_service_complaint` vs `damaged_defective_item`: Stage 1 failed to separate complaint domain from product domain cleanly.
- `general_inquiry_other` vs `digital_content_kindle_video`: Stage 1 failed completely on digital content (0% recall).

## 13. Confidence Methodology
Confidence is computed transparently without hardcoding arbitrary rule numbers:
$$\text{Confidence}_{\text{hierarchical}} = P(\text{Parent Domain}) \times P(\text{Child Intent} \mid \text{Parent Domain})$$
 probabilities are derived directly from the calibrated output of stage 1 and stage 2 Logistic Regression estimators.

## 14. Baseline Comparison

| Method | Dataset | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| **Hybrid Rule+Proto Baseline (Current Pipeline)** | Untouched Gold Test ($N=100$) | **56.00%** | **0.5315** | **0.5595** |
| **Flat TF-IDF Baseline** | Untouched Gold Test ($N=100$) | 24.00% | 0.0801 | 0.1292 |
| **Two-Stage Hierarchical Intent Classifier** | Untouched Gold Test ($N=100$) | 26.00% | 0.1391 | 0.1693 |

## 15. Error Analysis
The dramatic drop in accuracy (56.0% down to 26.0%) stems from **Cascading Multi-Stage Errors** and **Training Support Imbalance**:
1. **Dominant Class Bias in Stage 1**: In the 100 Gold Dev dataset, 61/100 cases belong to the `Delivery / Order` parent domain. A linear TF-IDF classifier trained on only 100 samples develops a massive decision boundary bias toward `Delivery / Order`, classifying 76% of test queries into `Delivery / Order`.
2. **Cascading Failure**: Once Stage 1 misclassifies a parent domain (e.g. mapping `account_access_login` to `Delivery / Order`), Stage 2 is restricted strictly to children of `Delivery / Order`, guaranteeing an E2E classification failure regardless of Stage 2's accuracy.

## 16. Limitations
- **Data Sparsity**: 100 Gold Dev examples is far too small to train an 8-estimator hierarchical pipeline (1 parent + 7 child models) without heavy over-fitting.
- **Cascading Error Bottleneck**: Hierarchical classification requires Stage 1 parent accuracy to exceed ~85-90% to be viable; at 49% Stage 1 accuracy, the pipeline degrades overall system performance.

## 17. Recommendation
**REJECT HIERARCHICAL CLASSIFIER FOR PRODUCTION**.
- Do NOT alter production configuration (`configs/config.yaml`).
- Retain `HybridRuleEmbeddingClassifier` as the default intent classification pipeline.
- Future work should explore unified dense embedding distance classifiers or class-imbalance reweighting rather than strict multi-stage hard routing.
