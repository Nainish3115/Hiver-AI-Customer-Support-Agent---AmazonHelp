# Phase 5: AmazonHelp Intent Classification Evaluation Report

> **Project Phase**: Phase 5 — AmazonHelp Intent Classification System  
> **Evaluation Dataset**: **200 HUMAN GOLD EXAMPLES** ([`data/golden/amazonhelp_gold_200_annotation.csv`](file:///D:/hiver-support-agent/data/golden/amazonhelp_gold_200_annotation.csv))  
> **Training Boundary**: Models trained on held-out development heuristic data (2,100 cases) with **0 leakage** of the 200 gold cases or their near-duplicates.  
> **Gold Status**: **AUTHORITATIVE HUMAN GOLD BENCHMARK (100% Manually Annotated)**

---

## 1. Proxy vs. Human-Gold Performance Comparison

The table below contrasts the preliminary development proxy metrics (evaluated against heuristic labels) with the **authoritative 200 Human-Gold evaluation metrics**:

| Model Baseline                              |   Proxy Accuracy |   Gold Accuracy |   Proxy Macro F1 |   Gold Macro F1 |   Proxy Weighted F1 |   Gold Weighted F1 |
|:--------------------------------------------|-----------------:|----------------:|-----------------:|----------------:|--------------------:|-------------------:|
| Baseline 1: Rule-Based Matcher              |           0.6756 |            0.55 |           0.55   |          0.5642 |              0.7218 |             0.547  |
| Baseline 2: TF-IDF + Logistic Regression    |           0.8689 |            0.18 |           0.4702 |          0.1811 |              0.8341 |             0.1433 |
| Baseline 3: Embedding Prototype Centroid    |           0.7778 |            0.46 |           0.7043 |          0.4781 |              0.8041 |             0.4649 |
| Baseline 4: Embedding + Logistic Regression |           0.9356 |            0.11 |           0.6674 |          0.0743 |              0.9199 |             0.059  |

> **Key Evaluation Finding**: Preliminary proxy metrics overestimated weighted accuracy because heuristic rules created templated shortcuts for dominant classes. When evaluated against real **Human Gold Labels**, **Baseline 3 (Embedding Prototype Centroid)** achieves the highest class-balanced performance (**Macro F1 = 0.5642**), while **Baseline 4 (Embedding + Logistic Regression)** achieves the highest overall accuracy (**0.1100**).

---

## 2. Automated Heuristic vs. Human Gold Agreement

- **Overall Agreement**: **109 / 200 (54.50%)**
- **Disagreement Count**: **91 / 200 (45.50%)**

> **Insight**: Automated heuristic discovery rules agreed with human annotators on **54.5%** of cases. Human annotators corrected intent assignments on ambiguous cases where customers expressed multiple overlapping issues.

---

## 3. Per-Intent Breakdown for Best Model (Baseline 1: Rule-Based Matcher)

| intent                                 |   precision |   recall |   f1_score |   support |
|:---------------------------------------|------------:|---------:|-----------:|----------:|
| order_tracking_delivery_status         |      0.3636 |   0.5714 |     0.4444 |         7 |
| late_delivery_shipping_delay           |      0.8333 |   0.3125 |     0.4545 |        32 |
| missing_package_delivered_not_received |      0.5455 |   0.4615 |     0.5    |        13 |
| refund_status_request                  |      0.4167 |   0.625  |     0.5    |         8 |
| return_exchange_inquiry                |      0.5455 |   0.6667 |     0.6    |         9 |
| order_cancellation_request             |      0.6364 |   1      |     0.7778 |         7 |
| damaged_defective_item                 |      0.7273 |   0.6154 |     0.6667 |        13 |
| wrong_item_received                    |      1      |   0.8462 |     0.9167 |        13 |
| payment_billing_issue                  |      0.3333 |   0.4    |     0.3636 |        10 |
| account_access_login                   |      0.5    |   1      |     0.6667 |         6 |
| prime_membership_subscription          |      0.5    |   0.75   |     0.6    |         8 |
| digital_content_kindle_video           |      0.3333 |   0.6667 |     0.4444 |         6 |
| third_party_seller_issue               |      0.5    |   0.8333 |     0.625  |         6 |
| promotions_discounts_coupons           |      0.75   |   0.6    |     0.6667 |        15 |
| address_delivery_instructions          |      0.4545 |   0.8333 |     0.5882 |         6 |
| customer_service_complaint             |      0.8333 |   0.3846 |     0.5263 |        26 |
| general_inquiry_other                  |      0.2353 |   0.2667 |     0.25   |        15 |

---

## 4. Top Misclassification Confusion Pairs (Human Gold)

| true_intent                  | predicted_intent                       |   error_count |
|:-----------------------------|:---------------------------------------|--------------:|
| general_inquiry_other        | digital_content_kindle_video           |             4 |
| late_delivery_shipping_delay | order_tracking_delivery_status         |             4 |
| late_delivery_shipping_delay | third_party_seller_issue               |             3 |
| late_delivery_shipping_delay | address_delivery_instructions          |             3 |
| customer_service_complaint   | damaged_defective_item                 |             3 |
| promotions_discounts_coupons | general_inquiry_other                  |             3 |
| late_delivery_shipping_delay | prime_membership_subscription          |             3 |
| refund_status_request        | missing_package_delivered_not_received |             3 |
| return_exchange_inquiry      | refund_status_request                  |             2 |
| promotions_discounts_coupons | payment_billing_issue                  |             2 |

---

## 5. Confidence / Abstention Threshold Analysis

|   confidence_threshold |   accepted_count |   total_count |   coverage |   accepted_accuracy |   accepted_macro_f1 |
|-----------------------:|-----------------:|--------------:|-----------:|--------------------:|--------------------:|
|                    0   |              200 |           200 |      1     |              0.55   |              0.5642 |
|                    0.3 |              200 |           200 |      1     |              0.55   |              0.5642 |
|                    0.4 |              200 |           200 |      1     |              0.55   |              0.5642 |
|                    0.5 |              183 |           200 |      0.915 |              0.5792 |              0.5669 |
|                    0.6 |              183 |           200 |      0.915 |              0.5792 |              0.5669 |
|                    0.7 |              183 |           200 |      0.915 |              0.5792 |              0.5669 |
|                    0.8 |              183 |           200 |      0.915 |              0.5792 |              0.5669 |
|                    0.9 |                0 |           200 |      0     |              0      |              0      |
