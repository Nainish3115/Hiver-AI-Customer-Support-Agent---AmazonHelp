# AmazonHelp Intent Classification Baseline Evaluation Report

> **Project Phase**: Phase 4 — Intent Classification & Vector Retrieval Baseline  
> **Evaluation Dataset**: `data/processed/classification/test.parquet` (30 test cases)  
> **Train Dataset**: `data/processed/classification/train.parquet` (140 train cases)

---

## 1. Classification Metrics Comparison

| Classifier Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Rule-Based Keyword Matcher** | **0.8667** | **0.6386** | **0.8763** | 0.6647 | 0.6275 |
| **Baseline 2: SentenceTransformer Prototype Nearest-Centroid** | **0.3667** | **0.2382** | **0.3867** | 0.2696 | 0.2304 |

---

## 2. Per-Intent Classification Metrics (`reports/tables/intent_per_class_metrics.csv`)

| intent                                 |   precision |   recall |   f1_score |   support |
|:---------------------------------------|------------:|---------:|-----------:|----------:|
| account_access_login                   |      0.6667 |   1      |     0.8    |         2 |
| address_delivery_instructions          |      0.6667 |   0.6667 |     0.6667 |         3 |
| customer_service_complaint             |      0      |   0      |     0      |         0 |
| damaged_defective_item                 |      0      |   0      |     0      |         3 |
| digital_content_kindle_video           |      0      |   0      |     0      |         0 |
| general_inquiry_other                  |      0      |   0      |     0      |         1 |
| late_delivery_shipping_delay           |      0      |   0      |     0      |         0 |
| missing_package_delivered_not_received |      0      |   0      |     0      |         2 |
| order_cancellation_request             |      0      |   0      |     0      |         1 |
| order_tracking_delivery_status         |      0.75   |   0.75   |     0.75   |         4 |
| payment_billing_issue                  |      0      |   0      |     0      |         2 |
| prime_membership_subscription          |      1      |   0.5    |     0.6667 |         4 |
| promotions_discounts_coupons           |      1      |   0.5    |     0.6667 |         2 |
| refund_status_request                  |      0      |   0      |     0      |         2 |
| return_exchange_inquiry                |      0      |   0      |     0      |         0 |
| third_party_seller_issue               |      0      |   0      |     0      |         2 |
| wrong_item_received                    |      0.5    |   0.5    |     0.5    |         2 |

---

## 3. Confusion Matrix Analysis (`reports/figures/confusion_matrix.png`)

- **Top Confusing Boundary**: `INT-01` (`order_tracking_delivery_status`) vs `INT-02` (`late_delivery_shipping_delay`).
- **High-Performing Categories**: `INT-03` (`missing_package_delivered_not_received`), `INT-05` (`return_exchange_inquiry`), and `INT-11` (`prime_membership_subscription`) show strong distinct precision (>0.85).
