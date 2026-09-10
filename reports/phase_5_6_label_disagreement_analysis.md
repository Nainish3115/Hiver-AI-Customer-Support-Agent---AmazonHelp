# Phase 5.6: Human-vs-Heuristic Label Disagreement Analysis Report

> **Project Phase**: Phase 5.6 — Diagnostic & Classifier Improvement Experiment  
> **Evaluation Dataset**: **200 HUMAN GOLD EXAMPLES** ([`data/golden/amazonhelp_gold_200_annotation.csv`](file:///D:/hiver-support-agent/data/golden/amazonhelp_gold_200_annotation.csv))  
> **Status**: **COMPLETE & VERIFIED**  

---

## 1. Executive Summary

This report presents an empirical audit of the disagreements between automated heuristic pattern matching (`suggested_intent`) and authoritative human annotations (`final_human_intent`) across the 200-example `AmazonHelp` gold evaluation set.

### Key Agreement Statistics
- **Total Gold Cases Evaluated**: **200**
- **Exact Agreements**: **109 / 200 (54.50%)**
- **Disagreements**: **91 / 200 (45.50%)**

> **Diagnostic Insight**: Automated heuristic rules disagreed with human annotators on **45.5%** of cases. Supervised machine learning classifiers trained on noisy heuristic labels overfitted to these rule-generator shortcuts, explaining why supervised models collapsed from 86%-93% proxy accuracy to 11%-18% accuracy when evaluated against true Human Gold labels.

---

## 2. Most Frequent Heuristic $\rightarrow$ Human Disagreement Pairs

The table below highlights the top 15 disagreement patterns where automated heuristic discovery rules misassigned intent compared to human gold standards:

| heuristic_suggested                    | human_gold_intent              |   disagreement_count |   pct_of_disagreements |
|:---------------------------------------|:-------------------------------|---------------------:|-----------------------:|
| digital_content_kindle_video           | general_inquiry_other          |                    4 |                    4.4 |
| order_tracking_delivery_status         | late_delivery_shipping_delay   |                    4 |                    4.4 |
| address_delivery_instructions          | late_delivery_shipping_delay   |                    3 |                    3.3 |
| damaged_defective_item                 | customer_service_complaint     |                    3 |                    3.3 |
| general_inquiry_other                  | promotions_discounts_coupons   |                    3 |                    3.3 |
| prime_membership_subscription          | late_delivery_shipping_delay   |                    3 |                    3.3 |
| missing_package_delivered_not_received | refund_status_request          |                    3 |                    3.3 |
| third_party_seller_issue               | late_delivery_shipping_delay   |                    3 |                    3.3 |
| refund_status_request                  | return_exchange_inquiry        |                    2 |                    2.2 |
| general_inquiry_other                  | damaged_defective_item         |                    2 |                    2.2 |
| account_access_login                   | general_inquiry_other          |                    2 |                    2.2 |
| general_inquiry_other                  | late_delivery_shipping_delay   |                    2 |                    2.2 |
| payment_billing_issue                  | order_tracking_delivery_status |                    2 |                    2.2 |
| payment_billing_issue                  | customer_service_complaint     |                    2 |                    2.2 |
| general_inquiry_other                  | customer_service_complaint     |                    2 |                    2.2 |

---

## 3. Per-Intent Agreement Rates

The table below breaks down the agreement rate between heuristic suggestions and human gold annotations across all 17 intents:

| intent                                 |   human_gold_support |   heuristic_match_count |   agreement_rate_pct |
|:---------------------------------------|---------------------:|------------------------:|---------------------:|
| order_cancellation_request             |                    7 |                       7 |               100    |
| account_access_login                   |                    6 |                       6 |               100    |
| wrong_item_received                    |                   13 |                      11 |                84.62 |
| address_delivery_instructions          |                    6 |                       5 |                83.33 |
| third_party_seller_issue               |                    6 |                       5 |                83.33 |
| prime_membership_subscription          |                    8 |                       6 |                75    |
| return_exchange_inquiry                |                    9 |                       6 |                66.67 |
| digital_content_kindle_video           |                    6 |                       4 |                66.67 |
| refund_status_request                  |                    8 |                       5 |                62.5  |
| damaged_defective_item                 |                   13 |                       8 |                61.54 |
| promotions_discounts_coupons           |                   15 |                       9 |                60    |
| order_tracking_delivery_status         |                    7 |                       4 |                57.14 |
| missing_package_delivered_not_received |                   13 |                       6 |                46.15 |
| payment_billing_issue                  |                   10 |                       4 |                40    |
| customer_service_complaint             |                   26 |                      10 |                38.46 |
| late_delivery_shipping_delay           |                   32 |                      10 |                31.25 |
| general_inquiry_other                  |                   15 |                       3 |                20    |

### Observations on High & Low Agreement Categories
- **Highest Agreement Intents**:
  - `account_access_login` (**100.0%**): Clear keyword signals like password reset, OTP, login lockouts.
  - `order_cancellation_request` (**100.0%**): Unambiguous cancellation phrasing ("cancel my order").
  - `wrong_item_received` (**84.6%**): Specific phrases like "received wrong item" or "ordered X got Y".
  - `third_party_seller_issue` (**83.3%**): Specific references to third-party marketplace sellers.
- **Lowest Agreement Intents**:
  - `general_inquiry_other` (**20.0%**): Heuristic rules assigned `general_inquiry_other` as a catch-all when regex patterns failed, whereas human annotators identified domain intent (e.g. promotions, damaged item).
  - `late_delivery_shipping_delay` (**31.2%**): Human annotators identified 32 late delivery cases (16.0% of gold dataset), but heuristic rules misclassified many as tracking inquiries or address issues.
  - `customer_service_complaint` (**38.5%**): Human annotators identified 26 customer complaint cases (13.0% of gold dataset), whereas heuristic rules assigned them to underlying transactional issues.

---

## 4. Taxonomy Boundary Analysis & Recommendations

1. **`order_tracking_delivery_status` vs `late_delivery_shipping_delay`**:
   - *Conflict Pattern*: Customers asking *"Where is my package?"* after the promised delivery date has passed.
   - *Recommendation*: Preserve both intents in the taxonomy. In classifier training, feature engineering must incorporate temporal delay keywords ("overdue", "was supposed to arrive yesterday") to separate tracking from delay.
2. **`refund_status_request` vs `return_exchange_inquiry`**:
   - *Conflict Pattern*: Customers initiating a return often ask about refund timing simultaneously.
   - *Recommendation*: Enforce the primary action rule: annotate `return_exchange_inquiry` if a label/method is requested, and `refund_status_request` if money reimbursement status is requested.
3. **`customer_service_complaint` vs Transactional Issue**:
   - *Conflict Pattern*: Angry customers complaining about agent phone drop or rude service.
   - *Recommendation*: Human gold annotations correctly prioritized agent conduct complaints (`customer_service_complaint`).
