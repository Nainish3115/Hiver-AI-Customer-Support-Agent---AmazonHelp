# Phase 5: AmazonHelp Human Gold Label Audit Report

> **Project Phase**: Phase 5 — Human Gold Evaluation & Benchmark Audit  
> **Gold Dataset File**: [`data/golden/amazonhelp_gold_200_annotation.csv`](file:///D:/hiver-support-agent/data/golden/amazonhelp_gold_200_annotation.csv)  
> **Status**: **PASS (100% Structural Integrity Confirmed)**  

---

## 1. Executive Summary

A structural audit was conducted on the completed 200-example `AmazonHelp` human gold dataset. All 200 customer support cases have been manually labeled by a human annotator across the approved 17-intent taxonomy.

### Key Audit Metrics
- **Total Gold Examples**: **200**
- **Unique Case IDs**: **200** (0 duplicates)
- **Unique Customer Problem Texts**: **200** (0 exact text duplicates)
- **Non-Empty Human Gold Labels**: **200** (0 missing labels)
- **Taxonomy Validation**: **0 invalid intent strings** (All 200 belong to the approved 17-intent taxonomy)
- **Category Coverage**: **All 17 intents are represented** ($N \ge 6$ cases per intent)

---

## 2. Human Gold Intent Distribution

| Intent Category ID & Name | Human Gold Count (N=200) | Proportion (%) |
| :--- | :---: | :---: |
| `order_tracking_delivery_status` | **7** | 3.50% |
| `late_delivery_shipping_delay` | **32** | 16.00% |
| `missing_package_delivered_not_received` | **13** | 6.50% |
| `refund_status_request` | **8** | 4.00% |
| `return_exchange_inquiry` | **9** | 4.50% |
| `order_cancellation_request` | **7** | 3.50% |
| `damaged_defective_item` | **13** | 6.50% |
| `wrong_item_received` | **13** | 6.50% |
| `payment_billing_issue` | **10** | 5.00% |
| `account_access_login` | **6** | 3.00% |
| `prime_membership_subscription` | **8** | 4.00% |
| `digital_content_kindle_video` | **6** | 3.00% |
| `third_party_seller_issue` | **6** | 3.00% |
| `promotions_discounts_coupons` | **15** | 7.50% |
| `address_delivery_instructions` | **6** | 3.00% |
| `customer_service_complaint` | **26** | 13.00% |
| `general_inquiry_other` | **15** | 7.50% |
| **Total** | **200** | **100.00%** |

---

## 3. Structural Audit Verification Checklist

- [x] **Sample Count**: Exactly 200 cases.
- [x] **Case ID Uniqueness**: 200 unique `case_id` values.
- [x] **Customer Text Uniqueness**: 0 exact duplicate customer texts.
- [x] **Label Completeness**: 200 non-null `final_human_intent` labels.
- [x] **Taxonomy Strictness**: All labels match approved intent identifiers in `AMAZON_INTENTS`.
- [x] **Class Coverage**: All 17 intents represented ($N \ge 6$).

---

## 4. Automated Heuristic vs. Human Annotation Agreement

- **Overall Agreement**: **109 / 200 (54.50%)**
- **Disagreement Count**: **91 / 200 (45.50%)**

> **Analysis**: Automated heuristic pattern matching agreed with human gold labels on **54.5%** of cases. The 45.5% disagreement rate underscores why human gold evaluation is essential—heuristic rules produced optimistic proxy metrics on ambiguous or templated cases.
