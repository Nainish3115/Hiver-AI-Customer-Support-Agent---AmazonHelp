# AmazonHelp Conversation Quality & Intent Analysis Report

> **Project Phase**: Phase 2 — Conversation Reconstruction, Quality Filtering, and Intent Discovery  
> **Target Brand**: `AmazonHelp`  
> **Extraction Source**: `data/raw/customer_support_on_twitter.csv`

---

## 1. AmazonHelp Extraction & Thread Statistics

- **Total Extracted Tweets**: **373,438**
- **Total Reconstructed Conversation Threads**: **82,676**
- **Threads with Both Customer and Support Messages**: **82,673** (**100.0%**)
- **Single-Tweet Orphans**: **0**
- **Threads with Missing External Parents**: **966**

### Thread Length Distribution

| Thread Length (Total Tweets) | Conversation Count | Percentage |
| :---: | :---: | :---: |
| **1 Tweet (Orphan)** | 0 | 0.0% |
| **2 Tweets (Single Turn: 1 Q + 1 A)** | 31,341 | 37.91% |
| **3–4 Tweets (Multi-Turn)** | 25,762 | 31.16% |
| **5+ Tweets (Deep Multi-Turn)** | 25,573 | 30.93% |

---

## 2. Conversation Quality & Eligibility Criteria

Conversations were evaluated against strict quality rules for downstream RAG candidate selection:
1. `contains_both_participants`: Must contain both customer and support messages.
2. `min_length_2`: Must have at least 2 total tweets.
3. `min_text_length_15`: Initial customer text length $\ge 15$ characters.
4. `valid_duration`: Duration $\ge 0$ seconds and $\le 7$ days.

### Filtering Impact

- **Total Reconstructed Conversations**: **82,676**
- **Total Eligible RAG Conversations**: **81,186**
- **Filtered Out (Incomplete / Noise)**: **1,490** (**1.8%**)

---

## 3. Discovered Intent Distribution Summary

The top 5 discovered customer intents for `AmazonHelp` represent **70.5%** of all inquiries:
1. `order_tracking_delivery_status`: **26.4%**
2. `late_delivery_shipping_delay`: **18.2%**
3. `missing_package_delivered_not_received`: **12.5%**
4. `refund_status_request`: **8.6%**
5. `return_exchange_inquiry`: **6.4%**

Full 17-intent taxonomy documentation is saved in [`reports/amazonhelp_intent_taxonomy.md`](file:///d:/hiver-support-agent/reports/amazonhelp_intent_taxonomy.md).

---

## 4. Risks & Limitations

1. **DM Redirection Truncation**: Support reps often reply *"Please DM us your order details to assist"*, truncating the public resolution text. **Mitigation**: Filter out 2-turn DM redirection responses when creating the final grounded resolution RAG pairs.
2. **Multi-Intent Customer Queries**: Customers frequently combine issues (e.g. late delivery + refund request). **Mitigation**: Support multi-label classification tag assignment in future phases.
