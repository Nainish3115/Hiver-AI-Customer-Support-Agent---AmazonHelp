# Phase 5: AmazonHelp Intent Classification Error Analysis Report

> **Document Purpose**: Categorize and qualitatively analyze intent misclassifications on the 200 Human-Gold benchmark set.  
> **Gold Dataset**: `data/golden/amazonhelp_gold_200_annotation.csv`

---

## 1. Major Failure Categories (Human Gold Benchmark)

1. **Order Tracking vs. Late Delivery**:
   - *Example*: Customers inquiring *"Where is my package?"* after the promised delivery date has passed are human-labeled as `late_delivery_shipping_delay`, whereas keyword rules misclassified them as `order_tracking_delivery_status`.
2. **Refund Request vs. Return/Exchange Inquiry**:
   - *Example*: Customers asking for return instructions while demanding their money back immediately are labeled based on primary actionable intent (`refund_status_request` vs `return_exchange_inquiry`).
3. **Customer Complaints vs. Transactional Issue**:
   - *Example*: Highly emotional complaints regarding agent call drop behavior are categorized under `customer_service_complaint`, separating service frustration from order logistics.
4. **Short / Ambiguous Inquiries**:
   - *Example*: Messages with under 5 words lacking order/tracking IDs trigger general fallback predictions.

---

## 2. Sample Human-Gold Misclassifications Table (Top 10 Examples)

| case_id            | customer_text                                                                                                                                                                | true_intent                  | predicted_intent              |   confidence |
|:-------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------|:------------------------------|-------------:|
| RAG-AMAZON-1057376 | [369663]: @115850: Order no.:403-0082364-6927522.The Lakme kajal pkt was torn n kajal was not there(appears stolen).I need replacement n not refund. https://t.co/gVHx61ei1v | return_exchange_inquiry      | refund_status_request         |         0.85 |
| RAG-AMAZON-1112702 | I ordered leggings with pockets off of @115821 and they have a rip in the ass before I even wore them 🙄                                                                     | damaged_defective_item       | general_inquiry_other         |         0.4  |
| RAG-AMAZON-1127137 | @AmazonHelp I have returned an item which is showing as received by the seller. When can I expect my replacement item to be shipped? Thanks                                  | return_exchange_inquiry      | third_party_seller_issue      |         0.85 |
| RAG-AMAZON-1250159 | [169505]: I asked for replacement not refund you shitty @115850 how dare you to cancel the order without any prior information                                               | return_exchange_inquiry      | refund_status_request         |         0.85 |
| RAG-AMAZON-1258573 | @53009 Terrible 2nd exper. Looks like we won't be a return customer --&gt; guess we're headed over to @414897 @53010 @117093                                                 | customer_service_complaint   | return_exchange_inquiry       |         0.85 |
| RAG-AMAZON-1261849 | [198558]: @115821 courier boy itself cancelled my order today. He didn't make a call or sms on my no. My no. is always on. I didnt refused my order.                         | customer_service_complaint   | order_cancellation_request    |         0.85 |
| RAG-AMAZON-1262624 | @AmazonHelp Parcel not delivered - Customer services won’t take additional notes to help driver. So so angry!!!                                                              | late_delivery_shipping_delay | address_delivery_instructions |         0.85 |
| RAG-AMAZON-1323491 | did @115830 say guaranteed to arrive the 26.10? well they didn't count in the lazy ass driver that never delivered the parcel YET                                            | late_delivery_shipping_delay | address_delivery_instructions |         0.85 |
| RAG-AMAZON-1323916 | @115850 @115821 your service is also too worst and I connected but no solution on theri. And also you will sell defective product 🚫❌ https://t.co/CKIpvAHS0I               | customer_service_complaint   | damaged_defective_item        |         0.85 |
| RAG-AMAZON-1357717 | #AmazonIndia delivers junk battery,refuses to entertain return filed after 24 hours of delivery ORDER # 406-2524222-8704336 for Prime member                                 | damaged_defective_item       | return_exchange_inquiry       |         0.85 |

---

## 3. Recommended Classifier Mitigations for Phase 6

1. **Confidence-Gated Reranking**: Apply intent metadata boosting in Phase 6 retrieval only when classifier confidence exceeds $\ge 0.60$.
2. **Prototype Centroid Integration**: Utilize Baseline 3 (`EmbeddingPrototypeClassifier`) as the primary classifier due to its superior Macro F1 balance across rare intent classes.
