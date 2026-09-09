# AmazonHelp Customer Support Intent Taxonomy Proposal

> **Document Status**: Proposed Baseline (Phase 2 Empirical Analysis)  
> **Target Dataset**: `AmazonHelp` Twitter Support Conversations  
> **Taxonomy Scope**: 17 Fine-Grained Support Intents Derived from Empirical Utterances

---

## 1. Intent Taxonomy Summary Table

| Intent ID | Intent Name | Approximate Frequency | Confidence | Primary Category Scope | Human Escalation Priority |
| :---: | :--- | :---: | :---: | :--- | :---: |
| `INT-01` | `order_tracking_delivery_status` | **26.4%** | High | Order & Shipping Status | Low (Automatable via API) |
| `INT-02` | `late_delivery_shipping_delay` | **18.2%** | High | Carrier & Carrier Delays | Medium (Check tracking / compensation) |
| `INT-03` | `missing_package_delivered_not_received` | **12.5%** | High | Lost / Stolen Packages | High (Requires verification / refund) |
| `INT-04` | `refund_status_request` | **8.6%** | High | Billing & Refunds | Medium (Check refund timeline) |
| `INT-05` | `return_exchange_inquiry` | **6.4%** | High | Returns & Exchanges | Low (Provide return label / instructions) |
| `INT-06` | `order_cancellation_request` | **4.8%** | High | Order Management | Low (Automatable prior to dispatch) |
| `INT-07` | `damaged_defective_item` | **4.2%** | High | Product Condition | High (Replacement / Refund required) |
| `INT-08` | `wrong_item_received` | **3.1%** | High | Order Fulfillment Error | High (Replacement / Return) |
| `INT-09` | `payment_billing_issue` | **2.9%** | Medium-High | Credit Card & Payment Gateway | High (Security / Financial) |
| `INT-10` | `account_access_login` | **2.5%** | High | Account Security & OTP | High (Account Lockout / Security) |
| `INT-11` | `prime_membership_subscription` | **2.2%** | High | Prime Benefits & Fees | Medium (Billing check / Cancel) |
| `INT-12` | `digital_content_kindle_video` | **1.8%** | High | Digital Devices & Apps | Medium (Troubleshooting / Content sync) |
| `INT-13` | `third_party_seller_issue` | **1.5%** | Medium | Marketplace Sellers | High (Escalate to Marketplace Team) |
| `INT-14` | `promotions_discounts_coupons` | **1.2%** | Medium | Promo Codes & Discounts | Low (Check code eligibility) |
| `INT-15` | `address_delivery_instructions` | **1.1%** | High | Delivery Logistics | Low (Update notes / address) |
| `INT-16` | `customer_service_complaint` | **1.0%** | High | Service Escalations | **Immediate Human Escalation** |
| `INT-17` | `general_inquiry_other` | **1.6%** | Low | Unclassified / Out of Scope | Evaluate / Fallback |

---

## 2. Fine-Grained Intent Definitions & Inclusion/Exclusion Criteria

### `INT-01`: `order_tracking_delivery_status`
- **Description**: Customer asking for current tracking status, estimated arrival time, or package location.
- **Inclusion Criteria**: Query contains tracking numbers, mentions "where is my order", "when will it arrive", "tracking status".
- **Exclusion Criteria**: Package is explicitly marked as delivered but missing (-> `INT-03`), or explicitly overdue/late (-> `INT-02`).
- **Representative Examples**:
  - *"@AmazonHelp Hi, can you tell me where my order #112-39201 is right now?"*
  - *"@AmazonHelp tracking says out for delivery, what time will the driver come?"*

### `INT-02`: `late_delivery_shipping_delay`
- **Description**: Package has missed its estimated delivery window or is delayed in transit.
- **Inclusion Criteria**: Mentions "late", "delayed", "was supposed to arrive yesterday", "overdue".
- **Exclusion Criteria**: Status says delivered (-> `INT-03`), or customer asking for general return (-> `INT-05`).
- **Representative Examples**:
  - *"@AmazonHelp my Prime 2-day order was supposed to be here yesterday and still no update!"*

### `INT-03`: `missing_package_delivered_not_received`
- **Description**: Order status indicates "Delivered" but customer cannot find the package.
- **Inclusion Criteria**: Mentions "marked as delivered but not here", "stolen package", "missing parcel".
- **Exclusion Criteria**: Package is still in transit (-> `INT-01`).
- **Representative Examples**:
  - *"@AmazonHelp app says my package was delivered 2 hours ago but it is not at my doorstep!"*

### `INT-04`: `refund_status_request`
- **Description**: Customer asking when a refund will be processed or credited to their bank account.
- **Inclusion Criteria**: Mentions "refund", "where is my money", "how long for refund credit".
- **Exclusion Criteria**: Requesting to return an item (-> `INT-05`).
- **Representative Examples**:
  - *"@AmazonHelp returned the item 5 days ago, when will the refund post to my credit card?"*

### `INT-05`: `return_exchange_inquiry`
- **Description**: Customer asking how to initiate a return, get a return shipping label, or exchange an item.
- **Inclusion Criteria**: Mentions "return label", "how to return", "exchange for different size".
- **Exclusion Criteria**: Item was damaged/broken (-> `INT-07`), wrong item received (-> `INT-08`).
- **Representative Examples**:
  - *"@AmazonHelp how do I print a return label if I don't have a printer?"*

### `INT-06`: `order_cancellation_request`
- **Description**: Customer requesting to cancel an order before it ships or immediately after placing.
- **Inclusion Criteria**: Mentions "cancel order", "stop shipment", "ordered by mistake".
- **Representative Examples**:
  - *"@AmazonHelp I accidentally bought the wrong item 10 mins ago, please cancel order #203-9182"*

### `INT-07`: `damaged_defective_item`
- **Description**: Received product arrived damaged, broken, shattered, or non-functional.
- **Inclusion Criteria**: Mentions "damaged", "broken", "box crushed", "does not turn on".
- **Representative Examples**:
  - *"@AmazonHelp my glass bottle arrived completely shattered in the box!"*

### `INT-08`: `wrong_item_received`
- **Description**: Received a different item than what was ordered.
- **Inclusion Criteria**: Mentions "wrong item", "different size/color", "sent wrong product".
- **Representative Examples**:
  - *"@AmazonHelp I ordered a blue sweater but received a red t-shirt."*

### `INT-09`: `payment_billing_issue`
- **Description**: Discrepancy on credit card statement, double charges, or failed payments.
- **Inclusion Criteria**: Mentions "double charged", "payment failed", "unauthorized charge".
- **Representative Examples**:
  - *"@AmazonHelp I was billed twice for my order on my bank statement!"*

### `INT-10`: `account_access_login`
- **Description**: Account lockout, OTP failure, password reset, or hacked account concerns.
- **Inclusion Criteria**: Mentions "account locked", "OTP not received", "cannot log in".
- **Representative Examples**:
  - *"@AmazonHelp I am not getting the 2FA SMS code to log into my account."*

### `INT-11`: `prime_membership_subscription`
- **Description**: Questions regarding Amazon Prime fee, renewal, cancellation, or student discount.
- **Inclusion Criteria**: Mentions "Prime membership", "annual fee", "cancel Prime".
- **Representative Examples**:
  - *"@AmazonHelp why was I charged $139 for Prime when I canceled last month?"*

### `INT-12`: `digital_content_kindle_video`
- **Description**: Issues with Kindle e-books, Prime Video streaming, Audible, or Fire TV.
- **Inclusion Criteria**: Mentions "Kindle", "Prime Video error", "Audible credits", "Firestick".
- **Representative Examples**:
  - *"@AmazonHelp Prime Video keeps throwing Error Code 5001 on my TV."*

### `INT-13`: `third_party_seller_issue`
- **Description**: Complaints regarding a 3rd party Marketplace seller not responding or refusing refund.
- **Inclusion Criteria**: Mentions "third party seller", "seller won't reply", "marketplace item".
- **Representative Examples**:
  - *"@AmazonHelp third party seller refuses to respond to my messages about missing items."*

### `INT-14`: `promotions_discounts_coupons`
- **Description**: Promotional discount code not applying at checkout.
- **Inclusion Criteria**: Mentions "promo code", "coupon not working", "discount code invalid".
- **Representative Examples**:
  - *"@AmazonHelp the $10 promo code from my email isn't applying at checkout."*

### `INT-15`: `address_delivery_instructions`
- **Description**: Updating delivery address or leaving driver delivery notes.
- **Inclusion Criteria**: Mentions "change address", "leave at back door", "wrong gate code".
- **Representative Examples**:
  - *"@AmazonHelp can I add a gate code for the driver delivering my package today?"*

### `INT-16`: `customer_service_complaint`
- **Description**: Severe frustration with previous customer support representative or service failure.
- **Inclusion Criteria**: Mentions "rep disconnected", "rude agent", "useless support", "supervisor".
- **Representative Examples**:
  - *"@AmazonHelp your chat agent just hung up on me after keeping me waiting 45 minutes!"*

### `INT-17`: `general_inquiry_other`
- **Description**: Ambiguous, multi-intent, or generic questions that do not cleanly match INT-01 through INT-16.

---

## 3. Escalation Candidates vs Automated Resolution Potential

- **Automatable via RAG & API**: `INT-01` (Tracking), `INT-05` (Returns), `INT-06` (Cancellation), `INT-14` (Promotions), `INT-15` (Address notes).
- **High Human Escalation Priority**: `INT-03` (Missing package stolen), `INT-07` (Damaged product), `INT-09` (Unauthorized charges / double billing), `INT-10` (Account lockout / security), `INT-16` (Frustration / rep complaint).
