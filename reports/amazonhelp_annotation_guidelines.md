# AmazonHelp Support Intent Annotation Guidelines & Decision Boundaries

> **Document Status**: Official Guideline Specification (Phase 3)  
> **Target Dataset**: `AmazonHelp` Customer Support Conversations  
> **Purpose**: Standardize human annotation rules and establish clear decision boundaries between neighboring support intents.

---

## 1. Core Intent Decision Matrix & Neighboring Boundaries

### Boundary 1: `INT-01` (Order Tracking) vs `INT-02` (Late Delivery)
- **Rule**: If the customer simply asks for tracking status, ETA, or current location of a parcel **within the expected window**, assign `INT-01` (`order_tracking_delivery_status`).
- **Rule**: If the customer explicitly states that the delivery date **has passed**, **missed Prime 2-day guarantee**, or is **overdue/delayed**, assign `INT-02` (`late_delivery_shipping_delay`).
- **Example Boundary**:
  - *"Where is my package? Is it coming today?"* -> `INT-01`
  - *"My package was guaranteed for yesterday and it still hasn't arrived!"* -> `INT-02`

### Boundary 2: `INT-02` (Late Delivery) vs `INT-03` (Missing Package)
- **Rule**: If the status is "In Transit" or delayed, assign `INT-02`.
- **Rule**: If the tracking status explicitly claims **"Delivered"** but the customer cannot locate the package on their porch/lobby, assign `INT-03` (`missing_package_delivered_not_received`).
- **Example Boundary**:
  - *"Tracking hasn't updated in 3 days, is my order delayed?"* -> `INT-02`
  - *"App says delivered 1 hour ago but there is no box at my door!"* -> `INT-03`

### Boundary 3: `INT-04` (Refund Status) vs `INT-05` (Return Request)
- **Rule**: If the customer is asking **how to return** an item, how to get a return label, or where to drop off a parcel, assign `INT-05` (`return_exchange_inquiry`).
- **Rule**: If the item has **already been returned/shipped back**, and the customer is asking **when the refund money will post** to their bank account, assign `INT-04` (`refund_status_request`).
- **Example Boundary**:
  - *"How do I print a return label for these shoes?"* -> `INT-05`
  - *"I dropped off my return at UPS 4 days ago, when will I get my refund?"* -> `INT-04`

### Boundary 4: `INT-06` (Order Cancellation) vs `INT-05` (Return Request)
- **Rule**: If the customer wants to stop an order **before it has been shipped/received**, assign `INT-06` (`order_cancellation_request`).
- **Rule**: If the customer has **already received** the item and wants to send it back, assign `INT-05` (`return_exchange_inquiry`).
- **Example Boundary**:
  - *"I just ordered 5 mins ago by mistake, please cancel it!"* -> `INT-06`
  - *"I opened the box today and don't like the color, can I return it?"* -> `INT-05`

### Boundary 5: `INT-09` (Payment Issue) vs `INT-11` (Prime Subscription)
- **Rule**: If the charge is an unauthorized credit card charge, double billing on an order, or gift card error, assign `INT-09` (`payment_billing_issue`).
- **Rule**: If the charge is specifically an **annual/monthly Amazon Prime membership fee**, Prime Student fee, or Prime benefit issue, assign `INT-11` (`prime_membership_subscription`).
- **Example Boundary**:
  - *"Why was my credit card charged twice for order #102?"* -> `INT-09`
  - *"Why was I billed $139 for Prime membership when I didn't sign up?"* -> `INT-11`

### Boundary 6: `INT-07` (Damaged Item) vs `INT-08` (Wrong Item)
- **Rule**: If the customer received the **correct product model**, but it is physically broken, cracked, defective, or crushed, assign `INT-07` (`damaged_defective_item`).
- **Rule**: If the customer received a **completely different product**, wrong size, wrong color, or missing components inside the box, assign `INT-08` (`wrong_item_received`).
- **Example Boundary**:
  - *"The monitor I ordered arrived with a cracked screen."* -> `INT-07`
  - *"I ordered a monitor but received a box of coffee mugs."* -> `INT-08`

### Boundary 7: `INT-17` (General Inquiry) vs Specific Intents
- **Rule**: Use `INT-17` (`general_inquiry_other`) ONLY when the customer utterance is ambiguous, generic praise/thanks, or unclassifiable into `INT-01` through `INT-16`.

---

## 2. Intent Annotation Schema Fields

Human annotators filling `data/golden/taxonomy_validation_sample.parquet` or `.csv` will populate:
- `annotation_intent`: One of the 17 intent names.
- `annotation_confidence`: Float between `0.0` (uncertain) and `1.0` (high confidence).
- `is_multi_intent`: Boolean `True` if customer utterance combines multiple distinct intents.
- `should_escalate`: Boolean `True` if the issue requires immediate human agent intervention.
- `annotation_notes`: Free text explanation for ambiguous boundary cases.
