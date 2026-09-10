# Phase 7.1 Human Escalation Annotation Guidelines

This document provides clear instructions for human annotators assessing whether customer support interactions in the `AmazonHelp` customer support dataset should be auto-handled by the AI agent (`AUTO_HANDLE`) or routed to human customer support agents (`ESCALATE_TO_HUMAN`).

---

## 1. Objective

The goal is to establish an independent **human-verified ground truth** for escalation suitability on a benchmark of 50 AmazonHelp customer support queries.

Annotators must judge the **support interaction itself** and the adequacy of available historical evidence, **NOT** whether the model's prediction happened to be right or wrong.

---

## 2. Dataset Schema

The annotation dataset `data/golden/amazonhelp_escalation_validation_50.csv` contains the following fields:

- `case_id`: Unique identifier for the customer support case.
- `customer_text`: The customer's message on Twitter (@AmazonHelp).
- `gold_intent`: The human-verified intent label from Phase 5.5.
- `should_auto_handle`: **ANNOTATOR FIELD**. Set to `TRUE` if safe to auto-handle, or `FALSE` if it requires human escalation.
- `escalation_reason`: **ANNOTATOR FIELD**. Reason code for decision (e.g., `CLEAR_QUERY_GROUNDED_EVIDENCE`, `AMBIGUOUS_QUERY`, `REQUIRES_ACCOUNT_ACTION`, `UNSAFE_OR_DEFECTIVE_PRODUCT`).
- `annotator_notes`: **ANNOTATOR FIELD**. Free-text explanation of the human judgment.

---

## 3. Decision Rules for Human Annotators

### A. Mark `should_auto_handle = TRUE` when:

1. **Clear Customer Query**: The customer's message clearly states a routine question, status check, or standard inquiry.
2. **Usable Resolution Pattern**: A standard customer support response (e.g. directing customer to tracking page, explaining refund timeline, explaining Prime policy) is appropriate and safe.
3. **No Account State Modification**: The query does not require performing sensitive, unverified account operations (e.g., balance transfers, password resets, manual money refund overrides).
4. **No Severe Escalation Triggers**: The message does not involve legal threats, security breaches, or severe safety hazards.

---

### B. Mark `should_auto_handle = FALSE` (Escalate to Human) when:

1. **Material Ambiguity / Insufficient Details**: The customer's issue cannot be understood without asking multiple clarifying questions or inspecting unprovided order numbers.
2. **Account Access & Security Breaches**: Hacked accounts, unauthorized credit card charges, suspicious logins (`account_access_login`).
3. **Complex Payment & Financial Disputes**: Unauthorized charges, double billing, bank cashbacks missing (`payment_billing_issue`).
4. **Damaged / Stolen Items Requiring Manual Inspection**: Damaged goods, stolen packages, broken screens (`damaged_defective_item`).
5. **High-Frustration / Legal / Regulatory Threat**: Escalations threatening legal action, consumer court, or extreme agent misbehavior.
6. **Third-Party Seller Fraud / Disputes**: Fraudulent marketplace sellers requiring agent intervention (`third_party_seller_issue`).

---

## 4. Annotation Process

1. Open `data/golden/amazonhelp_escalation_validation_50.csv`.
2. For each row (1 to 50), read `customer_text` and `gold_intent`.
3. Evaluate whether a automated grounded support bot response is safe and sufficient.
4. Populate `should_auto_handle` with `TRUE` or `FALSE`.
5. Populate `escalation_reason` and `annotator_notes`.
6. Save the file.

---

## 5. Multi-Annotator Agreement (Optional)

If a second annotator reviews the dataset:
- Store choices in `annotator_1_decision` and `annotator_2_decision`.
- Raw agreement and Cohen's Kappa will be calculated automatically by the evaluation suite.
- If only one annotator exists, single-annotator validation is reported.
