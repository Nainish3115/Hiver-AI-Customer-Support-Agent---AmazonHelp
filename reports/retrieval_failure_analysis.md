# Retrieval Failure Analysis & Edge-Case Report

> **Document Purpose**: Categorize and analyze retrieval failures across test queries.

---

## 1. Primary Failure Categories

1. **Semantically Similar but Different Support Workflow**:  
   *Example*: Query asking for *order cancellation* retrieves historical cases for *return shipping labels*. Both deal with stopping an order, but require different support resolution steps.
2. **Generic DM Redirection Historical Cases**:  
   *Example*: Retrieved candidate consists of generic support response *"Please DM us your email address"*, which lacks specific domain resolution guidance.
3. **Short / Ambiguous Initial Queries**:  
   *Example*: Customer query *"Help please"* lacks domain context, retrieving unhelpful general inquiry cases.

---

## 2. Recommended Mitigations for Phase 5

1. **DM Redirection Filtering**: Strictly filter out DM redirection candidates during context construction.
2. **Intent-Gated Filtering**: Require Intent Classifier confidence $\ge 0.70$ before applying intent metadata filters.
