# Phase 5.5 Mandatory Completion Report — Human Annotation Workflow Preparation

> **Execution Timestamp**: 2026-09-10T12:32:00+05:30  
> **Phase Objective**: Prepare a clean, deterministic, leakage-safe 200-example human gold annotation workflow and interactive CLI tool for AmazonHelp intent evaluation.  
> **Status**: **WORKFLOW PREPARATION COMPLETE | HUMAN ANNOTATION PENDING**  

---

## 1. Overview & Key Findings

- **Selected Annotation Examples**: Exactly **200 cases** sampled deterministically from `data/golden/amazonhelp_intent_annotation_ready.parquet`.
- **Human Gold Status**: All 200 rows in the annotation files have `final_human_intent` set to `NULL` / empty string.

> [!IMPORTANT]
> **Human annotation has NOT been performed by this phase.**  
> No gold intent labels were assigned automatically, synthetically, or via LLMs. The dataset is in a clean, annotation-ready state for manual human labeling. No classifier performance is reported as human-gold performance.

---

## 2. Selection Methodology

1. **Source Corpus**: `data/golden/amazonhelp_intent_annotation_ready.parquet` (800 unique non-duplicate candidate cases).
2. **Deduplication**: Applied normalized text hashing (`normalize_text_for_grouping`) to ensure no exact or templated near-duplicate customer texts were selected.
3. **Fixed Random Seed**: `seed = 42` for 100% deterministic sampling reproducibility.
4. **Intent Stratification**: Sampled proportionally across all 17 AmazonHelp intent categories, enforcing a minimum floor of $N \ge 8$ per intent to preserve coverage for rare intent classes (e.g. `wrong_item_received`, `third_party_seller_issue`).
5. **Deterministic Ordering**: Output records are sorted strictly by `case_id`.

---

## 3. Sampled 200-Example Intent Distribution

The exact allocation across the 17 intent categories in `data/golden/amazonhelp_gold_200_annotation.parquet` is detailed below:

| Intent Category ID & Name | Candidate Count (800) | Sampled Count (N=200) | Proportion (%) |
| :--- | :---: | :---: | :---: |
| `general_inquiry_other` | 68 | 16 | 8.00% |
| `account_access_login` | 46 | 12 | 6.00% |
| `customer_service_complaint` | 46 | 12 | 6.00% |
| `digital_content_kindle_video` | 47 | 12 | 6.00% |
| `late_delivery_shipping_delay` | 46 | 12 | 6.00% |
| `payment_billing_issue` | 46 | 12 | 6.00% |
| `prime_membership_subscription` | 46 | 12 | 6.00% |
| `promotions_discounts_coupons` | 46 | 12 | 6.00% |
| `refund_status_request` | 46 | 12 | 6.00% |
| `address_delivery_instructions` | 48 | 11 | 5.50% |
| `damaged_defective_item` | 45 | 11 | 5.50% |
| `missing_package_delivered_not_received` | 45 | 11 | 5.50% |
| `order_cancellation_request` | 45 | 11 | 5.50% |
| `order_tracking_delivery_status` | 45 | 11 | 5.50% |
| `return_exchange_inquiry` | 45 | 11 | 5.50% |
| `third_party_seller_issue` | 45 | 11 | 5.50% |
| `wrong_item_received` | 45 | 11 | 5.50% |
| **Total** | **800** | **200** | **100.00%** |

---

## 4. Annotation Schema & Files

The 200-example golden dataset is stored in both CSV and Parquet formats:
- `data/golden/amazonhelp_gold_200_annotation.csv`
- `data/golden/amazonhelp_gold_200_annotation.parquet`

### Field Definitions
- `case_id`: Unique historical conversation identifier (e.g. `RAG-AMAZON-282480`).
- `customer_text`: Initial customer problem statement text.
- `suggested_intent`: Non-binding reference intent label from automated heuristic discovery.
- `final_human_intent`: Target label field (**Set to `NULL` for all rows**).
- `annotator_notes`: Optional notes field for annotator comments (**Set to empty for all rows**).

---

## 5. Human Annotation Guidelines & Interactive CLI Tool

### Annotation Guidelines
Created [`reports/phase_5_human_annotation_guide.md`](file:///D:/hiver-support-agent/reports/phase_5_human_annotation_guide.md) containing:
- All 17 intent definitions, inclusion criteria, exclusion criteria, and actual AmazonHelp text examples.
- Single-label primary action rules for multi-intent customer queries.
- Tie-breaking rules for neighboring categories (e.g. Tracking vs Late Delivery, Refund Status vs Return Inquiry).

### Interactive CLI Tool
Created [`scripts/annotate_amazonhelp_gold.py`](file:///D:/hiver-support-agent/scripts/annotate_amazonhelp_gold.py) featuring:
- Interactive display of customer text, case ID, and reference suggested intent.
- Numbered menu selection (1–17) for the approved intent categories.
- Strict validation ensuring only valid intent names are saved.
- Immediate incremental progress saving after every decision.
- Resume support (automatically skips previously annotated cases).

---

## 6. Files Created & Modified

### Created Files
1. `scripts/create_200_annotation_subset.py`: Stratified deterministic 200-case dataset sampler.
2. `scripts/annotate_amazonhelp_gold.py`: Interactive CLI human annotation tool.
3. `data/golden/amazonhelp_gold_200_annotation.parquet`: 200-case dataset (Parquet format).
4. `data/golden/amazonhelp_gold_200_annotation.csv`: 200-case dataset (CSV format).
5. `reports/phase_5_human_annotation_guide.md`: Human annotation guidelines document.
6. `tests/test_phase5_5.py`: Pytest unit test suite for Phase 5.5 annotation workflow.
7. `reports/phase_5_5_completion.md`: Mandatory completion report.

---

## 7. Automated Test Results

- **Command**: `pytest tests/ -v`
- **Result**: **35 passed in 17.96s**
- **Test Modules**:
  - `tests/test_amazonhelp_threads.py`: 5 passed
  - `tests/test_config.py`: 2 passed
  - `tests/test_loader.py`: 3 passed
  - `tests/test_phase4.py`: 6 passed
  - `tests/test_phase5.py`: 8 passed
  - `tests/test_phase5_5.py`: 4 passed
  - `tests/test_rag_and_retrieval.py`: 4 passed
  - `tests/test_schema.py`: 3 passed

---

## 8. Exact Commands

```bash
# 1. Generate deterministic 200-example annotation dataset
python scripts/create_200_annotation_subset.py

# 2. Launch interactive CLI human annotation tool
python scripts/annotate_amazonhelp_gold.py

# 3. Run automated unit test suite
pytest tests/ -v
```

---

## 9. Mandatory Stop Condition

**STOPPED.** Human annotation workflow preparation is complete.  
- **Human annotation has NOT been performed by this phase.**  
- No gold intent labels were assigned automatically.  
- Classifier performance has NOT been evaluated on human gold labels.  
- Phase 6 has NOT been started.  

Awaiting explicit approval before proceeding to human annotation or subsequent phases.
