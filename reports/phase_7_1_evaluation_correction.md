# Phase 7.1: Human Escalation Validation & Evaluation Correction Report

> **Project Phase**: Phase 7.1 — Escalation Evaluation Correction & Validation  
> **Human Validation Benchmark**: 50 Cases ([`data/golden/amazonhelp_escalation_validation_50.csv`](file:///D:/hiver-support-agent/data/golden/amazonhelp_escalation_validation_50.csv))  
> **Diagnostic Benchmark**: 200 Human Gold Cases ([`data/golden/amazonhelp_gold_200_annotation.csv`](file:///D:/hiver-support-agent/data/golden/amazonhelp_gold_200_annotation.csv))  
> **Status**: **EVALUATION STRUCTURE COMPLETE & VERIFIED**  

---

## 1. Why Phase 7.1 Was Necessary

In Phase 7, the `Operational Escalation Proxy` was derived from upstream model and retrieval signals (such as predicted intent mismatch or evidence quality). Because the decision engine evaluates related signals, the operational proxy is not independent human ground truth. 

Phase 7.1 explicitly separates:
1. **Model Routing Rate vs Safe Automation Rate**: The 84.5% figure reported in Phase 7 represents a model routing decision rate, **NOT** a human-verified safe-automation coverage rate.
2. **Diagnostic Proxy ($N=200$) vs Independent Human Validation ($N=50$)**: We preserve the 200-case proxy for diagnostic stress-testing while creating an independent 50-case human escalation validation dataset.

---

## 2. Canonical Configuration

The canonical executable configuration (matching `configs/config.yaml` and `src/hiver_agent/config.py`) is:

```yaml
decision:
  risk_threshold: 0.35
  confidence_threshold: 0.60
  min_retrieval_score: 0.70
  weights:
    intent_uncertainty: 0.40
    retrieval_quality: 0.30
    category_risk: 0.30
```

---

## 3. Operational Escalation Proxy Evaluation (N=200 Benchmark)

> **IMPORTANT DISCLAIMER**: The operational proxy is not independent human ground truth. It is derived partly from upstream model and retrieval signals and is used solely for diagnostic stress-testing.

| Policy Approach                       | Model Auto-Handle Rate   | Model Escalation Rate   | False Auto-Handle (Proxy)   | False Escalation (Proxy)   |
|:--------------------------------------|:-------------------------|:------------------------|:----------------------------|:---------------------------|
| Baseline 1a: Confidence (th=0.50)     | 92.0%                    | 8.0%                    | 46.0%                       | 0.5%                       |
| Baseline 1b: Confidence (th=0.60)     | 91.5%                    | 8.5%                    | 46.0%                       | 1.0%                       |
| Baseline 1c: Confidence (th=0.70)     | 91.5%                    | 8.5%                    | 46.0%                       | 1.0%                       |
| Baseline 1d: Confidence (th=0.80)     | 91.5%                    | 8.5%                    | 46.0%                       | 1.0%                       |
| Baseline 2: Hard Safety Rules Only    | 100.0%                   | 0.0%                    | 53.5%                       | 0.0%                       |
| Proposed: Risk-Aware Policy (th=0.35) | 84.5%                    | 15.5%                   | 41.0%                       | 3.0%                       |

---

## 4. Human-Verified Escalation Evaluation (N=50 Validation Set)

> **Human Validation Set**: `data/golden/amazonhelp_escalation_validation_50.csv`  
> **Status**: **PENDING MANUAL ANNOTATION** (Blank label guard active)

| Status                   | Message                                                                                 |
|:-------------------------|:----------------------------------------------------------------------------------------|
| PENDING HUMAN ANNOTATION | data/golden/amazonhelp_escalation_validation_50.csv requires should_auto_handle values. |

### Confusion Matrix (N=50)

| Status                   | Message                                                                                 |
|:-------------------------|:----------------------------------------------------------------------------------------|
| PENDING HUMAN ANNOTATION | data/golden/amazonhelp_escalation_validation_50.csv requires should_auto_handle values. |

---

## 5. Failure Analysis & Future Mitigations

Top observed decision layer issues (from diagnostic proxy and edge cases):
1. **Classifier High-Confidence Misclassification**: Classifier predicts wrong intent with high confidence, lowering composite risk score.
   - *Mitigation (Future Phase)*: Require higher confidence threshold for auto-handling or cross-intent retrieval entropy checks.
2. **Top-1 Retrieval Score Threshold Bypass**: Retrieval similarity exceeds `0.70` despite subtle topic shift.
   - *Mitigation (Future Phase)*: Enforce strict intent-filtered retrieval matching.

---

## 6. Limitations & Unverified Items

1. **Single-Annotator Validation**: Unless a second human annotator completes `annotator_2_decision`, inter-annotator agreement (Cohen's Kappa) is not claimed.
2. **No Production API/UI Scope Expansion**: The decision engine remains a clean Python layer awaiting API/UI integration in future phases.
