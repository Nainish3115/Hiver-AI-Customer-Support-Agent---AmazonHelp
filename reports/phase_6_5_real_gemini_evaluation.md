# Controlled Real-LLM (Gemini) Response Generation Evaluation Report (N=50)

> **Evaluation Objective**: Validate actual Google Gemini generation & LLM-as-judge on N=50 sampled cases prior to Phase 7.  
> **Gold Sample**: **50 Cases** sampled with `seed=42` from `data/golden/amazonhelp_gold_200_annotation.csv`.  
> **Self-Case Exclusion**: Enforced (`self_leakage_detected`: 0).  
> **Provider**: `gemini` | **Model**: `gemini-3.6-flash` | **Mode**: `real`  
> **Status**: **COMPLETED & VERIFIED**  

---

## 1. Automated Metrics Comparison Across 3 Conditions

| Condition | Validity Rate | Safety Pass Rate | Unsupported Claim Rate | Unsupported Order ID Rate | Unsupported Monetary Amount Rate | Internal Term Leakage Rate | Exact Copy Rate | Insufficient Evidence Rate | Avg Length (chars) | Avg Latency (s) | Recall@1 | Recall@3 | Recall@5 | MRR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Condition A: Retrieval-Only Copy** | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 129.58 | 0.001 | 0.2800 | 0.4400 | 0.5400 | 0.3957 |
| **Condition B: Deterministic Synthesis** | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 136.58 | 0.001 | 0.2800 | 0.4400 | 0.5400 | 0.3957 |
| **Condition C: Real Gemini Generation** | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 122.18 | 1.620 | 0.2800 | 0.4400 | 0.5400 | 0.3957 |

---

## 2. Real Gemini LLM-as-Judge Evaluation (N=50)

- **Judge Status**: `COMPLETED`
- **Evaluated Count**: `50` / 50
- **Judge Failures**: `0`
- **Average Overall Score**: `4.72` / 5.0 (Median: `5.0`)
- **Average Groundedness**: `4.96` / 5.0 (Median: `5.0`)
- **Average Relevance**: `4.68` / 5.0 (Median: `5.0`)
- **Average Helpfulness**: `4.62` / 5.0 (Median: `5.0`)
- **Average Hallucination Risk**: `4.92` / 5.0 (5 = zero risk)
- **Average Tone**: `4.88` / 5.0 (Median: `5.0`)
- **Average Actionability**: `4.66` / 5.0 (Median: `5.0`)
- **Average Judge Latency**: `1.701s` per evaluation call

---

## 3. Failure Mode Breakdown (Real Gemini Condition)

```json
{
  "1. Wrong Retrieval": 18,
  "2. Wrong Intent": 7,
  "3. Poor Evidence Selection": 0,
  "4. Hallucinated/Unsupported Claim": 0,
  "5. Generic Response": 0,
  "6. Copied Mismatch": 0,
  "7. Unnecessary Info Request": 0,
  "8. Synthesis Error": 0,
  "9. Insufficient Evidence": 0,
  "10. Safety Check Violation": 0
}
```

---

## 4. Latency, Cost, and Runtime Summary

- **Generation Calls**: 50 real Gemini API calls
- **Judge Calls**: 50 real Gemini API calls
- **Avg Generation Latency**: `1.620s`
- **Avg Judge Latency**: `1.701s`
- **Token Usage / Cost**: Not directly exposed by the SDK for standard calls; estimated < 0.05 USD total.

---

## 5. 15-Case Manual Review Subset

The 15 representative inspection cases have been exported to [`reports/tables/phase_6_5_manual_review_15.csv`](file:///D:/hiver-support-agent/reports/tables/phase_6_5_manual_review_15.csv).

---

## 6. Critical Interpretation Guidelines & Bounds

1. **Validation Sample Boundary**: N=50 is a controlled validation sample to verify pipeline integration, safety checks, and judge behavior prior to Phase 7. It is not a definitive statistical estimate of full dataset performance.
2. **Deterministic Safety Checks**: Automated checks (regex entity detection, terminology leakage) provide necessary safety guardrails but do not guarantee complete factual correctness without LLM-as-judge and human auditing.
3. **Layer Separation**:
   $$\text{Intent Quality} \rightarrow \text{Retrieval Quality} \rightarrow \text{Generation Quality} \rightarrow \text{Safety Validation}$$
