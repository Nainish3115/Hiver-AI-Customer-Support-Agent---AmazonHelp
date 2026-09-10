# Phase 6.5 Controlled Real-LLM (Gemini) Evaluation Completion Report

- **Real Gemini Generation Verification**: SUCCESS (50 real API calls completed)
- **Real Gemini LLM-as-Judge Verification**: SUCCESS (50 real API calls completed)
- **Sample**: N=50 sampled with `seed=42` from `data/golden/amazonhelp_gold_200_annotation.csv`
- **Provider**: `gemini`
- **Model**: `gemini-3.6-flash`
- **Mode**: `real`
- **Self-Case Exclusion**: Enforced (`self_leakage_detected`: 0)
- **Judge Completion Status**: `COMPLETED` (`50`/50 evaluated cleanly, 0 failures)
- **Mean Overall Judge Score**: `4.72` / 5.0 (Median: `5.0`)
- **Mean Groundedness Score**: `4.96` / 5.0 (Median: `5.0`)
- **Mean Hallucination Risk Score**: `4.92` / 5.0 (5 = zero risk)
- **Automated Safety Pass Rate**: `100.0%`
- **Manual Review File**: [`reports/tables/phase_6_5_manual_review_15.csv`](file:///D:/hiver-support-agent/reports/tables/phase_6_5_manual_review_15.csv)
- **Full Test Suite**: `52/52 PASSED`
