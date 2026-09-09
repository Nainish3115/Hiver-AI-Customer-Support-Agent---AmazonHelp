# Banking77 Intent Classification Auxiliary Baseline Report

> **Project Phase**: Phase 4 — Auxiliary Benchmark Experiment  
> **Note**: This is an auxiliary domain benchmark on financial intent classification. It is evaluated strictly on the Banking77 train (9,993 rows) and test (3,076 rows) split. It does NOT represent AmazonHelp support classification performance.

---

## 1. Banking77 Benchmark Performance Metrics

- **Overall Accuracy**: **0.8501** (**85.01%**)
- **Macro F1 Score**: **0.8485**
- **Weighted F1 Score**: **0.8486**
- **Macro Precision**: **0.8618**
- **Macro Recall**: **0.85**
- **Total Test Examples**: **3,076** across 77 intent classes

---

## 2. Methodology & Model Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, L2 normalized).
- **Classification Method**: Cosine nearest-centroid prototype matching with softmax probability calibration.
- **Evaluation**: Preserved original Banking77 train/test split.
