"""
Auxiliary Banking77 baseline evaluation script.
Evaluates embedding prototype classifier on standard Banking77 train/test split.
Usage:
    python scripts/evaluate_banking77_baseline.py
"""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
from hiver_agent.data.loader import load_banking77
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from hiver_agent.classification.evaluation import evaluate_classifier_predictions


def main():
    print("=== Step 5: Auxiliary Banking77 Baseline Experiment ===", flush=True)
    t0 = time.time()

    tr_path = "data/external/banking77/banking77_train.csv"
    te_path = "data/external/banking77/banking77_test.csv"

    df_tr, df_te = load_banking77(tr_path, te_path)

    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    clf = EmbeddingPrototypeClassifier(embedder)

    print(f"Fitting Banking77 prototype classifier on {len(df_tr)} train examples across 77 intents...", flush=True)
    clf.fit(df_tr['text'].tolist(), df_tr['label_text'].tolist())

    print(f"Predicting Banking77 test set ({len(df_te)} examples)...", flush=True)
    test_texts = df_te['text'].tolist()
    y_true = df_te['label_text'].tolist()

    preds = [p.intent for p in clf.predict_batch(test_texts)]

    labels = sorted(list(set(y_true)))
    metrics, df_per_class = evaluate_classifier_predictions(y_true, preds, labels)

    report_file = Path("reports/banking77_baseline.md")
    report_md = f"""# Banking77 Intent Classification Auxiliary Baseline Report

> **Project Phase**: Phase 4 — Auxiliary Benchmark Experiment  
> **Note**: This is an auxiliary domain benchmark on financial intent classification. It is evaluated strictly on the Banking77 train ({len(df_tr):,} rows) and test ({len(df_te):,} rows) split. It does NOT represent AmazonHelp support classification performance.

---

## 1. Banking77 Benchmark Performance Metrics

- **Overall Accuracy**: **{metrics['accuracy']}** (**{round(metrics['accuracy']*100, 2)}%**)
- **Macro F1 Score**: **{metrics['macro_f1']}**
- **Weighted F1 Score**: **{metrics['weighted_f1']}**
- **Macro Precision**: **{metrics['macro_precision']}**
- **Macro Recall**: **{metrics['macro_recall']}**
- **Total Test Examples**: **{len(df_te):,}** across 77 intent classes

---

## 2. Methodology & Model Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, L2 normalized).
- **Classification Method**: Cosine nearest-centroid prototype matching with softmax probability calibration.
- **Evaluation**: Preserved original Banking77 train/test split.
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Saved Banking77 Baseline Report to: {report_file}", flush=True)
    print(f"Banking77 experiment finished in {time.time()-t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
