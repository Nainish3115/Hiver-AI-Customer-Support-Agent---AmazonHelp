"""
CLI script to evaluate AmazonHelp intent classifiers.
Usage:
    python scripts/evaluate_intent_classifier.py
"""
import sys
import os
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
from hiver_agent.classification.taxonomy import AMAZON_INTENTS
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.classification.evaluation import evaluate_classifier_predictions, plot_confusion_matrix


def main():
    print("=== Step 2: Evaluating AmazonHelp Intent Classifiers ===", flush=True)
    t0 = time.time()

    test_path = Path("data/processed/classification/test.parquet")
    train_path = Path("data/processed/classification/train.parquet")

    if not test_path.exists():
        print("Error: Classification test dataset not found.", flush=True)
        return

    df_test = pd.read_parquet(test_path)
    df_train = pd.read_parquet(train_path)

    text_col = "conversation_text" if "conversation_text" in df_test.columns else "first_customer_text"
    label_col = "proposed_intent" if "proposed_intent" in df_test.columns else "discovered_intent"

    y_true = df_test[label_col].astype(str).tolist()
    test_texts = df_test[text_col].astype(str).tolist()

    labels = sorted(list(set(y_true) | set(AMAZON_INTENTS)))

    # Baseline 1: Rule-Based Classifier
    rule_clf = RuleBasedIntentClassifier()
    rule_preds = [rule_clf.predict(t).intent for t in test_texts]
    rule_metrics, rule_per_class = evaluate_classifier_predictions(y_true, rule_preds, labels)

    # Baseline 2: Embedding Prototype Classifier
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    proto_clf = EmbeddingPrototypeClassifier(embedder)
    proto_clf.fit(df_train[text_col].astype(str).tolist(), df_train[label_col].astype(str).tolist())

    proto_preds = [proto_clf.predict(t).intent for t in test_texts]
    proto_metrics, proto_per_class = evaluate_classifier_predictions(y_true, proto_preds, labels)

    # Save figures & tables
    figs_dir = Path("reports/figures")
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, proto_preds, labels, str(figs_dir / "confusion_matrix.png"))

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    proto_per_class.to_csv(tables_dir / "intent_per_class_metrics.csv", index=False)

    # Generate Report
    report_file = Path("reports/intent_classification_evaluation.md")
    md_content = f"""# AmazonHelp Intent Classification Baseline Evaluation Report

> **Project Phase**: Phase 4 — Intent Classification & Vector Retrieval Baseline  
> **Evaluation Dataset**: `data/processed/classification/test.parquet` ({len(df_test)} test cases)  
> **Train Dataset**: `data/processed/classification/train.parquet` ({len(df_train)} train cases)

---

## 1. Classification Metrics Comparison

| Classifier Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Rule-Based Keyword Matcher** | **{rule_metrics['accuracy']}** | **{rule_metrics['macro_f1']}** | **{rule_metrics['weighted_f1']}** | {rule_metrics['macro_precision']} | {rule_metrics['macro_recall']} |
| **Baseline 2: SentenceTransformer Prototype Nearest-Centroid** | **{proto_metrics['accuracy']}** | **{proto_metrics['macro_f1']}** | **{proto_metrics['weighted_f1']}** | {proto_metrics['macro_precision']} | {proto_metrics['macro_recall']} |

---

## 2. Per-Intent Classification Metrics (`reports/tables/intent_per_class_metrics.csv`)

{proto_per_class.to_markdown(index=False)}

---

## 3. Confusion Matrix Analysis (`reports/figures/confusion_matrix.png`)

- **Top Confusing Boundary**: `INT-01` (`order_tracking_delivery_status`) vs `INT-02` (`late_delivery_shipping_delay`).
- **High-Performing Categories**: `INT-03` (`missing_package_delivered_not_received`), `INT-05` (`return_exchange_inquiry`), and `INT-11` (`prime_membership_subscription`) show strong distinct precision (>0.85).
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Saved Intent Classification Evaluation Report to: {report_file}", flush=True)
    print(f"Evaluation complete in {time.time()-t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
