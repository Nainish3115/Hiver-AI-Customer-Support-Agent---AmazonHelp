"""
Evaluation metrics, confusion matrix generator, confidence abstention analysis,
and error analysis utilities for intent classification.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from hiver_agent.classification.interface import IntentPrediction
from hiver_agent.classification.taxonomy import AMAZON_INTENTS


def evaluate_classifier_predictions(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Compute overall and per-class classification metrics.
    """
    if labels is None:
        labels = AMAZON_INTENTS

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)

    p_per, r_per, f1_per, supp_per = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class_rows = []
    for l, p, r, f1, s in zip(labels, p_per, r_per, f1_per, supp_per):
        per_class_rows.append({
            "intent": l,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "support": int(s)
        })

    df_per_class = pd.DataFrame(per_class_rows)

    metrics = {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "total_eval_samples": len(y_true)
    }

    return metrics, df_per_class


def extract_top_confusion_pairs(
    y_true: List[str],
    y_pred: List[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Identify and rank the most frequent misclassification pairs (True Intent vs Predicted Intent).
    """
    pairs = []
    for t, p in zip(y_true, y_pred):
        if t != p:
            pairs.append((t, p))

    counts = pd.Series(pairs).value_counts().head(top_n)

    rows = []
    for (t, p), cnt in counts.items():
        rows.append({
            "true_intent": t,
            "predicted_intent": p,
            "error_count": int(cnt)
        })

    return pd.DataFrame(rows)


def evaluate_confidence_abstention(
    predictions: List[IntentPrediction],
    y_true: List[str],
    thresholds: List[float] = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> pd.DataFrame:
    """
    Evaluate Coverage vs Accuracy and Macro F1 at varying confidence thresholds.
    """
    rows = []
    total = len(predictions)

    for th in thresholds:
        accepted = []
        accepted_y_true = []

        for p, target in zip(predictions, y_true):
            if p.confidence >= th:
                accepted.append(p.intent)
                accepted_y_true.append(target)

        coverage = len(accepted) / total if total > 0 else 0.0

        if len(accepted) > 0:
            acc = accuracy_score(accepted_y_true, accepted)
            _, _, f1_macro, _ = precision_recall_fscore_support(
                accepted_y_true, accepted, average='macro', zero_division=0
            )
        else:
            acc = 0.0
            f1_macro = 0.0

        rows.append({
            "confidence_threshold": th,
            "accepted_count": len(accepted),
            "total_count": total,
            "coverage": round(float(coverage), 4),
            "accepted_accuracy": round(float(acc), 4),
            "accepted_macro_f1": round(float(f1_macro), 4)
        })

    return pd.DataFrame(rows)


def extract_error_cases(
    predictions: List[IntentPrediction],
    y_true: List[str],
    records: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Extract detailed misclassified examples for qualitative failure analysis.
    """
    error_rows = []
    for p, target, r in zip(predictions, y_true, records):
        if p.intent != target:
            error_rows.append({
                "case_id": r.get("case_id", "RAG-UNKNOWN"),
                "customer_text": p.text,
                "true_intent": target,
                "predicted_intent": p.intent,
                "confidence": round(float(p.confidence), 4)
            })

    return pd.DataFrame(error_rows)


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    output_png_path: str
) -> None:
    """
    Generate and save a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("AmazonHelp Intent Classification Confusion Matrix")
    plt.xlabel("Predicted Intent")
    plt.ylabel("True Intent")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close()
