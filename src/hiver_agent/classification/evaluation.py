"""
Evaluation metrics and confusion matrix generator for intent classification.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_classifier_predictions(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str]
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Compute overall and per-class classification metrics.
    """
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
