"""
Phase 12 Day 3: Rule Calibration and Targeted Ambiguity Resolution Evaluation Script.

Evaluates:
1. System A: Existing Production Baseline (Hybrid Rule + Embedding Prototype)
2. System B: Calibrated-Rule Classifier (Evidence Scoring without Ambiguity Resolver)
3. System C: Calibrated-Rule + Ambiguity Resolver Classifier

Dataset: 200 AmazonHelp Human Gold dataset split deterministically into 100 Dev and 100 Untouched Test (seed 42).
"""
import sys
import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from hiver_agent.classification.taxonomy import AMAZON_INTENTS
from hiver_agent.classification.interface import IntentClassifier, IntentPrediction
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from hiver_agent.classification.calibrated_rule_classifier import (
    CalibratedRuleIntentClassifier,
    RuleAmbiguityResolver,
    PATTERN_SPECIFICITY
)
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.analysis.intent_discovery import INTENT_PATTERNS
from scripts.evaluate_phase12_hierarchical_classifier import load_gold_records
from scripts.run_phase_5_6_experiments import HybridRuleEmbeddingClassifier, split_gold_dataset_50_50


class CalibratedHybridClassifier(IntentClassifier):
    """
    Hybrid Classifier combining Calibrated Rule Classifier with Embedding Prototypes.
    """
    def __init__(self, rule_clf: CalibratedRuleIntentClassifier, proto_clf: EmbeddingPrototypeClassifier, threshold: float = 0.75):
        self.rule_clf = rule_clf
        self.proto_clf = proto_clf
        self.threshold = threshold

    def predict(self, text: str) -> IntentPrediction:
        rule_pred = self.rule_clf.predict(text)
        rule_meta = rule_pred.metadata or {}

        # High rule confidence without conflict -> trust rule
        if rule_pred.confidence >= self.threshold and not rule_meta.get("conflict_detected", False) and rule_pred.intent != "general_inquiry_other":
            return rule_pred

        proto_pred = self.proto_clf.predict(text)

        if rule_pred.metadata.get("rule_evidence") == "none":
            return proto_pred

        # Blend rule and prototype distributions
        blended_probs = {}
        rule_weight = rule_pred.confidence
        proto_weight = max(0.20, 1.0 - rule_weight)

        for intent in AMAZON_INTENTS:
            r_p = rule_pred.probabilities.get(intent, 0.0)
            p_p = proto_pred.probabilities.get(intent, 0.0)
            blended_probs[intent] = rule_weight * r_p + proto_weight * p_p

        total_b = sum(blended_probs.values())
        norm_probs = {k: float(v / total_b) for k, v in blended_probs.items()}
        best_intent = max(norm_probs.items(), key=lambda x: x[1])[0]
        best_conf = float(norm_probs[best_intent])

        return IntentPrediction(
            text=text,
            intent=best_intent,
            confidence=round(best_conf, 4),
            probabilities=norm_probs,
            metadata={
                "blended": True,
                "rule_intent": rule_pred.intent,
                "proto_intent": proto_pred.intent,
                "rule_confidence": rule_pred.confidence,
                "proto_confidence": proto_pred.confidence,
                "conflict_detected": rule_pred.metadata.get("conflict_detected", False)
            }
        )


def evaluate_rule_quality_on_dev(gold_dev_recs: list) -> pd.DataFrame:
    """
    Step 2: Rule Quality Analysis on 100 Gold Dev examples.
    """
    rule_stats = []
    for intent_name, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            matches = []
            for r in gold_dev_recs:
                text = r["customer_text"].lower()
                if re.search(pat, text):
                    matches.append(r)

            match_count = len(matches)
            correct_count = sum(1 for m in matches if m["intent"] == intent_name)
            precision = (correct_count / match_count * 100.0) if match_count > 0 else 0.0

            incorrect_intents = [m["intent"] for m in matches if m["intent"] != intent_name]
            common_incorrect = ", ".join([f"{k}:{v}" for k, v in pd.Series(incorrect_intents).value_counts().head(2).to_dict().items()]) if incorrect_intents else "None"

            category = "HIGH" if precision >= 90.0 else ("MEDIUM" if precision >= 70.0 else "LOW")
            if match_count == 0:
                category = "UNTESTED_DEV"

            rule_stats.append({
                "intent": intent_name,
                "pattern": pat,
                "match_count": match_count,
                "correct_count": correct_count,
                "precision_pct": round(precision, 2),
                "category": category,
                "common_incorrect": common_incorrect
            })

    return pd.DataFrame(rule_stats)


def compute_confidence_buckets(y_true: list, preds: list) -> pd.DataFrame:
    """
    Analyze accuracy across confidence buckets.
    """
    buckets = [
        ("0.00-0.49", 0.0, 0.4999),
        ("0.50-0.59", 0.50, 0.5999),
        ("0.60-0.69", 0.60, 0.6999),
        ("0.70-0.79", 0.70, 0.7999),
        ("0.80-0.89", 0.80, 0.8999),
        ("0.90-1.00", 0.90, 1.00)
    ]
    rows = []
    for label, min_val, max_val in buckets:
        indices = [idx for idx, p in enumerate(preds) if min_val <= p.confidence <= max_val]
        count = len(indices)
        correct = sum(1 for idx in indices if y_true[idx] == preds[idx].intent)
        acc = (correct / count * 100.0) if count > 0 else 0.0
        rows.append({
            "confidence_bucket": label,
            "count": count,
            "correct": correct,
            "accuracy_pct": round(acc, 2)
        })
    return pd.DataFrame(rows)


def run_experiment():
    print("=== Phase 12 Day 3: Rule Calibration & Ambiguity Resolution Evaluation ===", flush=True)

    # 1. Load Gold Records and split into 100 Dev / 100 Untouched Test (seed 42)
    df_gold, gold_records = load_gold_records()
    gold_dev_recs, gold_test_recs = split_gold_dataset_50_50(gold_records, seed=42)

    y_test_true = [r["intent"] for r in gold_test_recs]
    test_texts = [r["customer_text"] for r in gold_test_recs]

    print(f"Loaded Gold Dataset: Dev={len(gold_dev_recs)}, Untouched Test={len(gold_test_recs)}", flush=True)

    # 2. Rule Quality Analysis on Dev
    df_rule_quality = evaluate_rule_quality_on_dev(gold_dev_recs)
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_rule_quality.to_csv(tables_dir / "phase_12_rule_quality.csv", index=False)
    print(f"Saved Rule Quality Analysis to reports/tables/phase_12_rule_quality.csv", flush=True)

    # 3. Fit Embedding Prototype Classifier on 100 Dev Set
    print("\nFitting Embedding Prototype Classifier on Gold Dev Set...", flush=True)
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    proto_clf = EmbeddingPrototypeClassifier(embedder)
    dev_texts = [r["customer_text"] for r in gold_dev_recs]
    dev_labels = [r["intent"] for r in gold_dev_recs]
    proto_clf.fit(dev_texts, dev_labels)

    # 4. Instantiate Systems
    # System A: Existing Baseline (Hybrid Rule + Embedding Prototype)
    baseline_rule_clf = RuleBasedIntentClassifier()
    sys_a_baseline = HybridRuleEmbeddingClassifier(baseline_rule_clf, proto_clf)

    # System B: Calibrated Rule + Embedding Hybrid (Without Ambiguity Resolver)
    calib_rule_no_resolver = CalibratedRuleIntentClassifier(use_ambiguity_resolver=False)
    sys_b_calibrated = CalibratedHybridClassifier(calib_rule_no_resolver, proto_clf)

    # System C: Calibrated Rule + Ambiguity Resolver + Embedding Hybrid
    calib_rule_with_resolver = CalibratedRuleIntentClassifier(use_ambiguity_resolver=True)
    sys_c_resolver = CalibratedHybridClassifier(calib_rule_with_resolver, proto_clf)

    # 5. Evaluate Predictions on 100 Untouched Test Examples
    print("\nEvaluating Systems on 100 Untouched Gold Test Examples...", flush=True)
    preds_a = [sys_a_baseline.predict(t) for t in test_texts]
    preds_b = [sys_b_calibrated.predict(t) for t in test_texts]
    preds_c = [sys_c_resolver.predict(t) for t in test_texts]

    y_pred_a = [p.intent for p in preds_a]
    y_pred_b = [p.intent for p in preds_b]
    y_pred_c = [p.intent for p in preds_c]

    # Metrics
    prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(y_test_true, y_pred_a, average='macro', zero_division=0)
    w_prec_a, w_rec_a, w_f1_a, _ = precision_recall_fscore_support(y_test_true, y_pred_a, average='weighted', zero_division=0)
    acc_a = accuracy_score(y_test_true, y_pred_a)

    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(y_test_true, y_pred_b, average='macro', zero_division=0)
    w_prec_b, w_rec_b, w_f1_b, _ = precision_recall_fscore_support(y_test_true, y_pred_b, average='weighted', zero_division=0)
    acc_b = accuracy_score(y_test_true, y_pred_b)

    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(y_test_true, y_pred_c, average='macro', zero_division=0)
    w_prec_c, w_rec_c, w_f1_c, _ = precision_recall_fscore_support(y_test_true, y_pred_c, average='weighted', zero_division=0)
    acc_c = accuracy_score(y_test_true, y_pred_c)

    # High-confidence error analysis (conf >= 0.70)
    high_conf_errors_a = sum(1 for idx in range(len(y_test_true)) if preds_a[idx].confidence >= 0.70 and y_pred_a[idx] != y_test_true[idx])
    high_conf_errors_b = sum(1 for idx in range(len(y_test_true)) if preds_b[idx].confidence >= 0.70 and y_pred_b[idx] != y_test_true[idx])
    high_conf_errors_c = sum(1 for idx in range(len(y_test_true)) if preds_c[idx].confidence >= 0.70 and y_pred_c[idx] != y_test_true[idx])

    summary_rows = [
        {
            "Method": "System A: Existing Hybrid Baseline",
            "Dataset": "Untouched Gold Test (N=100)",
            "Accuracy": round(acc_a, 4),
            "Macro F1": round(f1_a, 4),
            "Weighted F1": round(w_f1_a, 4),
            "High_Conf_Errors_ge_0.70": high_conf_errors_a
        },
        {
            "Method": "System B: Calibrated Rule (No Resolver)",
            "Dataset": "Untouched Gold Test (N=100)",
            "Accuracy": round(acc_b, 4),
            "Macro F1": round(f1_b, 4),
            "Weighted F1": round(w_f1_b, 4),
            "High_Conf_Errors_ge_0.70": high_conf_errors_b
        },
        {
            "Method": "System C: Calibrated Rule + Ambiguity Resolver",
            "Dataset": "Untouched Gold Test (N=100)",
            "Accuracy": round(acc_c, 4),
            "Macro F1": round(f1_c, 4),
            "Weighted F1": round(w_f1_c, 4),
            "High_Conf_Errors_ge_0.70": high_conf_errors_c
        }
    ]

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(tables_dir / "phase_12_rule_calibration_results.csv", index=False)
    print("\n--- Summary Results ---")
    print(df_summary.to_string())

    # Confidence Buckets Comparison
    df_conf_a = compute_confidence_buckets(y_test_true, preds_a)
    df_conf_c = compute_confidence_buckets(y_test_true, preds_c)
    df_conf_comp = pd.merge(df_conf_a, df_conf_c, on="confidence_bucket", suffixes=("_baseline", "_calibrated"))
    df_conf_comp.to_csv(tables_dir / "phase_12_confidence_comparison.csv", index=False)

    # Per Intent Breakdown
    report_a = classification_report(y_test_true, y_pred_a, labels=AMAZON_INTENTS, output_dict=True, zero_division=0)
    report_c = classification_report(y_test_true, y_pred_c, labels=AMAZON_INTENTS, output_dict=True, zero_division=0)

    per_intent_rows = []
    for intent in AMAZON_INTENTS:
        supp = report_a[intent]["support"]
        r_a = report_a[intent]["recall"]
        f1_a_val = report_a[intent]["f1-score"]
        r_c = report_c[intent]["recall"]
        f1_c_val = report_c[intent]["f1-score"]
        per_intent_rows.append({
            "intent": intent,
            "test_support": supp,
            "baseline_recall": round(r_a, 4),
            "baseline_f1": round(f1_a_val, 4),
            "calibrated_recall": round(r_c, 4),
            "calibrated_f1": round(f1_c_val, 4),
            "recall_delta": round(r_c - r_a, 4)
        })

    df_per_intent = pd.DataFrame(per_intent_rows)
    df_per_intent.to_csv(tables_dir / "phase_12_ambiguity_results.csv", index=False)
    print("\n--- Per-Intent Recall & F1 Comparison ---")
    print(df_per_intent.to_string())

    # Confusion matrix
    conf_matrix_c = pd.crosstab(
        pd.Series(y_test_true, name="Gold Intent"),
        pd.Series(y_pred_c, name="Predicted Intent"),
        dropna=False
    )
    conf_matrix_c.to_csv(tables_dir / "phase_12_calibrated_confusion_matrix.csv")

    print(f"\nAll Day 3 tables saved under {tables_dir}/")


if __name__ == "__main__":
    run_experiment()
