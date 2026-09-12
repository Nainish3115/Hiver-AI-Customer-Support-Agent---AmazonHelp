"""
Phase 12 Day 4: Gold Prototype Intent Classifier Evaluation Script.

Evaluates:
1. Baseline Production Classifier (Hybrid Rule + Prototype)
2. Clean Gold Dev Prototype (Variant A: Mean Centroid)
3. Prototype Variant B (Medoid Centroid)
4. Prototype Variant C (Multi-Prototype Sub-clustering)
5. Rule + Prototype Hybrid (Calibrated Rule + Prototype Semantic Search)

Dataset: 200 AmazonHelp Human Gold split into 100 Dev and 100 Untouched Test (seed 42).
"""
import sys
import os
import time
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

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from hiver_agent.classification.taxonomy import AMAZON_INTENTS
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from hiver_agent.classification.calibrated_rule_classifier import CalibratedRuleIntentClassifier
from hiver_agent.classification.gold_prototype_classifier import (
    GoldPrototypeIntentClassifier,
    CalibratedRulePrototypeHybrid
)
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from scripts.evaluate_phase12_hierarchical_classifier import load_gold_records
from scripts.run_phase_5_6_experiments import HybridRuleEmbeddingClassifier, split_gold_dataset_50_50


def compute_margin_buckets(y_true: list, preds: list) -> pd.DataFrame:
    """
    Analyze accuracy across prototype margin buckets.
    """
    buckets = [
        ("Margin < 0.03", 0.0, 0.0299),
        ("Margin 0.03-0.07", 0.03, 0.0799),
        ("Margin 0.08-0.12", 0.08, 0.1299),
        ("Margin >= 0.13", 0.13, 1.00)
    ]
    rows = []
    for label, min_val, max_val in buckets:
        indices = [
            idx for idx, p in enumerate(preds)
            if min_val <= (p.metadata or {}).get("margin", p.confidence) <= max_val
        ]
        count = len(indices)
        correct = sum(1 for idx in indices if y_true[idx] == preds[idx].intent)
        acc = (correct / count * 100.0) if count > 0 else 0.0
        rows.append({
            "margin_bucket": label,
            "count": count,
            "correct": correct,
            "accuracy_pct": round(acc, 2)
        })
    return pd.DataFrame(rows)


def run_experiment():
    print("=== Phase 12 Day 4: Gold Prototype Intent Classifier Evaluation ===", flush=True)

    # 1. Load Gold Records and split 100 Dev / 100 Untouched Test (seed 42)
    df_gold, gold_records = load_gold_records()
    gold_dev_recs, gold_test_recs = split_gold_dataset_50_50(gold_records, seed=42)

    dev_texts = [r["customer_text"] for r in gold_dev_recs]
    dev_labels = [r["intent"] for r in gold_dev_recs]
    test_texts = [r["customer_text"] for r in gold_test_recs]
    test_labels = [r["intent"] for r in gold_test_recs]

    print(f"Loaded Gold Dataset: Dev={len(gold_dev_recs)}, Untouched Test={len(gold_test_recs)}", flush=True)

    # Measure latency
    t_start_embedder = time.time()
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    t_embedder_load = time.time() - t_start_embedder

    # 2. Fit Prototype Models on Gold Dev ONLY
    t_start_fit = time.time()
    proto_variant_a = GoldPrototypeIntentClassifier(embedder, variant_mode="mean")
    proto_variant_a.fit(dev_texts, dev_labels)

    proto_variant_b = GoldPrototypeIntentClassifier(embedder, variant_mode="medoid")
    proto_variant_b.fit(dev_texts, dev_labels)

    proto_variant_c = GoldPrototypeIntentClassifier(embedder, variant_mode="multi")
    proto_variant_c.fit(dev_texts, dev_labels)
    t_fit_time = time.time() - t_start_fit

    # Report Dev Support per intent
    print("\n--- Dev Support Count per Intent ---")
    for intent, count in proto_variant_a.dev_support.items():
        print(f"  {intent}: {count} dev examples")

    # 3. Fit Baseline and Hybrid Models
    baseline_rule_clf = RuleBasedIntentClassifier()
    legacy_proto_clf = EmbeddingPrototypeClassifier(embedder)
    legacy_proto_clf.fit(dev_texts, dev_labels)
    baseline_hybrid = HybridRuleEmbeddingClassifier(baseline_rule_clf, legacy_proto_clf)

    calib_rule_clf = CalibratedRuleIntentClassifier(use_ambiguity_resolver=True)
    rule_proto_hybrid = CalibratedRulePrototypeHybrid(calib_rule_clf, proto_variant_a)

    # 4. Predict on Untouched Test set and measure inference latency
    t_start_infer = time.time()
    preds_baseline = [baseline_hybrid.predict(t) for t in test_texts]
    preds_var_a = [proto_variant_a.predict(t) for t in test_texts]
    preds_var_b = [proto_variant_b.predict(t) for t in test_texts]
    preds_var_c = [proto_variant_c.predict(t) for t in test_texts]
    preds_hybrid = [rule_proto_hybrid.predict(t) for t in test_texts]
    t_infer_total = time.time() - t_start_infer
    avg_latency_ms = (t_infer_total / (5 * len(test_texts))) * 1000.0

    print(f"\nLatency Benchmark: Model Load = {t_embedder_load:.2f}s, Fit = {t_fit_time:.2f}s, Avg Inference = {avg_latency_ms:.2f} ms/query")

    # Evaluate Metrics
    systems = [
        ("System A: Existing Baseline Hybrid", preds_baseline),
        ("Clean Gold Dev Prototype (Variant A - Mean)", preds_var_a),
        ("Prototype Variant B (Medoid Centroid)", preds_var_b),
        ("Prototype Variant C (Multi-Prototype)", preds_var_c),
        ("Rule + Prototype Hybrid (Calibrated Rule + Proto)", preds_hybrid)
    ]

    summary_rows = []
    for name, preds in systems:
        y_pred = [p.intent for p in preds]
        acc = accuracy_score(test_labels, y_pred)
        _, _, f1_macro, _ = precision_recall_fscore_support(test_labels, y_pred, average="macro", zero_division=0)
        _, _, f1_weighted, _ = precision_recall_fscore_support(test_labels, y_pred, average="weighted", zero_division=0)
        high_conf_errs = sum(1 for idx in range(len(test_labels)) if preds[idx].confidence >= 0.70 and y_pred[idx] != test_labels[idx])

        summary_rows.append({
            "Method": name,
            "Dataset": "Untouched Gold Test (N=100)",
            "Accuracy": round(acc, 4),
            "Macro F1": round(f1_macro, 4),
            "Weighted F1": round(f1_weighted, 4),
            "High_Conf_Errors_ge_0.70": high_conf_errs
        })

    df_summary = pd.DataFrame(summary_rows)
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(tables_dir / "phase_12_gold_prototype_results.csv", index=False)

    print("\n--- Summary Performance Table ---")
    print(df_summary.to_string())

    # 5. Margin Confidence Analysis for Variant A
    df_margin = compute_margin_buckets(test_labels, preds_var_a)
    df_margin.to_csv(tables_dir / "phase_12_gold_prototype_confidence.csv", index=False)

    # 6. Per Intent Metrics for Variant A vs Baseline vs Hybrid
    report_base = classification_report(test_labels, [p.intent for p in preds_baseline], labels=AMAZON_INTENTS, output_dict=True, zero_division=0)
    report_var_a = classification_report(test_labels, [p.intent for p in preds_var_a], labels=AMAZON_INTENTS, output_dict=True, zero_division=0)
    report_hybrid = classification_report(test_labels, [p.intent for p in preds_hybrid], labels=AMAZON_INTENTS, output_dict=True, zero_division=0)

    per_intent_rows = []
    for intent in AMAZON_INTENTS:
        supp = report_base[intent]["support"]
        per_intent_rows.append({
            "intent": intent,
            "test_support": supp,
            "baseline_recall": round(report_base[intent]["recall"], 4),
            "baseline_f1": round(report_base[intent]["f1-score"], 4),
            "prototype_recall": round(report_var_a[intent]["recall"], 4),
            "prototype_f1": round(report_var_a[intent]["f1-score"], 4),
            "hybrid_recall": round(report_hybrid[intent]["recall"], 4),
            "hybrid_f1": round(report_hybrid[intent]["f1-score"], 4)
        })

    df_per_intent = pd.DataFrame(per_intent_rows)
    df_per_intent.to_csv(tables_dir / "phase_12_gold_prototype_per_intent.csv", index=False)
    print("\n--- Per-Intent Recall & F1 Table ---")
    print(df_per_intent.to_string())

    # 7. Confusion Matrix for Variant A
    conf_matrix = pd.crosstab(
        pd.Series(test_labels, name="Gold Intent"),
        pd.Series([p.intent for p in preds_var_a], name="Predicted Intent"),
        dropna=False
    )
    conf_matrix.to_csv(tables_dir / "phase_12_gold_prototype_confusion_matrix.csv")

    print(f"\nAll Day 4 tables saved under {tables_dir}/")


if __name__ == "__main__":
    run_experiment()
