"""
Phase 12 Day 2: Hierarchical Intent Classifier Evaluation Script.

Evaluates two-stage hierarchical classification vs baseline on 100 Untouched Gold Test cases:
- Stage 1: Parent Domain Classifier performance
- Stage 2: Child Intent Classifier performance (conditional on correct parent)
- End-to-End: 17-intent classification performance (Accuracy, Macro F1, Weighted F1)
- Focus Intent Recalls & Specific Confusion Pair Inspection

Strict Leakage Controls:
- Training: ONLY 100 Gold Dev Set cases (seed=42)
- Evaluation: ONLY 100 Untouched Gold Test Set cases

Generates:
- reports/tables/phase_12_hierarchical_results.csv
- reports/tables/phase_12_hierarchical_per_intent.csv
- reports/tables/phase_12_hierarchical_confusion_matrix.csv
"""
import sys
import json
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)
sys.path.insert(0, str(Path(root_dir) / "src"))

from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from hiver_agent.classification.tfidf_classifier import TfidfIntentClassifier
from hiver_agent.classification.hierarchical_classifier import (
    HierarchicalIntentClassifier,
    PARENT_HIERARCHY,
    INTENT_TO_PARENT
)
from hiver_agent.classification.taxonomy import AMAZON_INTENTS
from scripts.run_phase_5_6_experiments import HybridRuleEmbeddingClassifier, split_gold_dataset_50_50


def load_gold_records():
    gold_csv = Path(root_dir) / "data/golden/amazonhelp_gold_200_annotation.csv"
    df_gold = pd.read_csv(gold_csv, encoding="utf-8")
    records = df_gold.to_dict(orient="records")
    for r in records:
        r["case_id"] = str(r["case_id"])
        r["customer_text"] = str(r["customer_text"])
        r["intent"] = str(r["final_human_intent"])
    return df_gold, records


def evaluate_classifier_on_records(clf, records):
    y_true = [r["intent"] for r in records]
    y_pred = []
    confidences = []
    
    for r in records:
        pred = clf.predict(r["customer_text"])
        y_pred.append(pred.intent)
        confidences.append(pred.confidence)

    acc = round(accuracy_score(y_true, y_pred), 4)
    macro_f1 = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)
    weighted_f1 = round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confidences,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def main():
    print("=== Phase 12 Day 2: Two-Stage Hierarchical Intent Classifier Evaluation ===", flush=True)

    # 1. Load Gold Records and Split 50/50 (Dev=100, Untouched Test=100)
    df_gold, gold_records = load_gold_records()
    gold_dev_recs, gold_test_recs = split_gold_dataset_50_50(gold_records, seed=42)

    print(f"Loaded {len(gold_records)} Human Gold benchmark cases.", flush=True)
    print(f"Data Split: Gold Dev = {len(gold_dev_recs)} cases (TRAINING) | Untouched Gold Test = {len(gold_test_recs)} cases (EVALUATION ONLY)", flush=True)

    dev_texts = [r["customer_text"] for r in gold_dev_recs]
    dev_labels = [r["intent"] for r in gold_dev_recs]

    test_texts = [r["customer_text"] for r in gold_test_recs]
    test_labels = [r["intent"] for r in gold_test_recs]

    # Report Dev Training Examples per Parent Domain
    dev_parent_counts = Counter([INTENT_TO_PARENT[l] for l in dev_labels])
    print("\n--- Dev Training Examples per Parent Domain ---", flush=True)
    for parent_domain, children in PARENT_HIERARCHY.items():
        cnt = dev_parent_counts[parent_domain]
        print(f"   - {parent_domain} ({len(children)} child intents): {cnt} dev training examples", flush=True)

    # 2. Fit Baseline Pipeline Classifier (Hybrid Rule + Proto fit on Dev 100)
    print("\nFitting Baseline Pipeline Classifier on Dev 100...", flush=True)
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2", batch_size=128)
    rule_clf = RuleBasedIntentClassifier()
    proto_clf = EmbeddingPrototypeClassifier(embedder)
    proto_clf.fit(dev_texts, dev_labels)
    hybrid_baseline_clf = HybridRuleEmbeddingClassifier(rule_clf, proto_clf)

    # 3. Fit Flat TF-IDF Classifier on Dev 100
    print("Fitting Flat TF-IDF Classifier on Dev 100...", flush=True)
    tfidf_baseline_clf = TfidfIntentClassifier(c_param=1.0)
    tfidf_baseline_clf.fit(dev_texts, dev_labels)

    # 4. Fit Hierarchical Classifier on Dev 100 ONLY
    print("Fitting Hierarchical Intent Classifier on Dev 100 ONLY...", flush=True)
    hier_clf = HierarchicalIntentClassifier(c_param=1.0, max_features=3000)
    hier_clf.fit(dev_texts, dev_labels)

    # 5. Evaluate Stage 1 (Parent Domain Classifier) on Untouched Test Set
    test_parent_true = [INTENT_TO_PARENT[l] for l in test_labels]
    test_parent_pred = []
    test_parent_conf = []

    for t in test_texts:
        p_name, p_conf, _ = hier_clf.predict_parent(t)
        test_parent_pred.append(p_name)
        test_parent_conf.append(p_conf)

    parent_acc = round(accuracy_score(test_parent_true, test_parent_pred), 4)
    parent_macro_f1 = round(f1_score(test_parent_true, test_parent_pred, average="macro", zero_division=0), 4)
    parent_weighted_f1 = round(f1_score(test_parent_true, test_parent_pred, average="weighted", zero_division=0), 4)

    print("\n==================================================", flush=True)
    print("STAGE 1: PARENT DOMAIN CLASSIFIER (UNTOUCHED TEST SET N=100)", flush=True)
    print("==================================================", flush=True)
    print(f"Parent Accuracy   : {parent_acc*100:.2f}% ({sum(1 for a, b in zip(test_parent_true, test_parent_pred) if a == b)}/100)", flush=True)
    print(f"Parent Macro F1   : {parent_macro_f1:.4f}", flush=True)
    print(f"Parent Weighted F1: {parent_weighted_f1:.4f}", flush=True)

    # 6. Evaluate Stage 2 (Child Intent Classifier Conditional on Correct Parent)
    correct_parent_indices = [i for i, (gt_p, pr_p) in enumerate(zip(test_parent_true, test_parent_pred)) if gt_p == pr_p]
    child_conditional_correct = 0
    
    for i in correct_parent_indices:
        pred_intent = hier_clf.predict(test_texts[i]).intent
        if pred_intent == test_labels[i]:
            child_conditional_correct += 1

    child_conditional_acc = round(child_conditional_correct / len(correct_parent_indices), 4) if correct_parent_indices else 0.0

    print("\n==================================================", flush=True)
    print("STAGE 2: CHILD CLASSIFIER CONDITIONAL ON CORRECT PARENT", flush=True)
    print("==================================================", flush=True)
    print(f"Correct Parent Subsample Size: {len(correct_parent_indices)} / 100 cases", flush=True)
    print(f"Conditional Child Accuracy   : {child_conditional_acc*100:.2f}% ({child_conditional_correct} / {len(correct_parent_indices)})", flush=True)

    # 7. Evaluate End-to-End Classifiers on Untouched Gold Test Set (N=100)
    print("\n==================================================", flush=True)
    print("END-TO-END EVALUATION ON UNTOUCHED GOLD TEST SET (N=100)", flush=True)
    print("==================================================", flush=True)

    res_baseline = evaluate_classifier_on_records(hybrid_baseline_clf, gold_test_recs)
    res_tfidf = evaluate_classifier_on_records(tfidf_baseline_clf, gold_test_recs)
    res_hier = evaluate_classifier_on_records(hier_clf, gold_test_recs)

    print(f"\n1. Hybrid Rule+Proto Baseline (Current Pipeline):", flush=True)
    print(f"   - Untouched Test Accuracy   : {res_baseline['accuracy']*100:.2f}%", flush=True)
    print(f"   - Untouched Test Macro F1   : {res_baseline['macro_f1']:.4f}", flush=True)
    print(f"   - Untouched Test Weighted F1: {res_baseline['weighted_f1']:.4f}", flush=True)

    print(f"\n2. Flat TF-IDF Baseline:", flush=True)
    print(f"   - Untouched Test Accuracy   : {res_tfidf['accuracy']*100:.2f}%", flush=True)
    print(f"   - Untouched Test Macro F1   : {res_tfidf['macro_f1']:.4f}", flush=True)
    print(f"   - Untouched Test Weighted F1: {res_tfidf['weighted_f1']:.4f}", flush=True)

    print(f"\n3. Two-Stage Hierarchical Intent Classifier (Experimental Candidate):", flush=True)
    print(f"   - Untouched Test Accuracy   : {res_hier['accuracy']*100:.2f}%", flush=True)
    print(f"   - Untouched Test Macro F1   : {res_hier['macro_f1']:.4f}", flush=True)
    print(f"   - Untouched Test Weighted F1: {res_hier['weighted_f1']:.4f}", flush=True)

    # 8. Focus Intent Recall Comparison (5 Key Intents)
    focus_intents = [
        "general_inquiry_other",
        "late_delivery_shipping_delay",
        "customer_service_complaint",
        "payment_billing_issue",
        "missing_package_delivered_not_received"
    ]

    p_base, r_base, f1_base, supp_test = precision_recall_fscore_support(test_labels, res_baseline["y_pred"], labels=AMAZON_INTENTS, zero_division=0)
    p_hier, r_hier, f1_hier, _ = precision_recall_fscore_support(test_labels, res_hier["y_pred"], labels=AMAZON_INTENTS, zero_division=0)

    intent_map_base = {intent: (p_base[i], r_base[i], f1_base[i], supp_test[i]) for i, intent in enumerate(AMAZON_INTENTS)}
    intent_map_hier = {intent: (p_hier[i], r_hier[i], f1_hier[i], supp_test[i]) for i, intent in enumerate(AMAZON_INTENTS)}

    print("\n==================================================", flush=True)
    print("5 KEY FOCUS INTENTS RECALL COMPARISON (UNTOUCHED TEST SET)", flush=True)
    print("==================================================", flush=True)
    for fi in focus_intents:
        rb = intent_map_base[fi][1]
        rh = intent_map_hier[fi][1]
        fb = intent_map_base[fi][2]
        fh = intent_map_hier[fi][2]
        sp = intent_map_base[fi][3]
        delta_r = (rh - rb) * 100
        print(f"   - {fi} (Support={sp}): Baseline Recall={rb*100:.1f}% -> Hierarchical Recall={rh*100:.1f}% (Δ = {delta_r:+.1f}%) | F1: {fb:.4f} -> {fh:.4f}", flush=True)

    # 9. Specific Confusion Pair Comparison
    specific_pairs = [
        ("late_delivery_shipping_delay", "order_tracking_delivery_status"),
        ("customer_service_complaint", "damaged_defective_item"),
        ("general_inquiry_other", "digital_content_kindle_video")
    ]

    print("\n==================================================", flush=True)
    print("SPECIFIC CONFUSION PAIR INSPECTION (UNTOUCHED TEST SET)", flush=True)
    print("==================================================", flush=True)
    for g_intent, p_intent in specific_pairs:
        c_base = sum(1 for gt, pt in zip(test_labels, res_baseline["y_pred"]) if gt == g_intent and pt == p_intent)
        c_hier = sum(1 for gt, pt in zip(test_labels, res_hier["y_pred"]) if gt == g_intent and pt == p_intent)
        print(f"   - {g_intent} -> {p_intent}: Baseline Errors={c_base} | Hierarchical Errors={c_hier}", flush=True)

    # 10. Generate CSV Reports
    tables_dir = Path(root_dir) / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Comparative Results CSV
    comp_df = pd.DataFrame([
        {
            "Method": "Hybrid Rule+Proto Baseline (Current Pipeline)",
            "Dataset": "Untouched Gold Test (N=100)",
            "Accuracy": res_baseline["accuracy"],
            "Macro F1": res_baseline["macro_f1"],
            "Weighted F1": res_baseline["weighted_f1"]
        },
        {
            "Method": "Flat TF-IDF Baseline",
            "Dataset": "Untouched Gold Test (N=100)",
            "Accuracy": res_tfidf["accuracy"],
            "Macro F1": res_tfidf["macro_f1"],
            "Weighted F1": res_tfidf["weighted_f1"]
        },
        {
            "Method": "Two-Stage Hierarchical Intent Classifier",
            "Dataset": "Untouched Gold Test (N=100)",
            "Accuracy": res_hier["accuracy"],
            "Macro F1": res_hier["macro_f1"],
            "Weighted F1": res_hier["weighted_f1"]
        }
    ])
    comp_df.to_csv(tables_dir / "phase_12_hierarchical_results.csv", index=False)

    # Per-Intent Performance CSV
    per_intent_rows = []
    for intent in AMAZON_INTENTS:
        pb, rb, fb, sp = intent_map_base[intent]
        ph, rh, fh, _ = intent_map_hier[intent]
        per_intent_rows.append({
            "intent": intent,
            "test_support": sp,
            "baseline_precision": round(pb, 4),
            "baseline_recall": round(rb, 4),
            "baseline_f1": round(fb, 4),
            "hierarchical_precision": round(ph, 4),
            "hierarchical_recall": round(rh, 4),
            "hierarchical_f1": round(fh, 4),
            "recall_delta": round(rh - rb, 4)
        })
    df_per_intent = pd.DataFrame(per_intent_rows)
    df_per_intent.to_csv(tables_dir / "phase_12_hierarchical_per_intent.csv", index=False)

    # Confusion Matrix CSV
    cm = pd.crosstab(
        pd.Series(test_labels, name="Gold Intent"),
        pd.Series(res_hier["y_pred"], name="Predicted Intent")
    ).reindex(index=AMAZON_INTENTS, columns=AMAZON_INTENTS, fill_value=0)
    cm.to_csv(tables_dir / "phase_12_hierarchical_confusion_matrix.csv")

    print("\nSaved Phase 12 CSV reports to reports/tables/", flush=True)


if __name__ == "__main__":
    main()
